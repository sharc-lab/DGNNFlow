#include "utils.hpp"
#include "operators.hpp"
#include "message_passing.hpp"

// #include <iostream>
#include <fstream>

void message_passing(
    int peID,
    int numNodes,
    int layerNum,
    hls::stream<node_embedding_t>& target_embeddings,
    FM_TYPE curr_nodes[ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM], 
    hls::stream<mp_out_t> messages[NODE_PARALLEL]
) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<int, 20> degrees("degrees");
    hls::stream<node_t, 20> nonzero_degree_nodes("nonzero_degree_nodes");
    hls::stream<mp_out_t, (20 * EMB_DIM)> gather_messages("gather_messages");

    // Read the degrees of nodes in the degrees_table for MP[peID]
    read_degrees(peID, degrees, target_embeddings, nonzero_degree_nodes, numNodes);

    // Gather Node messages
    gather(peID, layerNum, nonzero_degree_nodes, curr_nodes, gather_messages);

    // Expand the messages to all nodes
    expand(gather_messages, degrees, messages, numNodes);
}

// Static Function to read the degrees of nodes in the degrees_table for MP[peID]
static void read_degrees(
    int peID,
    hls::stream<int>& degrees,
    hls::stream<node_embedding_t>& target_embeddings,
    hls::stream<node_t>& nonzero_degree_nodes,
    int num_of_nodes
) {
#pragma HLS INLINE off

    for (int i = 0; i < num_of_nodes; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
        // Get the degree per node
        int degree = degree_tables[peID][i];

        // Write the degree to the degrees stream
        degrees << degree;

        // Read the target embedding for the node
        node_embedding_t embedding;
        target_embeddings >> embedding;

        // Write the node to the nonzero_degree_nodes stream if the degree is nonzero
        if (degree != 0)
        {
            node_t node;
            node.nodeID = i;
            node.degree = degree;
            node.embedding = embedding;
            nonzero_degree_nodes << node;
        }
    }
}


// Static Function to gather Node messages
static void gather(
    int peID,
    int layerNum,
    hls::stream<node_t>& nonzero_degree_nodes,
    FM_TYPE nodeEmb[ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    hls::stream<mp_out_t>& messages
) {
#pragma HLS INLINE off

    // Initialize with the first node this MP is responsible for
    int v = 0;
    int edgeStart = 0;
    int edgeEnd = 0;
    node_embedding_t target_node_embedding;

    // Local storage for message buffer
    mp_out_t message_buffer;

    // Iterate over all edges this MP is responsible for
    for (int edge = 0; edge < numEdgesPerMP[peID]; edge++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_EDGE max=ANALYSIS_MAX_EDGE avg=ANALYSIS_AVG_EDGE

        // Get the source node which is already in the local index space
        int u = neighbor_table[peID][edge];

        node_embedding_t source_node_embedding;

        for (int i = 0; i < EMB_DIM; i++)
        {
#pragma HLS UNROLL
            source_node_embedding[i] = nodeEmb[u][i];
        }

        // Move to the next node if all edges for the current node have been processed
        if (edge >= edgeEnd)
        {
            node_t node;
            nonzero_degree_nodes >> node;
            v = node.nodeID;
            target_node_embedding = node.embedding;
            edgeStart = edge;
            edgeEnd = edgeStart + node.degree;

            // Initialize the message buffer with minimum value
            message_buffer = ap_fixed_min<FM_TYPE>();
        }
        
        // Temporary storage for concantenated features and their difference
        // FM_TYPE concat_features[2 * EMB_DIM];
        hls::vector<FM_TYPE, 2 * EMB_DIM> concat_features;
// #pragma HLS ARRAY_PARTITION variable=concat_features complete dim=1

        // Concatenate the features and compute the difference
        for (int j = 0; j < EMB_DIM; j++)
        {
#pragma HLS UNROLL
            concat_features[j] = target_node_embedding[j];
            concat_features[j + EMB_DIM] = source_node_embedding[j] - target_node_embedding[j];
        }

        // Transform the concatenated features
        // FM_TYPE transformed_features[EMB_DIM];
        hls::vector<FM_TYPE, EMB_DIM> transformed_features;
// #pragma HLS ARRAY_PARTITION variable=transformed_features complete dim=1
        // matmul_and_add_bias
        // (
        //     transformed_features, 
        //     concat_features, 
        //     graphmet_edge_conv_linear_weight[layerNum],
        //     graphmet_edge_conv_linear_bias[layerNum]
        // );

        for (int i = 0; i < EMB_DIM; ++i)
        {
#pragma HLS PIPELINE
            FM_TYPE acc = 0;
            for (int j = 0; j < EMB_DIM * 2; ++j)
            {
#pragma HLS UNROLL
                acc += concat_features[j] * edge_conv_linear_weight[peID][layerNum][i][j];
            }
            transformed_features[i] = acc;
        }

        // Add the bias
        transformed_features += edge_conv_linear_bias[peID][layerNum];

        // Perform max pooling aggregation
        for (int j = 0; j < EMB_DIM; ++j)
        {
#pragma HLS UNROLL
            // FM_TYPE current = message_buffer[j];
            // replace the message with the maximum value
            message_buffer[j] = (message_buffer[j] > transformed_features[j]) ? message_buffer[j] : transformed_features[j];
        }

        // Write the aggregated message to the messages stream when all edges for the current node have been processed
        if (edge == edgeEnd - 1)
        {
            messages << message_buffer;
        }
    }
}

// Static Function to expand the messages to all nodes
static void expand(
    hls::stream<mp_out_t>& messages_per_nz_deg_node,
    hls::stream<int>& degrees,
    hls::stream<mp_out_t> messages_per_node[NODE_PARALLEL],
    int num_of_nodes
) {
#pragma HLS INLINE off

    // Iterate over the nodes
    for (int i = 0; i < num_of_nodes; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE

        int degree;

        // Get the degree per node from the degrees stream
        degrees >> degree;

        // Initialize the message buffer with zero
        mp_out_t message_buffer = ap_fixed_min<FM_TYPE>();

        // If the degree is nonzero, read the message from the messages_per_nz_deg_node stream
        if (degree != 0)
        {
            messages_per_nz_deg_node >> message_buffer;
        }

        // Write the message to the appropriate message stream
        messages_per_node[i % NODE_PARALLEL] << message_buffer;
    }
}