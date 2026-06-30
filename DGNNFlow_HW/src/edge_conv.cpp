#include "utils.hpp"
#include "operators.hpp"
#include "edge_conv.hpp"
#include "mp_to_ne_adapter.hpp"
#include "message_passing.hpp"
#include "node_transformation.hpp"

void multicast_target_embeddings(
    int numNodes,
    // hls::stream<node_embedding_t>& nt_multicast_embeddings,
    hls::stream<node_embedding_t> mp_multicast_embeddings[EDGE_PARALLEL]
) {
#pragma HLS INLINE off

    // Broadtcast the target embeddings to all MPs
    for (int i = 0; i < numNodes; i++)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
        node_embedding_t target_embedding = NodeEmbeddings[i];

        // Send the target embedding to all MPs
        for (int pe = 0; pe < EDGE_PARALLEL; pe++)
        {
#pragma HLS UNROLL
            mp_multicast_embeddings[pe] << target_embedding;
        }

        // Multicast the target embedding to the NT
        // nt_multicast_embeddings << target_embedding;
    }
}

void edge_convolution_layer(
    int layerNum,
    int num_nodes,
    int num_edges,
    FM_TYPE node_embeddings[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE next_node_embeddings[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE duplicate_node_embeddings[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM]
) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW
#pragma HLS allocation function instances=edge_convolution_layer limit=1

// // Partition the Edge Convolution Linear weights and biases (Message Passing Unit)
// #pragma HLS ARRAY_PARTITION variable=graphmet_edge_conv_linear_weight complete dim=1
// #pragma HLS ARRAY_PARTITION variable=graphmet_edge_conv_linear_weight cyclic factor=8 dim=3
// #pragma HLS ARRAY_PARTITION variable=graphmet_edge_conv_linear_bias complete dim=1
// #pragma HLS ARRAY_PARTITION variable=graphmet_edge_conv_linear_bias cyclic factor=8 dim=2

// Partition the Linear Layer Paramaters
#pragma HLS ARRAY_PARTITION variable=edge_conv_linear_weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=edge_conv_linear_bias complete dim=1

// Partition the Edge Convolution BatchNorm weights, biases, mean and variance (Node Transformation Unit)
#pragma HLS ARRAY_PARTITION variable=graphmet_edge_conv_batchnorm_weight complete dim=1
#pragma HLS ARRAY_PARTITION variable=graphmet_edge_conv_batchnorm_weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=graphmet_edge_conv_batchnorm_bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=graphmet_edge_conv_batchnorm_bias complete dim=2
#pragma HLS ARRAY_PARTITION variable=graphmet_edge_conv_batchnorm_mean complete dim=1
#pragma HLS ARRAY_PARTITION variable=graphmet_edge_conv_batchnorm_mean complete dim=2
#pragma HLS ARRAY_PARTITION variable=graphmet_edge_conv_batchnorm_variance complete dim=1
#pragma HLS ARRAY_PARTITION variable=graphmet_edge_conv_batchnorm_variance complete dim=2


    hls::stream<mp_out_t> mp_out[EDGE_PARALLEL][NODE_PARALLEL];
    hls::stream<ne_in_t> ne_in[NODE_PARALLEL];
    hls::stream<node_embedding_t, 20> mp_multicast_target_embeddings[EDGE_PARALLEL];
    // hls::stream<node_embedding_t, 20> nt_multicast_target_embeddings;

#pragma HLS STREAM variable=mp_out depth=(20 * EMB_DIM)
#pragma HLS STREAM variable=ne_in depth=(4 * EMB_DIM)

    // Multicast the target embeddings to all MPs and NTs
    // multicast_target_embeddings(num_nodes, nt_multicast_target_embeddings, mp_multicast_target_embeddings);
    multicast_target_embeddings(num_nodes, mp_multicast_target_embeddings);

    // Instantiate the Message Passing Units
    for (int pe = 0; pe < EDGE_PARALLEL; pe++)
    {
#pragma HLS UNROLL
        message_passing(pe, num_nodes, layerNum, mp_multicast_target_embeddings[pe], node_embeddings[pe], mp_out[pe]);
    }

    // Adapter for the Message Passing to Node Embedding Unit
    for ( int nd_offset = 0; nd_offset < NODE_PARALLEL; nd_offset++ )  
    {
#pragma HLS UNROLL
        mp_to_ne_adapter(nd_offset, num_nodes, mp_out, ne_in[nd_offset]);
    }

    // Node Transformation Unit reading from the messages stream & duplicate node embeddings
    // and writing to the next node embeddings
    // node_transformation(num_nodes, layerNum, ne_in, nt_multicast_target_embeddings, next_node_embeddings);
    node_transformation(num_nodes, layerNum, ne_in, duplicate_node_embeddings, next_node_embeddings);
}