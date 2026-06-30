#include "hls_math.h"
#include "utils.hpp"
#include "node_transformation.hpp"
// #include <fstream>

void node_transformation(
    int numNodes,
    int layerNum,
    hls::stream<ne_in_t> messages[NODE_PARALLEL],
    FM_TYPE curr_embeddings[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE next_node_embeds[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM]
) {
#pragma HLS INLINE off
#pragma HLS DATAFLOW

    hls::stream<ne_in_t, 20> accumulated_messages[NODE_PARALLEL];

    accumulate(layerNum, numNodes, messages, accumulated_messages);
    output(layerNum, numNodes, accumulated_messages, curr_embeddings, next_node_embeds);
}

static void accumulate(
    int layerNum,
    int numNodes,
    hls::stream<ne_in_t> messages[NODE_PARALLEL],
    hls::stream<ne_in_t> accumulated_messages[NODE_PARALLEL]
) {
#pragma HLS INLINE off

    int num_iter = ceildiv(numNodes, NODE_PARALLEL);
    for ( int i = 0, v_base = 0; i < num_iter; i++, v_base += NODE_PARALLEL )
    {
#pragma HLS LOOP_TRIPCOUNT min=(ceildiv(ANALYSIS_MIN_NODE, NODE_PARALLEL)) max=(ceildiv(ANALYSIS_MAX_NODE, NODE_PARALLEL)) avg=(ceildiv(ANALYSIS_AVG_NODE, NODE_PARALLEL))

        for ( int offset = 0; offset < NODE_PARALLEL; offset++ )
        {
#pragma HLS UNROLL
            int v = v_base + offset;

            // // print the messages for v
            // std::ofstream messages_file;
            // messages_file.open("messages_dut.txt", std::ios::app);
            // messages_file << "Node[" << v << "]: " << std::endl;

            if ( v < numNodes )
            {
                ne_in_t message_buffer;
                ne_in_t acc_message;
                messages[offset] >> message_buffer;

                // // print the messages for v into a file
                // for (int emb = 0; emb < EMB_DIM; emb++)
                // {
                //     messages_file << message_buffer[emb] << std::endl;
                // }
                // messages_file << std::endl;
                // messages_file.close();

                // Process the message with batch normalization
                for (int emb = 0; emb < EMB_DIM; emb++)
                {
#pragma HLS UNROLL
                    FM_TYPE normalized = (message_buffer[emb] - graphmet_edge_conv_batchnorm_mean[layerNum][emb]) / hls::sqrt(graphmet_edge_conv_batchnorm_variance[layerNum][emb]+ EPSILON);
        
                    acc_message[emb] = (normalized * graphmet_edge_conv_batchnorm_weight[layerNum][emb]) + graphmet_edge_conv_batchnorm_bias[layerNum][emb];
                }

                accumulated_messages[offset] << acc_message;
            }
        }
    }
}

static void output(
    int layerNum,
    int numNodes,
    hls::stream<ne_in_t> accumulated_messages[NODE_PARALLEL],
    FM_TYPE curr_embeddings[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE nextNodeEmb[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM]
) {
#pragma HLS INLINE off

    int num_iter = ceildiv(numNodes, NODE_PARALLEL);
    for ( int i = 0, v_base = 0; i < num_iter; i++, v_base += NODE_PARALLEL )
    {
    #pragma HLS LOOP_TRIPCOUNT min=(ceildiv(ANALYSIS_MIN_NODE, NODE_PARALLEL)) max=(ceildiv(ANALYSIS_MAX_NODE, NODE_PARALLEL)) avg=(ceildiv(ANALYSIS_AVG_NODE, NODE_PARALLEL))

        for ( int offset = 0; offset < NODE_PARALLEL; offset++ )
        {
#pragma HLS UNROLL
            int v = v_base + offset;

            if ( v < numNodes )
            {
                // Read the accumulated message from the stream for node v
                ne_in_t acc_message;
                accumulated_messages[offset] >> acc_message;

                // // Get the node embeddings for node v from the stream
                // node_embedding_t node_embedding;
                // nt_multicast_embeddings >> node_embedding;

                for (int emb = 0; emb < EMB_DIM; emb++)
                {
#pragma HLS UNROLL
                    // nextNodeEmb[v % EDGE_PARALLEL][v / EDGE_PARALLEL][emb] = node_embedding[emb] + acc_message[emb];
                    nextNodeEmb[v % EDGE_PARALLEL][v / EDGE_PARALLEL][emb] = curr_embeddings[v % EDGE_PARALLEL][v / EDGE_PARALLEL][emb] + acc_message[emb];
                }
            }
        }
    }
}