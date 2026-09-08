#include "edge_conv_kernel.hpp"
#include "load_inputs.hpp"
#include "edge_conv.hpp"
#include "operators.hpp"
#include "utils.hpp"
#include "finalize.hpp"

/**
 * Implementation of the EdgeConv Kernel top-level function in HLS
 *
 * @param   num_of_nodes                Number of nodes
 * @param   num_of_edges                Number of edges
 * @param   reload_weights              Reload weights flag (Set to 1 to reload weights from memory to FPGA BRAM and done only once at the beginning)
 * @param   out                         Outputs
 * @param   node_feature_in             Vector Table of Node Features (Array of input features for each node in the graph)
 * @param   edge_list_in                List of Edges (Each edge is struct with source node u and destination node v)
 * @param   **all network weights       All neural network weights
 * @return  None
*/
extern "C" {
    void edge_conv_compute_kernel(
        int num_of_nodes,
        int num_of_edges,
        int reload_weights,
        FM_TYPE out[MAX_NODE],
        node_feature_t* node_feature_in,
        edge_t* edge_list_in,
        WT_TYPE graphmet_embed_charge_weight_in[3][EMB_DIM/4],
        WT_TYPE graphmet_embed_pdgid_weight_in[7][EMB_DIM/4],
        WT_TYPE graphmet_embed_continuous_0_weight_in[EMB_DIM/2][NUM_CONT_FEAT],
        WT_TYPE graphmet_embed_continuous_0_bias_in[EMB_DIM/2],
        WT_TYPE graphmet_embed_categorical_0_weight_in[EMB_DIM/2][EMB_DIM/2],
        WT_TYPE graphmet_embed_categorical_0_bias_in[EMB_DIM/2],
        WT_TYPE graphmet_encode_all_weight_in[EMB_DIM][EMB_DIM],
        WT_TYPE graphmet_encode_all_bias_in[EMB_DIM],
        WT_TYPE graphmet_bn_all_weight_in[EMB_DIM],
        WT_TYPE graphmet_bn_all_bias_in[EMB_DIM],
        WT_TYPE graphmet_bn_all_running_mean_in[EMB_DIM],
        WT_TYPE graphmet_bn_all_running_var_in[EMB_DIM],
        WT_TYPE graphmet_edge_conv_linear_weight_in[NUM_LAYERS][EMB_DIM][2*EMB_DIM],
        WT_TYPE graphmet_edge_conv_linear_bias_in[NUM_LAYERS][EMB_DIM],
        WT_TYPE graphmet_edge_conv_batchnorm_weight_in[NUM_LAYERS][EMB_DIM],
        WT_TYPE graphmet_edge_conv_batchnorm_bias_in[NUM_LAYERS][EMB_DIM],
        WT_TYPE graphmet_edge_conv_batchnorm_mean_in[NUM_LAYERS][EMB_DIM],
        WT_TYPE graphmet_edge_conv_batchnorm_variance_in[NUM_LAYERS][EMB_DIM],
        WT_TYPE graphmet_output_0_weight_in[EMB_DIM/2][EMB_DIM],
        WT_TYPE graphmet_output_0_bias_in[EMB_DIM/2],
        WT_TYPE graphmet_output_2_weight_in[1][EMB_DIM/2],
        WT_TYPE graphmet_output_2_bias_in[1]
    ) {

        // DEBUG: Print enetering the kernel
        // hls::print(%s, "[DEBUG] Inside the EdgeConv Kernel\n");
        // for (int i = 0; i < 10000; i++) printf("[DEBUG] Inside the EdgeConv Kernel\n");
        // printf("[DEBUG] Inside the EdgeConv Kernel\n");

#pragma HLS INTERFACE s_axilite port=return
// #pragma HLS INTERFACE m_axi depth=(1) port=num_of_nodes offset=slave bundle=mem
// #pragma HLS INTERFACE m_axi depth=(1) port=num_of_edges offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=out offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(500) port=node_feature_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(500) port=edge_list_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_embed_charge_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_embed_pdgid_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_embed_continuous_0_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_embed_continuous_0_bias_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_embed_categorical_0_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_embed_categorical_0_bias_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_encode_all_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_encode_all_bias_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_bn_all_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_bn_all_bias_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_bn_all_running_mean_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_bn_all_running_var_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_edge_conv_linear_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_edge_conv_linear_bias_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_edge_conv_batchnorm_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_edge_conv_batchnorm_bias_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_edge_conv_batchnorm_mean_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_edge_conv_batchnorm_variance_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_output_0_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_output_0_bias_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_output_2_weight_in offset=slave bundle=mem
#pragma HLS INTERFACE m_axi depth=(1) port=graphmet_output_2_bias_in offset=slave bundle=mem

#pragma HLS ARRAY_PARTITION variable=NodeEmbeddings cyclic factor=EDGE_PARALLEL dim=1


        // Load the weights
        if (reload_weights == 1)
        {
            load_weights(
                graphmet_embed_charge_weight_in,
                graphmet_embed_pdgid_weight_in,
                graphmet_embed_continuous_0_weight_in,
                graphmet_embed_continuous_0_bias_in,
                graphmet_embed_categorical_0_weight_in,
                graphmet_embed_categorical_0_bias_in,
                graphmet_encode_all_weight_in,
                graphmet_encode_all_bias_in,
                graphmet_bn_all_weight_in,
                graphmet_bn_all_bias_in,
                graphmet_bn_all_running_mean_in,
                graphmet_bn_all_running_var_in,
                graphmet_edge_conv_linear_weight_in,
                graphmet_edge_conv_linear_bias_in,
                graphmet_edge_conv_batchnorm_weight_in,
                graphmet_edge_conv_batchnorm_bias_in,
                graphmet_edge_conv_batchnorm_mean_in,
                graphmet_edge_conv_batchnorm_variance_in,
                graphmet_output_0_weight_in,
                graphmet_output_0_bias_in,
                graphmet_output_2_weight_in,
                graphmet_output_2_bias_in
            );
        }
        
        // Load graph edges
        load_graph(edge_list_in, num_of_nodes, num_of_edges);

        // Load node embeddings
        load_input_node_embeddings(node_feature_in, num_of_nodes);

        // Run EdgeConv layers
        for (int i = 0; i < NUM_LAYERS; i++)
        {
            if (i % 2 == 0)
            {
                // duplicate the ping buffer embeddings
                for (int node = 0; node < num_of_nodes; node++)
                {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
                    for ( int emb = 0; emb < EMB_DIM; emb++ )
                    {
#pragma HLS UNROLL
                        duplicateNodeEmbeddings[node % EDGE_PARALLEL][node / EDGE_PARALLEL][emb] = currentNodeEmbeddings[node % EDGE_PARALLEL][node / EDGE_PARALLEL][emb];
                        NodeEmbeddings[node][emb] = currentNodeEmbeddings[node % EDGE_PARALLEL][node / EDGE_PARALLEL][emb];

                    }
                }

                edge_convolution_layer(
                    i,
                    num_of_nodes,
                    num_of_edges,
                    currentNodeEmbeddings,
                    nextNodeEmbeddings,
                    duplicateNodeEmbeddings
                );
            }
            else
            {

                // duplicate the ping buffer embeddings
                for (int node = 0; node < num_of_nodes; node++)
                {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
                    for ( int emb = 0; emb < EMB_DIM; emb++ )
                    {
#pragma HLS UNROLL
                        duplicateNodeEmbeddings[node % EDGE_PARALLEL][node / EDGE_PARALLEL][emb] = nextNodeEmbeddings[node % EDGE_PARALLEL][node / EDGE_PARALLEL][emb];
                        NodeEmbeddings[node][emb] = nextNodeEmbeddings[node % EDGE_PARALLEL][node / EDGE_PARALLEL][emb];
                    }
                }

                edge_convolution_layer(
                    i,
                    num_of_nodes,
                    num_of_edges,
                    nextNodeEmbeddings,
                    currentNodeEmbeddings,
                    duplicateNodeEmbeddings
                );
            }
        }

// Partition Output Layer weights and biases
#pragma HLS ARRAY_PARTITION variable=graphmet_output_0_weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=graphmet_output_0_bias complete dim=1
#pragma HLS ARRAY_PARTITION variable=graphmet_output_2_weight complete dim=2
#pragma HLS ARRAY_PARTITION variable=graphmet_output_2_bias complete dim=1

        // Run the Final Layer
        finalize(
            num_of_nodes, 
            ( NUM_LAYERS % 2 == 0 ) ? currentNodeEmbeddings : nextNodeEmbeddings,
            graphmet_output_0_weight, 
            graphmet_output_0_bias, 
            graphmet_output_2_weight, 
            graphmet_output_2_bias, 
            out
        );
    }
}