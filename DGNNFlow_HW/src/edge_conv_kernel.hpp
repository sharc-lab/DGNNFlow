#ifndef EDGE_CONV_KERNEL_HPP
#define EDGE_CONV_KERNEL_HPP

#include "dcl.hpp"

/**
 * Declaration of the EdgeConv Kernel top-level interface
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
    );
}

#endif // EDGE_CONV_KERNEL_HPP
