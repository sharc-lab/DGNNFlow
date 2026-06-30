#ifndef LOAD_INPUTS_HPP
#define LOAD_INPUTS_HPP

#include "dcl.hpp"

/**
 *
 * Function Declaration for loading weights into the Kernel
 * @param   fixed point weight arrays
 * @return  None
 */
void load_weights(
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


/**
 *
 * Function Declaration for acquiring graph connectivity information
 * @param   edge_list_in    // List of edges for graph connectivity
 * @param   num_of_nodes    // Number of nodes in the graph
 * @param   num_of_edges    // Number of edges in the graph
 * @return  None
 */
void load_graph(
    edge_t* edge_list_in,
    int num_of_nodes,
    int num_of_edges
);

/**
 *
 * Function Declaration for loading node embeddings for EdgeConv Layer
 * @param   node_feature    // Vector Table of Node Features
 * @param   num_of_nodes    // Number of nodes in the graph
 * @return  None
 */
void load_input_node_embeddings(
    node_feature_t* node_feature,
    int num_of_nodes
);

#endif // LOAD_INPUTS_HPP
