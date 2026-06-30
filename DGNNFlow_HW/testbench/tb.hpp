#ifndef TB_HPP
#define TB_HPP

#include "../src/dcl.hpp"
#include <vector>
#include <string>

// Weights for loading input node embeddings
extern WT_TYPE graphmet_embed_charge_weight_fixed[3][EMB_DIM / 4];
extern WT_TYPE graphmet_embed_pdgid_weight_fixed[7][EMB_DIM / 4];
extern WT_TYPE graphmet_embed_continuous_0_weight_fixed[EMB_DIM / 2][NUM_CONT_FEAT];
extern WT_TYPE graphmet_embed_continuous_0_bias_fixed[EMB_DIM / 2];
extern WT_TYPE graphmet_embed_categorical_0_weight_fixed[EMB_DIM / 2][EMB_DIM / 2];
extern WT_TYPE graphmet_embed_categorical_0_bias_fixed[EMB_DIM / 2];
extern WT_TYPE graphmet_encode_all_weight_fixed[EMB_DIM][EMB_DIM];
extern WT_TYPE graphmet_encode_all_bias_fixed[EMB_DIM];
extern WT_TYPE graphmet_bn_all_weight_fixed[EMB_DIM];
extern WT_TYPE graphmet_bn_all_bias_fixed[EMB_DIM];
extern WT_TYPE graphmet_bn_all_running_mean_fixed[EMB_DIM];
extern WT_TYPE graphmet_bn_all_running_var_fixed[EMB_DIM];

// Weights for EdgeConv Linear Layers
extern WT_TYPE graphmet_edge_conv_linear_weight_fixed[NUM_LAYERS][EMB_DIM][2 * EMB_DIM];

// Weights for EdgeConv Bias Layers
extern WT_TYPE graphmet_edge_conv_linear_bias_fixed[NUM_LAYERS][EMB_DIM];

// Weights for EdgeConv BatchNorm1D Layers
extern WT_TYPE graphmet_edge_conv_batchnorm_weight_fixed[NUM_LAYERS][EMB_DIM];
extern WT_TYPE graphmet_edge_conv_batchnorm_bias_fixed[NUM_LAYERS][EMB_DIM];
extern WT_TYPE graphmet_edge_conv_batchnorm_mean_fixed[NUM_LAYERS][EMB_DIM];
extern WT_TYPE graphmet_edge_conv_batchnorm_variance_fixed[NUM_LAYERS][EMB_DIM];

// Weights for final forward layer
extern WT_TYPE graphmet_output_0_weight_fixed[EMB_DIM / 2][EMB_DIM];
extern WT_TYPE graphmet_output_0_bias_fixed[EMB_DIM / 2];
extern WT_TYPE graphmet_output_2_weight_fixed[1][EMB_DIM / 2];
extern WT_TYPE graphmet_output_2_bias_fixed[1];


/**
 *
 * Function Declaration of load_weights()
 *
 * Description: Load weights of the Trained L1DeepMetv2 model from binary files
 *
 * @param   None
 * @return  None
 */
void load_float_weights(std::string weights);

/**
 *
 * Function Declaration of initialize_node_embedding_parameters()
 *
 * Description: Convert float weights to fixed point weights for node embeddings
 *
 * @param   None
 * @return  None
 */
void initialize_node_embedding_parameters();

/**
 *
 * Function Declaration of initialize_edge_conv_parameters()
 *
 * Description: Convert float weights to fixed point weights for EdgeConv Parameters
 *
 * @param   None
 * @return  None
 */
void initialize_edge_conv_parameters();

float euclidean(node_feature_t p1, node_feature_t p2) ;

/**
 *
 * Function Declaration of fetch_one_graph()
 *
 * Description: Get the parameters of a graph for inference
 *
 * @param   node_feature    aligned_vector<node_feature_t>&     // Reference to an aligned vector of arrays of node features
 * @param   edge_list       aligned_vector<edge_t>&             // Reference to an aligned vector
 * @return  None
 */
// Fetch One Graph Function
void fetch_one_graph(
    node_feature_t* node_feature,
    edge_t* edge_list,
    float radius,
    int num_nodes,
    int& num_edges,
    int max_edges
);

// Function to print all weights of type WT_TYPE
void print_weights_debug();


#endif
