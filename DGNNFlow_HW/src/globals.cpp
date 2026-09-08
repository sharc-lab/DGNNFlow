#include "dcl.hpp"
#include "utils.hpp"

/**
 *
 * Declaration of Weight/Parameter Vectors for L1DeepMetv2
 */
int pdgs[NUM_PDGS] = {1, 2, 11, 13, 22, 130, 211};
WT_TYPE EPSILON = 0.00001;
WT_TYPE norm[NUM_CONT_FEAT] = {1.0 / 128, 1.0 / 128, 1.0 / 128, 1.0, 1.0, 1.0};

// Weights for loading input node embeddings
WT_TYPE graphmet_embed_charge_weight[3][EMB_DIM/4];
WT_TYPE graphmet_embed_pdgid_weight[7][EMB_DIM/4];
WT_TYPE graphmet_embed_continuous_0_weight[EMB_DIM/2][NUM_CONT_FEAT];
WT_TYPE graphmet_embed_continuous_0_bias[EMB_DIM/2];
WT_TYPE graphmet_embed_categorical_0_weight[EMB_DIM/2][EMB_DIM/2];
WT_TYPE graphmet_embed_categorical_0_bias[EMB_DIM/2];
WT_TYPE graphmet_encode_all_weight[EMB_DIM][EMB_DIM];
WT_TYPE graphmet_encode_all_bias[EMB_DIM];
WT_TYPE graphmet_bn_all_weight[EMB_DIM];
WT_TYPE graphmet_bn_all_bias[EMB_DIM];
WT_TYPE graphmet_bn_all_running_mean[EMB_DIM];
WT_TYPE graphmet_bn_all_running_var[EMB_DIM];

// Weights for EdgeConv Linear Layers
// WT_TYPE graphmet_edge_conv_linear_weight[NUM_LAYERS][EMB_DIM][2*EMB_DIM];
// WT_TYPE graphmet_edge_conv_linear_bias[NUM_LAYERS][EMB_DIM];

// Linear Layer Parameters for MP Units
hls::vector<WT_TYPE, 2*EMB_DIM> edge_conv_linear_weight[EDGE_PARALLEL][NUM_LAYERS][EMB_DIM];
hls::vector<WT_TYPE, EMB_DIM> edge_conv_linear_bias[EDGE_PARALLEL][NUM_LAYERS];

// Weights for EdgeConv BatchNorm1D Layers
WT_TYPE graphmet_edge_conv_batchnorm_weight[NUM_LAYERS][EMB_DIM];
WT_TYPE graphmet_edge_conv_batchnorm_bias[NUM_LAYERS][EMB_DIM];
WT_TYPE graphmet_edge_conv_batchnorm_mean[NUM_LAYERS][EMB_DIM];
WT_TYPE graphmet_edge_conv_batchnorm_variance[NUM_LAYERS][EMB_DIM];

// Weights for final forward layer
WT_TYPE graphmet_output_0_weight[EMB_DIM/2][EMB_DIM];
WT_TYPE graphmet_output_0_bias[EMB_DIM/2];
WT_TYPE graphmet_output_2_weight[1][EMB_DIM/2];
WT_TYPE graphmet_output_2_bias[1];

// Intermediate BRAMs for EdgeConv
FM_TYPE currentNodeEmbeddings[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];
FM_TYPE nextNodeEmbeddings[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];
FM_TYPE duplicateNodeEmbeddings[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];

// Intermediate BRAMs for Message Passing
node_embedding_t NodeEmbeddings[MAX_NODE];

int degree_table[MAX_NODE];
int degree_tables[EDGE_PARALLEL][MAX_NODE];
int neighbor_table[EDGE_PARALLEL][MAX_EDGE];
int numEdgesPerMP[EDGE_PARALLEL];

