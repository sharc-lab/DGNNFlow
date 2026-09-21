#ifndef DCL_HPP
#define DCL_HPP

#include <ap_fixed.h>
#include <hls_vector.h>

#include "utils.hpp"

// EdgeConv Model Parameters
constexpr int MAX_EDGE = 3000;
constexpr int MAX_NODE = 400;
constexpr int ND_FEATURE = 8;
constexpr int NUM_CONT_FEAT = 6;
constexpr int NUM_CAT_FEAT = 2;
constexpr int EMB_DIM = 32;
constexpr int NUM_LAYERS = 2;
constexpr int NUM_PDGS = 7;
constexpr int NUM_OUTPUTS = 1;
constexpr int NUM_TASK = 1;

// #region Hardware Parameters
constexpr int GATHER_PARALLEL = 8; // how many dimensions of EMB_DIM should a message passing PE process each cycle?
constexpr int APPLY_PARALLEL = 1; // how many dimensions of EMB_DIM should the node embedding PE process each cycle?
constexpr int NODE_PARALLEL = 2; // how many nodes should the node embedding PE process simultaneously?
constexpr int EDGE_PARALLEL = 4; // how many message passing PEs are there?

// Graph Statistics
constexpr int ANALYSIS_MAX_NODE = 151;
constexpr int ANALYSIS_MIN_NODE = 6;
constexpr int ANALYSIS_AVG_NODE = 48;
constexpr int ANALYSIS_MAX_EDGE = 2500;
constexpr int ANALYSIS_MIN_EDGE = 2;
constexpr int ANALYSIS_AVG_EDGE = 200;

// 16-bit fixed point numbers for Floating point numbers
typedef ap_fixed<16, 6> FM_TYPE;

// 16-bit fixed point numbers for Weights
typedef ap_fixed<16, 6> WT_TYPE;

// Node Feature Vector
typedef hls::vector<FM_TYPE, ND_FEATURE> node_feature_t;

// Node Embedding Vector
typedef hls::vector<FM_TYPE, EMB_DIM> node_embedding_t;

// Buffers to store streamed outputs of MP & NT units
typedef hls::vector<FM_TYPE, EMB_DIM> mp_out_t;
typedef hls::vector<FM_TYPE, EMB_DIM> ne_in_t;

// Struct Representation for an Edge
typedef struct
{
   int u;      // Source Node
   int v;      // Target Node
} edge_t;

// Struct Representation for a Node
typedef struct
{
   int nodeID;
   int degree;
   node_embedding_t embedding; // Embedding of the node
} node_t;

/**
 * Declaration of Weight/Parameter Vectors for L1DeepMetv2
 */
extern int pdgs[NUM_PDGS];
extern WT_TYPE deltaR;
extern WT_TYPE EPSILON;
extern WT_TYPE norm[NUM_CONT_FEAT];

// Weights for loading input node embeddings
extern WT_TYPE graphmet_embed_charge_weight[3][EMB_DIM/4];
extern WT_TYPE graphmet_embed_pdgid_weight[7][EMB_DIM/4];
extern WT_TYPE graphmet_embed_continuous_0_weight[EMB_DIM/2][NUM_CONT_FEAT];
extern WT_TYPE graphmet_embed_continuous_0_bias[EMB_DIM/2];
extern WT_TYPE graphmet_embed_categorical_0_weight[EMB_DIM/2][EMB_DIM/2];
extern WT_TYPE graphmet_embed_categorical_0_bias[EMB_DIM/2];
extern WT_TYPE graphmet_encode_all_weight[EMB_DIM][EMB_DIM];
extern WT_TYPE graphmet_encode_all_bias[EMB_DIM];
extern WT_TYPE graphmet_bn_all_weight[EMB_DIM];
extern WT_TYPE graphmet_bn_all_bias[EMB_DIM];
extern WT_TYPE graphmet_bn_all_running_mean[EMB_DIM];
extern WT_TYPE graphmet_bn_all_running_var[EMB_DIM];

// Weights for EdgeConv Linear Layers
// extern WT_TYPE graphmet_edge_conv_linear_weight[NUM_LAYERS][EMB_DIM][2*EMB_DIM];
// extern WT_TYPE graphmet_edge_conv_linear_bias[NUM_LAYERS][EMB_DIM];

// Linear Layer Parameters
extern hls::vector<WT_TYPE, 2*EMB_DIM> edge_conv_linear_weight[EDGE_PARALLEL][NUM_LAYERS][EMB_DIM];
extern hls::vector<WT_TYPE, EMB_DIM> edge_conv_linear_bias[EDGE_PARALLEL][NUM_LAYERS];

// Weights for EdgeConv BatchNorm1D Layers
extern WT_TYPE graphmet_edge_conv_batchnorm_weight[NUM_LAYERS][EMB_DIM];
extern WT_TYPE graphmet_edge_conv_batchnorm_bias[NUM_LAYERS][EMB_DIM];
extern WT_TYPE graphmet_edge_conv_batchnorm_mean[NUM_LAYERS][EMB_DIM];
extern WT_TYPE graphmet_edge_conv_batchnorm_variance[NUM_LAYERS][EMB_DIM];

// Weights for final forward layer
extern WT_TYPE graphmet_output_0_weight[EMB_DIM/2][EMB_DIM];
extern WT_TYPE graphmet_output_0_bias[EMB_DIM/2];
extern WT_TYPE graphmet_output_2_weight[1][EMB_DIM/2];
extern WT_TYPE graphmet_output_2_bias[1];

// Intermediate BRAMs for EdgeConv
extern FM_TYPE currentNodeEmbeddings[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];
extern FM_TYPE nextNodeEmbeddings[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];
extern FM_TYPE duplicateNodeEmbeddings[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM];

// Intermediate BRAMs for Message Passing
extern node_embedding_t NodeEmbeddings[MAX_NODE];

// Intermediate BRAMs for Edge Information
// extern hls::vector<int, MAX_NODE> degree_table;
extern int degree_table[MAX_NODE];
extern int degree_tables[EDGE_PARALLEL][MAX_NODE];
extern int neighbor_table[EDGE_PARALLEL][MAX_EDGE];
extern int numEdgesPerMP[EDGE_PARALLEL];

#endif
