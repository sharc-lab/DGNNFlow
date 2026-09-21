#ifndef HOST_HPP
#define HOST_HPP

#include <vector>
#include <string>
#include <array>
#include <ap_fixed.h>
#include <hls_vector.h>
#include "xcl2.hpp"


// EdgeConv model parameters
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

// Hardware parallelism parameters
constexpr int GATHER_PARALLEL = 8; // how many dimensions of EMB_DIM should a message passing PE process each cycle?
constexpr int APPLY_PARALLEL = 1; // how many dimensions of EMB_DIM should the node embedding PE process each cycle?
constexpr int NODE_PARALLEL = 2; // how many nodes should the node embedding PE process simultaneously?
constexpr int EDGE_PARALLEL = 4; // how many message passing PEs are there?

// Data types
typedef ap_fixed<16, 6> FM_TYPE;
typedef ap_fixed<16, 6> WT_TYPE;

typedef std::array<FM_TYPE, ND_FEATURE> node_feature_t;
typedef std::array<FM_TYPE, EMB_DIM> node_embedding_t;

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


template <typename T>
using aligned_vector = std::vector<T, aligned_allocator<T>>;

extern aligned_vector<WT_TYPE> graphmet_embed_charge_weight_fixed;
extern aligned_vector<WT_TYPE> graphmet_embed_pdgid_weight_fixed;
extern aligned_vector<WT_TYPE> graphmet_embed_continuous_0_weight_fixed;
extern aligned_vector<WT_TYPE> graphmet_embed_continuous_0_bias_fixed;
extern aligned_vector<WT_TYPE> graphmet_embed_categorical_0_weight_fixed;
extern aligned_vector<WT_TYPE> graphmet_embed_categorical_0_bias_fixed;
extern aligned_vector<WT_TYPE> graphmet_encode_all_weight_fixed;
extern aligned_vector<WT_TYPE> graphmet_encode_all_bias_fixed;
extern aligned_vector<WT_TYPE> graphmet_bn_all_weight_fixed;
extern aligned_vector<WT_TYPE> graphmet_bn_all_bias_fixed;
extern aligned_vector<WT_TYPE> graphmet_bn_all_running_mean_fixed;
extern aligned_vector<WT_TYPE> graphmet_bn_all_running_var_fixed;

// Weights for EdgeConv Linear & Bias Layers
extern aligned_vector<WT_TYPE> graphmet_edge_conv_linear_weight_fixed;
extern aligned_vector<WT_TYPE> graphmet_edge_conv_linear_bias_fixed;

// Weights for EdgeConv BatchNorm1D Layers
extern aligned_vector<WT_TYPE> graphmet_edge_conv_batchnorm_weight_fixed;
extern aligned_vector<WT_TYPE> graphmet_edge_conv_batchnorm_bias_fixed;
extern aligned_vector<WT_TYPE> graphmet_edge_conv_batchnorm_mean_fixed;
extern aligned_vector<WT_TYPE> graphmet_edge_conv_batchnorm_variance_fixed;

// Weights for final forward layer
extern aligned_vector<WT_TYPE> graphmet_output_0_weight_fixed;
extern aligned_vector<WT_TYPE> graphmet_output_0_bias_fixed;
extern aligned_vector<WT_TYPE> graphmet_output_2_weight_fixed;
extern aligned_vector<WT_TYPE> graphmet_output_2_bias_fixed;

// Helper functions
void load_float_weights(std::string weights);
void initialize_node_embedding_parameters();
void initialize_edge_conv_parameters();
float euclidean(node_feature_t p1, node_feature_t p2);
void fetch_one_graph(
    node_feature_t* node_feature,
    edge_t* edge_list,
    float radius,
    int num_nodes,
    int& num_edges,
    int max_edges
);

#endif
