#ifndef HOST_LOAD_HPP
#define HOST_LOAD_HPP

#include "host.hpp"

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
