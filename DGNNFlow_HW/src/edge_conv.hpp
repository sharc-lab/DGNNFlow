#ifndef EDGE_CONV_H
#define EDGE_CONV_H

#include "dcl.hpp"
#include "hls_stream.h"

void edge_convolution_layer(
    int layerNum,
    int num_nodes,
    int num_edges,
    FM_TYPE node_embeddings[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE next_node_embeddings[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE duplicate_node_embeddings[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM]
);

void multicast_target_embeddings(
    int numNodes,
    // hls::stream<node_embedding_t>& nt_multicast_embeddings,
    hls::stream<node_embedding_t> mp_multicast_embeddings[EDGE_PARALLEL]
);

#endif
