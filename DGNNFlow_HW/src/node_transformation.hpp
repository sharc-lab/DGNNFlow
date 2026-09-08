#ifndef NODE_TRANSFORMATION_HPP
#define NODE_TRANSFORMATION_HPP

#include "dcl.hpp"
#include "hls_stream.h"


static void accumulate(
    int layerNum,
    int numNodes,
    hls::stream<ne_in_t> messages[NODE_PARALLEL],
    hls::stream<ne_in_t> accumulated_messages[NODE_PARALLEL]
);

static void output(
    int layerNum,
    int numNodes,
    hls::stream<ne_in_t> accumulated_messages[NODE_PARALLEL],
    FM_TYPE curr_embeddings[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE nextNodeEmb[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM]
);

void node_transformation(
    int numNodes,
    int layerNum,
    hls::stream<ne_in_t> messages[NODE_PARALLEL],
    FM_TYPE curr_embeddings[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE next_node_embeds[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM]
);

#endif // NODE_TRANSFORMATION_HPP