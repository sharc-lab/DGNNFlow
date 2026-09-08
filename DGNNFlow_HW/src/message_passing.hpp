#ifndef MESSAGE_PASSING_HPP
#define MESSAGE_PASSING_HPP

#include "dcl.hpp"
#include "hls_stream.h"

// Message Passing Unit
// @param[in]  numNodes                 Number of nodes
// @param[in]  layerNum                 Layer Number
// @param[in]  currentNodeEmbeddings    Embeddings of the current node
// @param[out] messages                 Messages to be sent to the neighbors
// @return     void
void message_passing(
    int peID,
    int numNodes,
    int layerNum,
    hls::stream<node_embedding_t>& target_embeddings,
    FM_TYPE curr_nodes[ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM], 
    hls::stream<mp_out_t> messages[NODE_PARALLEL]
);

// Static Function to read the degrees of nodes in the degrees_table for MP[peID]
static void read_degrees(
    int peID,
    hls::stream<int>& degrees,
    hls::stream<node_embedding_t>& target_embeddings,
    hls::stream<node_t>& nonzero_degree_nodes,
    int num_of_nodes
);

// Static Function to gather Node messages
static void gather(
    int peID,
    int layerNum,
    hls::stream<node_t>& nonzero_degree_nodes,
    FM_TYPE nodeEmb[ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    hls::stream<mp_out_t>& messages
);

static void expand(
    hls::stream<mp_out_t>& messages_per_nz_deg_node,
    hls::stream<int>& degrees,
    hls::stream<mp_out_t> messages_per_node[NODE_PARALLEL],
    int num_of_nodes
);


#endif // MESSAGE_PASSING_HPP