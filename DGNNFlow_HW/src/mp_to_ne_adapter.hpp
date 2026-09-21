#ifndef MP_TO_NE_ADAPTER_HPP
#define MP_TO_NE_ADAPTER_HPP

#include "dcl.hpp"
#include "hls_stream.h"

/**
 * Adapter for the Message Passing to Node Embedding Unit
 * @param   nd_offset       // Offset for the current node
 * @param   num_of_nodes    // Number of nodes in the graph
 * @param   mp_out          // Stream of Message Passing Outputs
 * @param   ne_in           // Stream of Node Embedding Inputs
 * @return  None
 */
void mp_to_ne_adapter(
    int nd_offset,
    int num_of_nodes,
    hls::stream<mp_out_t> mp_out[EDGE_PARALLEL][NODE_PARALLEL],
    hls::stream<ne_in_t>& ne_in
);

#endif // MP_TO_NE_ADAPTER_HPP