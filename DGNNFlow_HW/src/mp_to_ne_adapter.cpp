#include "mp_to_ne_adapter.hpp"
#include "utils.hpp"

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
) {
#pragma HLS INLINE off

    int num_iters = ceildiv(num_of_nodes - nd_offset, NODE_PARALLEL);

    mp_out_t noMessages = ap_fixed_min<FM_TYPE>();

    for ( int i = 0; i < num_iters; i++ )
    {
#pragma HLS LOOP_TRIPCOUNT min=ceildiv(ANALYSIS_MIN_NODE, NODE_PARALLEL) max=ceildiv(ANALYSIS_MAX_NODE, NODE_PARALLEL) avg=ceildiv(ANALYSIS_AVG_NODE, NODE_PARALLEL)

        // assume MP 0 has the max message
        mp_out_t message;
        mp_out[0][nd_offset] >> message;

        // Get the aggregate message for the current node from all MPs
        for ( int mp = 1; mp < EDGE_PARALLEL; mp++ )
        {
            mp_out_t mp_message = FM_TYPE(0);
            mp_out[mp][nd_offset] >> mp_message;

            // Aggregate the max message from all MPs
            for ( int emb = 0; emb < EMB_DIM; emb++ )
            {
#pragma HLS UNROLL
                message[emb] = (message[emb] > mp_message[emb]) ? message[emb] : mp_message[emb];
            }
        }

        mp_out_t aggregated_message = message == noMessages ? FM_TYPE(0) : message;

        // Write the aggregated message to the Node Embedding stream
        ne_in << aggregated_message;

    }
}