#ifndef FINALIZE_HPP
#define FINALIZE_HPP

#include "dcl.hpp"

// Output layer forward pass function
void finalize(
    int numNodes,
    FM_TYPE emb[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE weight1[EMB_DIM/2][EMB_DIM],
    FM_TYPE bias1[EMB_DIM/2],
    FM_TYPE weight2[NUM_TASK][EMB_DIM/2],
    FM_TYPE bias2[NUM_TASK],
    FM_TYPE output[MAX_NODE]
);

#endif // FINALIZE_HPP