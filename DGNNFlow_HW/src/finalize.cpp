#include "finalize.hpp"
#include "operators.hpp"
#include "hls_math.h"

void finalize(
    int numNodes,
    FM_TYPE emb[EDGE_PARALLEL][ceildiv(MAX_NODE, EDGE_PARALLEL)][EMB_DIM],
    FM_TYPE weight1[EMB_DIM/2][EMB_DIM],
    FM_TYPE bias1[EMB_DIM/2],
    FM_TYPE weight2[NUM_TASK][EMB_DIM/2], 
    FM_TYPE bias2[NUM_TASK],
    FM_TYPE output[MAX_NODE]
) {
#pragma HLS INLINE

    FM_TYPE hidden[MAX_NODE][EMB_DIM/2];

    // First linear layer
    for (int i = 0; i < numNodes; ++i)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
        matvec_multiply_and_add_bias(hidden[i], weight1, emb[i % EDGE_PARALLEL][i / EDGE_PARALLEL], bias1);
    }

    FM_TYPE local_hidden[MAX_NODE][EMB_DIM/2];
// #pragma HLS ARRAY_PARTITION variable=hidden cyclic factor=8 dim=2
    // Elu activation
    for (int i = 0; i < numNodes; ++i)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE
        for (int j = 0; j < EMB_DIM/2; j++)
        { 
// #pragma HLS UNROLL factor=8
            WT_TYPE val = hidden[i][j];
            local_hidden[i][j] = (val > 0) ? val : static_cast<WT_TYPE>(hls::exp(val) - 1);
        }
    }

    // Second linear layer with ReLU
    for (int i = 0; i < numNodes; ++i)
    {
#pragma HLS LOOP_TRIPCOUNT min=ANALYSIS_MIN_NODE max=ANALYSIS_MAX_NODE avg=ANALYSIS_AVG_NODE   
        // Compute final output
        FM_TYPE acc = bias2[0];
        for (int k = 0; k < EMB_DIM/2; ++k)
        {
// #pragma HLS UNROLL factor=8
            acc += weight2[0][k] * local_hidden[i][k];
        }
        
        // Apply ReLU and store
        // output[i] = hls::signbit(acc) ? FM_TYPE(0) : acc;

        // Apply sigmoid and store
        output[i] = FM_TYPE(1.0) / (FM_TYPE(1.0) + hls::exp(FM_TYPE(0.0) - acc));
    }
}