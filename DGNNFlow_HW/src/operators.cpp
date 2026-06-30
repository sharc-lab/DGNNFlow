#include "dcl.hpp"
#include "utils.hpp"
#include "operators.hpp"

/**
*
* Implementation for elu activation function
* @param   fixed point weight arrays
* @return  None
*/
WT_TYPE elu(WT_TYPE x) {
#pragma HLS INLINE  // Inline the function for better optimization

    // Assuming alpha is 1.0 for ELU
    return (x > 0) ? x : static_cast<WT_TYPE>(hls::exp(x) - 1);
}

void matmul_and_add_bias(
    FM_TYPE result[EMB_DIM], 
    WT_TYPE vector[EMB_DIM * 2], 
    WT_TYPE weight[EMB_DIM][EMB_DIM * 2], 
    WT_TYPE bias[EMB_DIM]
) {
#pragma HLS INLINE
    for (int i = 0; i < EMB_DIM; ++i)
    {
    #pragma HLS PIPELINE II = 1
        FM_TYPE acc = bias[i];
        for (int j = 0; j < EMB_DIM * 2; ++j)
        {
        // #pragma HLS UNROLL
            acc += vector[j] * weight[i][j];
        }
        result[i] = acc;
    }
}

void matvec_multiply_and_add_bias(
    FM_TYPE result[EMB_DIM/2],
    FM_TYPE mat[EMB_DIM/2][EMB_DIM],
    FM_TYPE vec[EMB_DIM], 
    FM_TYPE bias[EMB_DIM/2]
) {
#pragma HLS INLINE
    for (int i = 0; i < EMB_DIM/2; ++i)
    {
    #pragma HLS PIPELINE II = 1
        FM_TYPE acc = bias[i]; // start with bias
        for (int j = 0; j < EMB_DIM; ++j)
        {
        // #pragma HLS UNROLL
            acc += mat[i][j] * vec[j];
        }

        result[i] = acc;
    }
}