#ifndef OPERATORS_HPP
#define OPERATORS_HPP

#include "dcl.hpp"

// Implementation for elu activation function
// @param   fixed point weight arrays
// @return  None
WT_TYPE elu(WT_TYPE x);


void matmul_and_add_bias(
    FM_TYPE result[EMB_DIM], 
    WT_TYPE vector[EMB_DIM * 2], 
    WT_TYPE weight[EMB_DIM][EMB_DIM * 2], 
    WT_TYPE bias[EMB_DIM]
);

// Utility function to perform matrix-vector multiplication and add bias
void matvec_multiply_and_add_bias(
    FM_TYPE result[EMB_DIM/2],
    FM_TYPE mat[EMB_DIM/2][EMB_DIM],
    FM_TYPE vec[EMB_DIM],
    FM_TYPE bias[EMB_DIM/2]
);

#endif