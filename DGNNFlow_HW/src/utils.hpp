#ifndef __UTIL_HPP__
#define __UTIL_HPP__

#include "hls_math.h"

template <typename T>
static constexpr T ceildiv(T dividend, T divisor)
{
#pragma HLS INLINE
    return (dividend + divisor - 1) / divisor;
}

template <typename T>
static constexpr T roundup(T dividend, T divisor)
{
#pragma HLS INLINE
    return ceildiv(dividend, divisor) * divisor;
}

template <typename T>
static constexpr T min(T a, T b)
{
#pragma HLS INLINE
    return (a < b) ? a : b;
}

template <typename T>
static constexpr T max(T a, T b)
{
#pragma HLS INLINE
    return (a > b) ? a : b;
}

template <typename T>
static constexpr T ap_fixed_relu(T x)
{
#pragma HLS INLINE
    return hls::signbit(x) ? T(0) : x;
}

template <typename T>
static constexpr T ap_fixed_epsilon()
{
#pragma HLS INLINE
    return T(1.0 / (1 << (T::width - T::iwidth)));
}

template <typename T>
static constexpr T ap_fixed_min()
{
#pragma HLS INLINE
    return T(-(1 << (T::iwidth - 1)));
}

template <typename T>
static constexpr T ap_fixed_max()
{
#pragma HLS INLINE
    return T((1 << (T::iwidth - 1)) - (1.0 / (1 << (T::width - T::iwidth))));
}

#endif
