#pragma once
#include <cudaDefs.h>
#include <math.h>

inline __device__ float convert_relu(float sum) {
    return sum > 0.0f ? sum : 0.0f;
}

inline __device__ float convert_sigmoid(float sum) {
    return 1.0f / (1.0f + expf(-sum));
}

inline __device__ float derivate_relu(float sum) {
    return sum > 0.0f ? 1.0f : 0.0f;
}

inline __device__ float derivate_sigmoid(float sum) {
    return sum * (1.0f - sum);
}
