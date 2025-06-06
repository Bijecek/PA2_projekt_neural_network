#pragma once
#include <cudaDefs.h>

constexpr unsigned int THREADS_PER_BLOCK = 128;

//256 velikost -> pracuju s 128 THREADS_PER_BLOCK
__global__ void compute_loss(float* y_predicted, float* y_true, float* loss, int size);
__global__ void compute_gradient(float* y_pred, float* y_true, float* gradient, int size);