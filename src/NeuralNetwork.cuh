#pragma once

#include <cudaDefs.h>
#include "ActivationFunctions.cuh"

__constant__ int num_samples;

constexpr unsigned int THREADS_PER_BLOCK = 128;

// Pole pointerù na aktivaèní funkce
__device__ float (*activation_functions[2])(float) = { convert_relu, convert_sigmoid};

__device__ float (*derivate_activation_functions[2])(float) = { derivate_relu, derivate_sigmoid };

void setNumSamplesConstant(int input_size);

__global__ void forward(const float* __restrict__ input_data, int input_size, const float* __restrict__ weight_matrix, const float* __restrict__ bias, float* __restrict__ output_data, int output_size, int activation_type);

__global__ void compute_loss(float* y_predicted, float* y_true, float* loss, int size);

__global__ void compute_gradient(float* y_pred, float* y_true, float* gradient, int size);

__global__ void backward(float* input, float* activations, int input_size, float* weight_matrix, bool first, float* gradient_in
	, float* gradient_out, int output_size, int next_out_size, int activation_type);

__global__ void update_parameters(float* input, float* gradient, float* weight_matrix, float* biases, int input_size, int output_size);

