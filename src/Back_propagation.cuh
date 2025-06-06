#pragma once
#include <cudaDefs.h>
#include "Activation_functions.cuh"

void setNumSamplesConstant(int input_size);

// Jedno vlákno = jeden sample - jeden neuron
__global__ void forward_pass(const float* __restrict__ input_data, int input_size, const float* __restrict__ weight_matrix, const float* __restrict__ bias, float* __restrict__ output_data, int output_size, int activation);
__global__ void apply_dropout_forward(float* activations, bool* mask, float rate, int total_size);

__global__ void backward_pass(float* input, float* activations, int input_size, float* weight_matrix, bool first, float* gradient_in, float* gradient_out, int output_size, int next_out_size);
__global__ void apply_dropout_backward(float* gradients, bool* mask, float rate, int total_size);

__global__ void update_parameters(float* input, float* gradient, float* weight_matrix, float* biases, int input_size, int output_size);