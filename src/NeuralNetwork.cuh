#pragma once

#include <cudaDefs.h>
#include <random>
#include <algorithm>
#include <iostream>

#include "Datasets.cuh"
#include "ActivationFunctions.cuh"
#include "Layer.cuh"

struct KernelSettings {
	unsigned int x_thread_count;
	unsigned int y_thread_count;
	dim3 dimBlock;
	dim3 dimGrid;
};

struct TrainingContext {
	int input_size;
	int input_dim;
	int output_size;
	int output_dim;
	int hidden_size;
	int num_hidden;
	int n_of_iterations;

	float* d_input = nullptr;
	float* d_target = nullptr;
	float* d_gradient = nullptr;
	float* d_loss = nullptr;

	std::vector<Layer> layers;
	Dataset dataset;
	KernelSettings kernel_settings;
};

__constant__ int num_samples;
__constant__ float learning_rate;

extern std::vector<std::pair<int, ActivationFunction>> layer_specifications;

constexpr unsigned int THREADS_PER_BLOCK = 128;

// Pole pointerù na aktivaèní funkce
__device__ float (*activation_functions[2])(float) = { convert_relu, convert_sigmoid};

__device__ float (*derivate_activation_functions[2])(float) = { derivate_relu, derivate_sigmoid };

void setNumSamplesConstant(int input_size);

void setLearningRateConstant(int learning_rate);

__global__ void forward(const float* __restrict__ input_data, int input_size, const float* __restrict__ weight_matrix, const float* __restrict__ bias, float* __restrict__ output_data, int output_size, int activation_type);

__global__ void compute_loss(float* y_predicted, float* y_true, float* loss, int size);

__global__ void compute_gradient(float* y_pred, float* y_true, float* gradient, int size);

__global__ void backward(float* activations, int input_size, float* weight_matrix, bool first, float* gradient_in
	, float* gradient_out, int output_size, int next_out_size, int activation_type);

__global__ void update_parameters(float* input, float* gradient, float* weight_matrix, float* biases, int input_size, int output_size);

void forward_phase(TrainingContext& tc, bool enable_logging);

void loss_and_gradient_phase(TrainingContext& tc, int iteration, bool enable_logging);

void backward_phase(TrainingContext& tc, bool enable_logging);

