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
	int num_samples;
	int input_dim;
	int output_dim;
	int hidden_size;
	int num_hidden;
	int n_of_iterations;
	float learning_rate;
	int batch_size;
	int total_batch_count;

	float* d_input = nullptr;
	float* d_target = nullptr;
	float* d_gradient = nullptr;
	float* d_loss = nullptr;
	float* d_accuracy = nullptr;
	float* d_tp = nullptr;
	float* d_fp = nullptr;
	float* d_fn = nullptr;

	std::vector<float> batch_loss, batch_accuracy, batch_f1;

	std::vector<Layer> layers;
	Dataset dataset;
	KernelSettings kernel_settings;
};


extern std::vector<std::pair<int, ActivationFunction>> layer_specifications;

constexpr unsigned int THREADS_PER_BLOCK = 1028;

// Pole pointerù na aktivaèní funkce
__device__ float (*activation_functions[2])(float) = { convert_relu, convert_sigmoid};

__device__ float (*derivate_activation_functions[2])(float) = { derivate_relu, derivate_sigmoid };

void setNumSamplesConstant(int input_size);

void setLearningRateConstant(float learning_rate);

__global__ void forward(const float* __restrict__ input_data, int input_size, const float* __restrict__ weight_matrix, const float* __restrict__ bias, float* __restrict__ output_data, int output_size, int activation_type);

__global__ void compute_loss(const float* __restrict__ y_predicted, const float* __restrict__ y_true, float* loss, const int size);

__global__ void compute_gradient(float* y_pred, float* y_true, float* gradient, int size, float* accuracy, float* d_tp, float* d_fp, float* d_fn);

__global__ void backward(float* activations, int input_size, float* weight_matrix, bool first, float* gradient_in
	, float* gradient_out, int output_size, int next_out_size, int activation_type);

__global__ void update_parameters(float* input, float* gradient, float* weight_matrix, float* biases, int input_size, int output_size);

void forward_phase(TrainingContext& tc, bool enable_logging, bool enable_results, int actual_batch_size, std::vector<float> target_data);

void loss_and_gradient_phase(TrainingContext& tc, int iteration, int actual_batch_size, bool enable_logging);

void backward_phase(TrainingContext& tc, bool enable_logging);

