#include <cudaDefs.h>
#include <random>
#include <algorithm>
#include <iostream>
#include "NeuralNetwork.cuh"
#include "ActivationFunctions.cuh"

using std::cout;
using std::endl;


void setNumSamplesConstant(int input_size) {
	checkCudaErrors(cudaMemcpyToSymbol((const void*)&num_samples, &input_size, sizeof(int)));
}

// Jedno vlákno = jeden sample - jeden neuron
__global__ void forward(const float* __restrict__ input_data, int input_size, const float* __restrict__ weight_matrix, const float* __restrict__ bias, float* __restrict__ output_data, int output_size, int activation_type) {
	int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
	int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

	if (sample_id < num_samples && neuron_id < output_size) {
		float sum = 0.0;

		for (int i = 0; i < input_size; i++) {
			sum += input_data[sample_id * input_size + i] * weight_matrix[neuron_id * input_size + i];
		}

		// Pøidej bias (input 0,0 -> target 0 by jinak nefungovalo)
		sum += bias[neuron_id];

		output_data[sample_id * output_size + neuron_id] = activation_functions[activation_type](sum);

	}
}

//256 velikost -> pracuju s 128 THREADS_PER_BLOCK
__global__ void compute_loss(float* y_predicted, float* y_true, float* loss, int size) {
	__shared__ float s_loss[THREADS_PER_BLOCK];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int next = size / 2;

	float epsilon = 1e-7;

	// TODO SIZE/2
	if (idx < size) {
		s_loss[idx] = -y_true[idx] * logf(fmax(y_predicted[idx], epsilon));
		__syncthreads();

		if (idx + next > size) {
			return;
		}

		while (next > 0) {
			s_loss[idx] += s_loss[idx + next];

			__syncthreads();

			next >>= 1;

			if (idx > next) {
				return;
			}
		}
		if (idx == 0) {
			loss[0] = s_loss[0];
		}
	}
}

__global__ void compute_gradient(float* y_pred, float* y_true, float* gradient, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		gradient[idx] = y_pred[idx] - y_true[idx];
	}
}
__global__ void backward(float* input, float* activations, int input_size, float* weight_matrix, bool first, float* gradient_in
	, float* gradient_out, int output_size, int next_out_size, int activation_type) {

	int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
	int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

	if (sample_id < num_samples && neuron_id < output_size) {

		if (first) {
			gradient_out[sample_id * output_size + neuron_id] = input[sample_id * output_size + neuron_id] * derivate_activation_functions[activation_type](activations[sample_id * output_size + neuron_id]);
		}
		else {

			float sum = 0.0;
			for (int j = 0; j < next_out_size; j++) {
				sum += weight_matrix[j * output_size + neuron_id] * gradient_in[sample_id * next_out_size + j];
			}

			gradient_out[sample_id * output_size + neuron_id] = sum * derivate_activation_functions[activation_type](activations[sample_id * output_size + neuron_id]);

		}
	}
}
// Funkce pro aktualizaci vah a biasu
// TODO:
//      Pøepoèítat si zmìnu váhy pro N vzorkù a až potom volat tuhle funkci
__global__ void update_parameters(float* input, float* gradient, float* weight_matrix, float* biases, int input_size, int output_size) {
	int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
	int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

	if (sample_id < num_samples) {
		// Learning rate
		float learning_rate = 0.1f;


		// Get the gradient for the current sample and output neuron
		float grad = gradient[sample_id * output_size + neuron_id];

		// Update each weight for this output neuron
		for (int i = 0; i < input_size; i++) {
			// weight_matrix is [output_size][input_size]
			atomicAdd(&weight_matrix[neuron_id * input_size + i], -learning_rate * input[sample_id * input_size + i] * grad);

		}

		// Update bias
		atomicAdd(&biases[neuron_id], -learning_rate * grad);

	}
}