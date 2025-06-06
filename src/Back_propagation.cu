#include "Back_propagation.cuh"

__constant__ int num_samples;

void setNumSamplesConstant(int input_size) {
	checkCudaErrors(cudaMemcpyToSymbol((const void*)&num_samples, &input_size, sizeof(int)));
}

__global__ void apply_dropout_forward(float* activations, bool* mask, float rate, int total_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < total_size) {
		// Aktivace buï zùstane, nebo je vynulována (pøi tréninku)
		activations[idx] *= mask[idx] ? (1.0f / (1.0f - rate)) : 0.0f;
	}
}

// Jedno vlákno = jeden sample - jeden neuron
__global__ void forward_pass(const float* __restrict__ input_data, int input_size, const float* __restrict__ weight_matrix, const float* __restrict__ bias, float* __restrict__ output_data, int output_size, int activation) {
	int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
	int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

	if (sample_id < num_samples && neuron_id < output_size) {
		float sum = 0.0;

		for (int i = 0; i < input_size; i++) {
			sum += input_data[sample_id * input_size + i] * weight_matrix[neuron_id * input_size + i];
		}

		// Pøidej bias (input 0,0 -> target 0 by jinak nefungovalo)
		sum += bias[neuron_id];

		if (activation) {
			output_data[sample_id * output_size + neuron_id] = convert_sigmoid(sum);
		}
		else {
			output_data[sample_id * output_size + neuron_id] = convert_relu(sum);
		}
	}
}

__global__ void apply_dropout_backward(float* gradients, bool* mask, float rate, int total_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < total_size) {
		// Gradient vynuluj, pokud byl neuron vypnutý
		gradients[idx] *= mask[idx] ? (1.0f / (1.0f - rate)) : 0.0f;
	}
}

__global__ void backward_pass(float* input, float* activations, int input_size, float* weight_matrix, bool first, float* gradient_in
	, float* gradient_out, int output_size, int next_out_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_samples) {

		if (first) {
			for (int i = 0; i < output_size; i++) {
				gradient_out[idx * output_size + i] = input[idx * output_size + i] * derivate_sigmoid(activations[idx * output_size + i]);
			}
		}
		else {
			for (int i = 0; i < output_size; i++) {
				float sum = 0.0;
				for (int j = 0; j < next_out_size; j++) {
					sum += weight_matrix[j * output_size + i] * gradient_in[idx * next_out_size + j];
				}


				gradient_out[idx * output_size + i] = sum * derivate_relu(activations[idx * output_size + i]);
			}
		}
	}
}

// Funkce pro aktualizaci vah a biasu
__global__ void update_parameters(float* input, float* gradient, float* weight_matrix, float* biases, int input_size, int output_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < num_samples) {
		// Learning rate
		float learning_rate = 0.1f;

		// For each output neuron
		for (int j = 0; j < output_size; j++) {
			// Get the gradient for the current sample and output neuron
			float grad = gradient[idx * output_size + j];

			// Update each weight for this output neuron
			for (int i = 0; i < input_size; i++) {
				// weight_matrix is [output_size][input_size]
				atomicAdd(&weight_matrix[j * input_size + i], -learning_rate * input[idx * input_size + i] * grad);
				//atomicAdd(&weight_matrix[j * input_size + i], 0.1);
			}

			// Update bias
			atomicAdd(&biases[j], -learning_rate * grad);
		}
	}
}