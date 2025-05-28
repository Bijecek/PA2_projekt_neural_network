#include <cudaDefs.h>
#include <random>
#include <algorithm>
#include <iostream>

using std::cout;
using std::endl;

constexpr unsigned int THREADS_PER_BLOCK = 128;

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

struct Layer {
	int in;

	int out;
	// 2D matice jako 1D pole
	float* weights;

	// 1D matice
	float* biases;

	//float* pre_activations;
		
	float* activations;

	float* gradients;
};

// Pomocná funkce pro výpoèet RELU
__device__ float convert_relu(float sum) {
	return sum > 0.0 ? sum : 0.0;
}

// Dopøedný prùchod - jedna vrstva neuronové sítì
__global__ void forward(float* input_data, int input_size, float* weight_matrix, float* bias, float* output_data, int output_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < output_size) {
		float sum = 0.0;

		for (int i = 0; i < input_size; i++) {
			sum += input_data[idx * input_size + i] * weight_matrix[idx * input_size + i];
		}
		sum += bias[idx];
		output_data[idx] = convert_relu(sum);
	}
}

//256 velikost -> pracuju s 128 THREADS_PER_BLOCK
__global__ void compute_loss(float* y_predicted, float* y_true, float* loss, int size, float* loss_output) {
	__shared__ float s_loss[THREADS_PER_BLOCK];
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int next = THREADS_PER_BLOCK;

	float epsilon = 1e-7;

	if (idx < size) {
		loss_output[idx] = y_true[idx] * logf(fmax(y_predicted[idx], epsilon));
		s_loss[idx] = loss_output[idx];
		__syncthreads();

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

// Pomocná funkce pro derivaci RELU
__device__ float derivate_relu(float sum) {
	return sum > 0 ? 1.0 : 0.0;
}

__global__ void backward(float* input, float* activations, int input_size, float* weight_matrix, bool first, float* gradient_in
	, float* gradient_out, int output_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < input_size) {
		if (first) {
			gradient_out[idx] = input[idx] * derivate_relu(activations[idx]);
		}
		else {
			float sum = 0.0;
			for (int i = 0; i < output_size; i++) {
				// Vahova matice je transponovana
				sum += weight_matrix[i * input_size + idx] * gradient_in[i];
			}

			gradient_out[idx] = sum * derivate_relu(activations[idx]);
		}
	}

}
// Funkce pro aktualizaci vah a biasu
__global__ void update_parameters(float* input, float* gradient, float* weight_matrix, float* biases, int input_size, int output_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < input_size) {
		float sum = 0.0;
		for (int i = 0; i < output_size; i++) {
			sum += input[i * input_size + idx] * gradient[i];
		}
		// Learning rate
		sum *= 0.05;
		weight_matrix[idx] += sum;



		// Bias TODO
	}
}


int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	// Hyperparametry
	const int input_size = 1024;
	const int hidden_size = 512;
	const int output_size = 1;
	// Poèet hidden layers
	const int num_hidden = 5;
	const int epochs = 100;
	// TODO: learning rate jako konstant memory
	const float lr = 0.01f;

	// Nahodny generator
	std::mt19937 gen(42);
	std::uniform_real_distribution<float> dist(-0.1f, 0.1f);


	std::vector<Layer> layers;

	// Vstupní -> Hidden1
	layers.push_back({ input_size, hidden_size, nullptr, nullptr, nullptr, nullptr });

	// Hidden1 -> HiddenN
	for (int i = 0; i < num_hidden; i++) {
		layers.push_back({ hidden_size, hidden_size, nullptr, nullptr, nullptr, nullptr });
	}

	// HiddenN -> Výstupní
	layers.push_back({ hidden_size, output_size, nullptr, nullptr, nullptr, nullptr });

	// Buffery pro vstupní data
	float* h_input = new float[input_size];
	float* h_target = new float[output_size];

	// Náhodná inicializace
	// TODO: používat reálná data
	for (int i = 0; i < input_size; i++) h_input[i] = dist(gen);
	for (int i = 0; i < output_size; i++) h_target[i] = dist(gen);


	// Alokace vrstev
	for (auto &layer : layers) {
		// Váhová matice
		checkCudaErrors(cudaMalloc(&layer.weights, layer.in * layer.out * sizeof(float)));
		// Bias
		checkCudaErrors(cudaMalloc(&layer.biases, layer.out * sizeof(float)));
		// Aktivace
		checkCudaErrors(cudaMalloc(&layer.activations, layer.out * sizeof(float)));
		// Gradient
		checkCudaErrors(cudaMalloc(&layer.gradients, layer.out * sizeof(float)));

		// Inicializace defaultních hodnot pro váhy a bias
		// TODO: Generaci pøesunout na GPU
		std::vector<float> temporary_weights(layer.in * layer.out), temporary_biases(layer.out);

		for (auto& v : temporary_weights) v = dist(gen);
		for (auto& v : temporary_biases) v = dist(gen);

		checkCudaErrors(cudaMemcpy(layer.weights, temporary_weights.data(), layer.in * layer.out * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(layer.biases, temporary_biases.data(), layer.out * sizeof(float), cudaMemcpyHostToDevice));

	}

	float* d_input;
	checkCudaErrors(cudaMalloc(&d_input, input_size * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));

	float* d_target;
	checkCudaErrors(cudaMalloc(&d_target, output_size * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_target, h_target, output_size * sizeof(float), cudaMemcpyHostToDevice));


	float* d_loss;
	checkCudaErrors(cudaMalloc(&d_loss, sizeof(float)));



	const int n_of_iterations = 10;

	// Velikost všech výstupù -- 1024 * 1
	const int compute_loss_size = input_size * output_size;

	float* d_output_loss;
	checkCudaErrors(cudaMalloc(&d_output_loss, compute_loss_size * sizeof(float)));

	float* h_output_loss = new float[compute_loss_size];

	dim3 dimBlock{ THREADS_PER_BLOCK,1,1 };
	dim3 dimGrid{ 1,1,1 };
	// Hlavní trénovací smyèka

	for (int iteration = 0; iteration < n_of_iterations; iteration++) {

		// Resetovani loss
		checkCudaErrors(cudaMemset(d_loss, 0, sizeof(float)));


		// Forward fáze
		float* current_input = d_input;
		for (int i = 0; i < layers.size(); i++) {
			Layer& current_layer = layers[i];

			forward << <dimGrid, dimBlock >> > (current_input, current_layer.in, current_layer.weights, current_layer.biases,
				current_layer.activations, current_layer.out);

			// Zmìnit vstup
			current_input = current_layer.activations;
		}
		std::cout << "Forward ok" << std::endl;

		// Poèítání loss -- jako vstup je output z poslední vrstvy
		compute_loss << <dimGrid, dimBlock >> > (layers.back().activations, d_target, d_loss, compute_loss_size, d_output_loss);

		// Pøesun loss pole zpátky na host 
		// Mozna zbytecne
		//checkCudaErrors(cudaMemcpy(h_output_loss, d_output_loss, compute_loss_size * sizeof(float), cudaMemcpyDeviceToHost));
		float* tmp_loss = new float[1];
		checkCudaErrors(cudaMemcpy(tmp_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));

		// VYPSANI CELKOVE LOSS
		cout << "Iteration: " << iteration << " -- loss: " << tmp_loss[0] << std::endl;


		std::cout << "Loss ok" << std::endl;

		// Backward fáze
		for (int i = layers.size() - 1; i >= 0; i--) {
			float* input = (i == layers.size() - 1 ? d_output_loss : nullptr);
			float* activation = layers[i].activations;
			int in_size = layers[i].in;
			float* weight_matrix = layers[i].weights;
			bool first = (i == layers.size() - 1) ? true : false;
			float* gradient_in = (i == layers.size() - 1) ? nullptr : layers[i + 1].gradients;
			float* gradient_out = layers[i].gradients;
			int out_size = layers[i].out;
			

			backward << <dimGrid, dimBlock >> > (input, activation, in_size, weight_matrix, first, gradient_in, gradient_out, out_size);

			//TODO AKTUALIZACE VAH -- kernel update_parameters
			if (i > 0) {
				float* input_activations = (i == 1) ? d_input : layers[i - 1].activations;
				int prev_layer_size = (i == 1) ? input_size : layers[i - 1].out;

				update_parameters << <dimGrid, dimBlock >> > (input_activations, layers[i].gradients, layers[i].weights
					, layers[i].biases, prev_layer_size, layers[i].out);
			}


		}

		std::cout << "Backward ok" << std::endl;



	}


	cout << "That is all ..." << endl;
}
