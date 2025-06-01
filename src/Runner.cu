#include <cudaDefs.h>
#include <random>
#include <algorithm>
#include <iostream>

using std::cout;
using std::endl;

constexpr unsigned int THREADS_PER_BLOCK = 128;

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();

__constant__ int num_samples;

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

// Pomocná funkce pro výpočet RELU
__device__ float convert_relu(float sum) {
	return sum > 0.0 ? sum : 0.0;
}

// Pomocná funkce pro výpočet SIGMOID
__device__ float convert_sigmoid(float sum) {
	return 1.0f / (1.0f + expf(-sum));
}

// Dopředný průchod
__global__ void forward(float* input_data, int input_size, float* weight_matrix, float* bias, float* output_data, int output_size, bool compute_relu) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_samples) {

		for (int j = 0; j < output_size; j++) {
			float sum = 0.0;

			for (int i = 0; i < input_size; i++) {
				sum += input_data[idx * input_size + i] * weight_matrix[j * input_size + i];
			}

			// Přidej bias (input 0,0 -> target 0 by jinak nefungovalo)
			sum += bias[j];
			
			if (compute_relu) {
				output_data[idx * output_size + j] = convert_relu(sum);
			}
			else {
				output_data[idx * output_size + j] = convert_sigmoid(sum);
			}
		}
	}
}

//256 velikost -> pracuju s 128 THREADS_PER_BLOCK
__global__ void compute_loss(float* y_predicted, float* y_true, float* loss, int size) {
	__shared__ float s_loss[THREADS_PER_BLOCK*2];
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int next = size/2;

	float epsilon = 1e-7;
	
	// TODO SIZE/2
	if (idx < size) {
		s_loss[idx] = -y_true[idx] * logf(fmax(y_predicted[idx], epsilon));
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

__global__ void compute_gradient(float* y_pred, float* y_true, float* gradient, int size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		gradient[idx] = y_pred[idx] - y_true[idx];
	}
}

// Pomocná funkce pro derivaci RELU
__device__ float derivate_relu(float sum) {
	return sum > 0 ? 1.0 : 0.0;
}

// Pomocná funkce pro derivaci SIGMOID
__device__ float derivate_sigmoid(float sum) {
	return sum * (1.0f - sum);
}

__global__ void backward(float* input, float* activations, int input_size, float* weight_matrix, bool first, float* gradient_in
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


int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	// Parametry vstupních dat
	//const int input_size = 4;
	const int input_size = 4;
	const int input_dimension = 2;
	//const int output_size = 4;
	const int output_size = 4;
	const int output_dimension = 1;

	// Hyperparametry
	const int hidden_size = 20;
	// Počet hidden layers
	const int num_hidden = 0;
	// Počet iterací tréninku
	const int n_of_iterations = 100;
	// TODO: learning rate jako konstant memory
	//const float lr = 0.01f;

	// Nahodny generator
	std::mt19937 gen(42);
	std::uniform_real_distribution<float> dist(-1.0, 1.0f);


	std::vector<Layer> layers;

	// Vstupní -> Hidden1
	layers.push_back({ input_dimension, hidden_size, nullptr, nullptr, nullptr, nullptr });

	// Hidden1 -> HiddenN
	for (int i = 0; i < num_hidden; i++) {
		layers.push_back({ hidden_size, hidden_size, nullptr, nullptr, nullptr, nullptr });
	}

	// HiddenN -> Výstupní
	layers.push_back({ hidden_size, output_dimension, nullptr, nullptr, nullptr, nullptr });

	// Výstupní -> Konec
	//layers.push_back({ output_dimension, 0, nullptr, nullptr, nullptr, nullptr });

	// Buffery pro vstupní data + XOR data
	/*
	float* h_input = new float[input_size * input_dimension] 
		{
			0, 0, 0, 1, 1, 0, 1, 1
		};
	float* h_target = new float[output_size * output_dimension] 
		{
			0, 1, 1, 0
		};
	*/

	float* h_input = new float[input_size * input_dimension]
		{
			0, 0, 1, 1, 0, 1, 1, 0
		};
	float* h_target = new float[output_size * output_dimension]
		{
			0, 0, 1, 1
		};



	// Alokace vrstev
	for (auto &layer : layers) {
		// Váhová matice
		checkCudaErrors(cudaMalloc(&layer.weights, layer.in * layer.out * sizeof(float)));
		// Bias
		checkCudaErrors(cudaMalloc(&layer.biases, layer.out * sizeof(float)));
		// Aktivace
		checkCudaErrors(cudaMalloc(&layer.activations, input_size * layer.out * sizeof(float)));
		// Gradient
		checkCudaErrors(cudaMalloc(&layer.gradients, input_size * layer.out * sizeof(float)));

		// Inicializace defaultních hodnot pro váhy a bias
		// TODO: Generaci přesunout na GPU
		std::vector<float> temporary_weights(layer.in * layer.out), temporary_biases(layer.out);

		for (auto& v : temporary_weights) v = dist(gen);
		for (auto& v : temporary_biases) v = 0.1;//dist(gen);

		checkCudaErrors(cudaMemcpy(layer.weights, temporary_weights.data(), layer.in * layer.out * sizeof(float), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(layer.biases, temporary_biases.data(), layer.out * sizeof(float), cudaMemcpyHostToDevice));

		// LOGOVANI
		//checkDeviceMatrix<float>(layer.weights, layer.in * layer.out * sizeof(float), 1, layer.in * layer.out, "%f ", "Weights: ");

	}

	float* d_input;
	checkCudaErrors(cudaMalloc(&d_input, input_size * input_dimension * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_input, h_input, input_size * input_dimension * sizeof(float), cudaMemcpyHostToDevice));

	float* d_target;
	checkCudaErrors(cudaMalloc(&d_target, output_size * output_dimension * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_target, h_target, output_size * output_dimension * sizeof(float), cudaMemcpyHostToDevice));


	float* d_calculated_loss;
	checkCudaErrors(cudaMalloc(&d_calculated_loss, sizeof(float)));


	
	//int h_num_samples = input_size;
	//int d_num_samples;
	//checkCudaErrors(cudaMalloc(&d_num_samples, sizeof(int)));
	//checkCudaErrors(cudaMemcpy(&d_num_samples, &h_num_samples, sizeof(int), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpyToSymbol((const void*)&num_samples, &input_size, sizeof(int)));


	// Definice loss pole
	const int gradient_size = output_size * output_dimension;

	float* d_gradient;
	checkCudaErrors(cudaMalloc(&d_gradient, gradient_size * sizeof(float)));

	float* h_gradient = new float[gradient_size];

	dim3 dimBlock{ THREADS_PER_BLOCK,1,1 };
	dim3 dimGrid{ 1,1,1 };


	// Hlavní trénovací smyčka
	for (int iteration = 0; iteration < n_of_iterations; iteration++) {

		// Resetovani loss
		checkCudaErrors(cudaMemset(d_calculated_loss, 0, sizeof(float)));


		// Forward fáze
		float* current_input = d_input;
		for (int i = 0; i < layers.size(); i++) {
			Layer& current_layer = layers[i];

			// LOGOVANI
			//checkDeviceMatrix<float>(current_layer.activations, input_size * current_layer.out * sizeof(float), 1, input_size * current_layer.out, "%f ", "Before: ");

			// Pokud se jedná o poslední vrstvu -> nepočítáme RELU ale SIGMOID
			if (i == layers.size() - 1) {
				forward << <dimGrid, dimBlock >> > (current_input, current_layer.in, current_layer.weights, current_layer.biases,
					current_layer.activations, current_layer.out, false);
			}
			else {
				forward << <dimGrid, dimBlock >> > (current_input, current_layer.in, current_layer.weights, current_layer.biases,
					current_layer.activations, current_layer.out, true);
			}

			// Změnit vstup
			current_input = current_layer.activations;

			// LOGOVANI
			//checkDeviceMatrix<float>(current_layer.activations, input_size *  current_layer.out * sizeof(float), 1, input_size *  current_layer.out, "%f ", "After: ");
		}

		// LOGOVANI - vypis výstupu poslední vrstvy pro všechny vstupy
		checkDeviceMatrix<float>(layers[layers.size() - 1].activations, input_size* layers[layers.size() - 1].out * sizeof(float), 1, input_size* layers[layers.size() - 1].out, "%f ", "Activations: ");

		std::cout << "Forward ok" << std::endl;

		// Počítání loss -- jako vstup je output z předposlední do poslední vrstvy (proto size() - 2)
		compute_loss << <dimGrid, dimBlock >> > (layers[layers.size() - 1].activations, d_target, d_calculated_loss, gradient_size);

		// Přesun loss pole zpátky na host 
		// Mozna zbytecne
		//checkCudaErrors(cudaMemcpy(h_output_loss, d_output_loss, compute_loss_size * sizeof(float), cudaMemcpyDeviceToHost));
		float* tmp_loss = new float[1];
		checkCudaErrors(cudaMemcpy(tmp_loss, d_calculated_loss, sizeof(float), cudaMemcpyDeviceToHost));

		// VYPSANI CELKOVE categorical crossentropy LOSS
		cout << "Iteration: " << iteration << " -- loss: " << tmp_loss[0] << std::endl;


		std::cout << "Loss ok" << std::endl;


		compute_gradient << <dimGrid, dimBlock >> > (layers[layers.size() - 1].activations, d_target, d_gradient, gradient_size);
		// LOGOVANI
		//checkDeviceMatrix<float>(d_gradient, gradient_size * sizeof(float), 1, gradient_size, "%f ", "Gradient: ");


		// Copy to GPU
		//checkCudaErrors(cudaMemcpy(d_gradient, h_gradient, gradient_size * sizeof(float), cudaMemcpyHostToDevice));


		// Backward fáze
		for (int i = layers.size() - 1; i >= 0; i--) {
			float* input = (i == layers.size() - 1 ? d_gradient : layers[i].gradients);
			float* activation = layers[i].activations;
			int in_size = layers[i].in;
			int out_size = layers[i].out;
			float* weight_matrix = (i == layers.size() - 1) ? nullptr : layers[i+1].weights;
			bool first = (i == layers.size() - 1) ? true : false;
			float* gradient_in = (i == layers.size() - 1) ? nullptr : layers[i+1].gradients;
			float* gradient_out = layers[i].gradients;
			
			if (i == layers.size() - 1) {
				backward << <dimGrid, dimBlock >> > (input, activation, in_size, weight_matrix, first, gradient_in, gradient_out, out_size, 0);
			}
			else {
				backward << <dimGrid, dimBlock >> > (input, activation, in_size, weight_matrix, first, gradient_in, gradient_out, out_size, layers[i+1].out);
			}

			// LOGOVANI
			//checkDeviceMatrix<float>(layers[i].gradients, input_size * layers[i].out * sizeof(float), 1, input_size * layers[i].out, "%f ", "Gradient calc: ");

			//TODO AKTUALIZACE VAH -- kernel update_parameters
			
			
			float* input_activations = (i == 0) ? d_input : layers[i-1].activations;
			//int prev_layer_size = (i == 0) ? input_size : layers[i - 1].out;

			update_parameters << <dimGrid, dimBlock >> > (input_activations, layers[i].gradients, layers[i].weights
				, layers[i].biases, layers[i].in, layers[i].out);
			

		}
		for (int i = 0; i < layers.size(); i++) {
			// LOGOVANI
			//checkDeviceMatrix<float>(layers[i].weights, layers[i].in * layers[i].out * sizeof(float), 1, layers[i].in * layers[i].out, "%f ", "Weights: ");
		}

		std::cout << "Backward ok" << std::endl;



	}


	cout << "That is all ..." << endl;
}
