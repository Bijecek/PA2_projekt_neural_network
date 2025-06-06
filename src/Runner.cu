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

// Jedno vlákno = jeden sample - jeden neuron
__global__ void forward(const float* __restrict__ input_data, int input_size, const float* __restrict__ weight_matrix, const float* __restrict__ bias, float* __restrict__ output_data, int output_size, bool compute_relu) {
	int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
	int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

	if (sample_id < num_samples && neuron_id < output_size) {
		float sum = 0.0;

		for (int i = 0; i < input_size; i++) {
			sum += input_data[sample_id * input_size + i] * weight_matrix[neuron_id * input_size + i];
		}

		// Přidej bias (input 0,0 -> target 0 by jinak nefungovalo)
		sum += bias[neuron_id];

		if (compute_relu) {
			output_data[sample_id * output_size + neuron_id] = convert_relu(sum);
		}
		else {
			output_data[sample_id * output_size + neuron_id] = convert_sigmoid(sum);
		}
		
	}
}

//256 velikost -> pracuju s 128 THREADS_PER_BLOCK
__global__ void compute_loss(float* y_predicted, float* y_true, float* loss, int size) {
	__shared__ float s_loss[THREADS_PER_BLOCK];
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int next = size/2;

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
	
	int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
	int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

	if (sample_id < num_samples && neuron_id < output_size) {

		if (first) {
			gradient_out[sample_id * output_size + neuron_id] = input[sample_id * output_size + neuron_id] * derivate_sigmoid(activations[sample_id * output_size + neuron_id]);
		}
		else {
			
			float sum = 0.0;
			for (int j = 0; j < next_out_size; j++) {
				sum += weight_matrix[j * output_size + neuron_id] * gradient_in[sample_id * next_out_size + j];
			}

			gradient_out[sample_id * output_size + neuron_id] = sum * derivate_relu(activations[sample_id * output_size + neuron_id]);
			
		}
	}
}
// Funkce pro aktualizaci vah a biasu
// TODO:
//      Přepočítat si změnu váhy pro N vzorků a až potom volat tuhle funkci
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
	const int num_hidden = 1;
	// Počet iterací tréninku
	const int n_of_iterations = 50;
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
		for (auto& v : temporary_biases) v = dist(gen);

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


	checkCudaErrors(cudaMemcpyToSymbol((const void*)&num_samples, &input_size, sizeof(int)));


	// Definice loss pole
	const int gradient_size = output_size * output_dimension;

	float* d_gradient;
	checkCudaErrors(cudaMalloc(&d_gradient, gradient_size * sizeof(float)));

	float* h_gradient = new float[gradient_size];

	const int x_thread_count = 16;
	const int y_thread_count = 16;
	

	dim3 dimBlock{ x_thread_count, y_thread_count ,1 };
	dim3 dimGrid{ 1,1 ,1 };



	// Hlavní trénovací smyčka
	for (int iteration = 0; iteration < n_of_iterations; iteration++) {

		// Resetovani loss
		checkCudaErrors(cudaMemset(d_calculated_loss, 0, sizeof(float)));

		// Forward fáze
		float* current_input = d_input;
		for (int i = 0; i < layers.size(); i++) {
			Layer& current_layer = layers[i];

			
			// Nastavení rozměrů gridu - dynamicky ho upravujeme podle rozměrů <input_size; layers[i].out>
		
			// (4 + 16 - 1) / 16
			unsigned int x_grid_dim = (input_size + x_thread_count - 1) / x_thread_count;
			// (20 + 16 - 1) / 16
			unsigned int y_grid_dim = (layers[i].out + y_thread_count - 1) / y_thread_count;

			dimGrid.x = x_grid_dim;
			dimGrid.y = y_grid_dim;
			dimGrid.z = 1;
				
			cout << "Forward kernel executed with: " << x_grid_dim << " " << y_grid_dim << endl;

			// LOGOVANI
			checkDeviceMatrix<float>(current_layer.activations, input_size * current_layer.out * sizeof(float), 1, input_size * current_layer.out, "%f ", "Before: ");

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
			checkDeviceMatrix<float>(current_layer.activations, input_size *  current_layer.out * sizeof(float), 1, input_size *  current_layer.out, "%f ", "After: ");
		}

		// LOGOVANI - vypis výstupu poslední vrstvy pro všechny vstupy
		checkDeviceMatrix<float>(layers[layers.size() - 1].activations, input_size* layers[layers.size() - 1].out * sizeof(float), 1, input_size* layers[layers.size() - 1].out, "%f ", "Activations: ");

		std::cout << "Forward ok" << std::endl;

		// TODO Nastavit dle batch_size -- momentálně je to na 128
		// S timto pocitame LOSS a GRADIENTy
		dimBlock.x = 128;
		dimBlock.y = 1;
		dimBlock.z = 1;

		dimGrid.x = 1;
		dimGrid.y = 1;
		dimGrid.z = 1;


		// Počítání loss -- jako vstup je output z předposlední do poslední vrstvy (proto size() - 1)
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
		checkDeviceMatrix<float>(d_gradient, gradient_size * sizeof(float), 1, gradient_size, "%f ", "Gradient: ");


		// Copy to GPU
		//checkCudaErrors(cudaMemcpy(d_gradient, h_gradient, gradient_size * sizeof(float), cudaMemcpyHostToDevice));

		// Nastav původní velikosti bloku pro 2D
		dimBlock.x = 16;
		dimBlock.y = 16;
		dimBlock.z = 1;

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

			
			// (4 + 16 - 1) / 16
			// (20 + 16 - 1) / 16
			unsigned int x_grid_dim = (input_size + x_thread_count - 1) / x_thread_count;
			unsigned int y_grid_dim = (layers[i].out + y_thread_count - 1) / y_thread_count;

			dimGrid.x = x_grid_dim;
			dimGrid.y = y_grid_dim;
			dimGrid.z = 1;


			cout << "Backward kernel executed with: " << x_grid_dim << " " << y_grid_dim << endl;

			if (i == layers.size() - 1) {
				backward << <dimGrid, dimBlock >> > (input, activation, in_size, weight_matrix, first, gradient_in, gradient_out, out_size, 0);
			}
			else {
				backward << <dimGrid, dimBlock >> > (input, activation, in_size, weight_matrix, first, gradient_in, gradient_out, out_size, layers[i + 1].out);
			}



			// LOGOVANI
			//checkDeviceMatrix<float>(layers[i].gradients, input_size * layers[i].out * sizeof(float), 1, input_size * layers[i].out, "%f ", "Gradient calc: ");

			//TODO AKTUALIZACE VAH -- kernel update_parameters


			float* input_activations = (i == 0) ? d_input : layers[i - 1].activations;
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
