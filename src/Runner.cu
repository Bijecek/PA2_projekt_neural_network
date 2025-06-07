#include <cudaDefs.h>
#include <random>
#include <algorithm>
#include <iostream>

#include "NeuralNetwork.cuh"

using std::cout;
using std::endl;

//constexpr unsigned int THREADS_PER_BLOCK = 128;

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();


void allocate_input_data(TrainingContext& tc) {
	checkCudaErrors(cudaMalloc(&tc.d_input, tc.input_size * tc.input_dim * sizeof(float)));
	checkCudaErrors(cudaMemcpy(tc.d_input, tc.dataset.input.data(), tc.input_size * tc.input_dim * sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&tc.d_target, tc.output_size * tc.output_dim * sizeof(float)));
	checkCudaErrors(cudaMemcpy(tc.d_target, tc.dataset.target.data(), tc.output_size * tc.output_dim * sizeof(float), cudaMemcpyHostToDevice));
}

// TODO CUSTOM NETWORK
void create_network(TrainingContext& tc) {
	// Vstupní -> Hidden1
	tc.layers.push_back(createDenseLayer(tc.input_dim, tc.hidden_size, ActivationFunction::RELU));

	// Hidden1 -> HiddenN
	for (int i = 0; i < tc.num_hidden; i++) {
		tc.layers.push_back(createDenseLayer(tc.hidden_size, tc.hidden_size, ActivationFunction::RELU));
	}

	// HiddenN -> Výstupní
	tc.layers.push_back(createDenseLayer(tc.hidden_size, tc.output_dim, ActivationFunction::SIGMOID));

	// GPU ALOKACE
	for (auto& layer : tc.layers) {
		initLayer(layer, tc.input_size);
	}
}

void train(TrainingContext& tc) {
	float* d_calculated_loss;
	checkCudaErrors(cudaMalloc(&d_calculated_loss, sizeof(float)));


	setNumSamplesConstant(tc.input_size);


	// Definice loss pole
	const int gradient_size = tc.output_size * tc.output_dim;

	float* d_gradient;
	checkCudaErrors(cudaMalloc(&d_gradient, gradient_size * sizeof(float)));

	float* h_gradient = new float[gradient_size];

	const int x_thread_count = 16;
	const int y_thread_count = 16;


	dim3 dimBlock{ x_thread_count, y_thread_count ,1 };
	dim3 dimGrid{ 1,1 ,1 };

	checkCudaErrors(cudaMalloc(&tc.d_loss, sizeof(float)));
	checkCudaErrors(cudaMalloc(&tc.d_gradient, gradient_size * sizeof(float)));


	//******************************************************************************************|
	//								         MAIN TRAINING LOOP						            |					
	//******************************************************************************************|
	for (int iteration = 0; iteration < tc.n_of_iterations; iteration++) {

		// Resetovani loss
		checkCudaErrors(cudaMemset(d_calculated_loss, 0, sizeof(float)));

		// Forward fáze
		forward_phase(tc);

		// LOSS A GRADIENT FAZE
		loss_and_gradient_phase(tc, iteration);

		// Backward fáze
		backward_phase(tc);

	}
}
int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	
	TrainingContext tc;
	tc.input_size = 4;
	tc.input_dim = 2;
	tc.output_size = 4;
	tc.output_dim = 1;
	tc.hidden_size = 20;
	tc.num_hidden = 1;
	tc.n_of_iterations = 50;

	KernelSettings kernel_settings;
	kernel_settings.x_thread_count = 16;
	kernel_settings.y_thread_count = 16;
	kernel_settings.dimBlock = { kernel_settings.x_thread_count, kernel_settings.y_thread_count ,1 };
	kernel_settings.dimGrid = { 1, 1 ,1 };

	tc.kernel_settings = kernel_settings;

	// VYBER DATASET A ALOKUJ DATA
	tc.dataset = getDatasetByName("dataset1");
	allocate_input_data(tc);

	// VYTVOR ARCHITEKTURU NEURONOVE SITE
	create_network(tc);
	
	// HLAVNI TRENOVACI SMYCKA
	train(tc);

	cout << "That is all ..." << endl;
}
