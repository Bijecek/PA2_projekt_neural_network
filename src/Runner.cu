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

void create_input(TrainingContext& tc, int neurons) {
	tc.input_dim = neurons;
}
void create_dense(TrainingContext& tc, int neurons, ActivationFunction act) {
	// Pokud se jedná o první vrstvu
	if (tc.layers.empty()) {
		tc.layers.push_back(createDenseLayer(tc.input_dim, neurons, act));
	}
	else {
		int last_size = tc.layers[tc.layers.size() - 1].out;
		tc.layers.push_back(createDenseLayer(last_size, neurons, act));
	}
}
void build_network(TrainingContext& tc) {
	// GPU ALOKACE
	for (auto& layer : tc.layers) {
		initLayer(layer, tc.input_size);
	}

	cout << "Network architecture:" << endl;
	cout << "Input: " << tc.input_dim << " neurons" << endl;
	for (int i = 0; i < tc.layers.size(); i++) {
		auto& layer = tc.layers[i];
		cout << "Hidden layer " << i << ": " << layer.out << " neurons | activation: " << getActivationFunction(layer.activation) << endl;
	}
	//cout << "Input: " << tc.input_dim << "neurons" << endl;
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
	create_input(tc, 2);
	create_dense(tc, 20, ActivationFunction::RELU);
	create_dense(tc, 20, ActivationFunction::RELU);
	create_dense(tc, 1, ActivationFunction::SIGMOID);

	build_network(tc);
	
	// HLAVNI TRENOVACI SMYCKA
	train(tc);

	cout << "That is all ..." << endl;
}
