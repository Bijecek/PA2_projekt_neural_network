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
	checkCudaErrors(cudaMalloc(&tc.d_input, tc.num_samples * tc.input_dim * sizeof(float)));
	checkCudaErrors(cudaMemcpy(tc.d_input, tc.dataset.input.data(), tc.num_samples * tc.input_dim * sizeof(float), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc(&tc.d_target, tc.num_samples * tc.output_dim * sizeof(float)));
	checkCudaErrors(cudaMemcpy(tc.d_target, tc.dataset.target.data(), tc.num_samples * tc.output_dim * sizeof(float), cudaMemcpyHostToDevice));
}

void create_input(TrainingContext& tc, int neurons) {
	tc.input_dim = neurons;
}
void create_dense(TrainingContext& tc, int neurons, ActivationFunction act, LayerLogicalType type) {
	// Pokud se jedná o první vrstvu
	if (type == LayerLogicalType::INPUT) {
		// Nastavení vstupní dimenze (počet neuronů vstupu)
		tc.input_dim = neurons;
	}
	else if(type == LayerLogicalType::OUTPUT){
		tc.output_dim = neurons;
	}
	layer_specifications.push_back(std::make_pair(neurons, act));
	
}
void build_network(TrainingContext& tc) {

	// Naplnění TrainingContext vrstev
	for (int i = 0; i < layer_specifications.size() - 1;i++) {
		auto& first = layer_specifications[i];
		auto& next = layer_specifications[i + 1];

		tc.layers.push_back(createDenseLayer(first.first, next.first, next.second));
	}

	// GPU ALOKACE
	for (auto& layer : tc.layers) {
		initLayer(layer, tc.num_samples);
	}

	cout << "Network architecture:" << endl;
	cout << "Input: " << tc.input_dim << " neurons" << endl;
	for (int i = 0; i < tc.layers.size(); i++) {
		auto& layer = tc.layers[i];
		cout << "Hidden layer " << i << ": " << layer.out << " neurons | activation: " << getActivationFunction(layer.activation) << endl;
	}
}


void train(TrainingContext& tc, bool enable_logging) {
	float* d_calculated_loss;
	checkCudaErrors(cudaMalloc(&d_calculated_loss, sizeof(float)));


	setNumSamplesConstant(tc.num_samples);

	setLearningRateConstant(tc.learning_rate);


	// Definice loss pole
	const int gradient_size = tc.num_samples * tc.output_dim;

	float* d_gradient;
	checkCudaErrors(cudaMalloc(&d_gradient, gradient_size * sizeof(float)));

	float* h_gradient = new float[gradient_size];

	checkCudaErrors(cudaMalloc(&tc.d_loss, sizeof(float)));
	checkCudaErrors(cudaMalloc(&tc.d_gradient, gradient_size * sizeof(float)));


	//******************************************************************************************|
	//								         MAIN TRAINING LOOP						            |					
	//******************************************************************************************|
	for (int iteration = 0; iteration < tc.n_of_iterations; iteration++) {

		// Resetovani loss
		checkCudaErrors(cudaMemset(d_calculated_loss, 0, sizeof(float)));

		// Forward fáze
		forward_phase(tc, enable_logging);

		// LOSS A GRADIENT FAZE
		loss_and_gradient_phase(tc, iteration, enable_logging);

		// Backward fáze
		backward_phase(tc, enable_logging);

	}
}
int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	
	TrainingContext tc;
	tc.num_samples = 4;
	tc.n_of_iterations = 1000;
	tc.learning_rate = 0.1;

	KernelSettings kernel_settings;
	kernel_settings.x_thread_count = 32;
	kernel_settings.y_thread_count = 32;
	kernel_settings.dimBlock = { kernel_settings.x_thread_count, kernel_settings.y_thread_count ,1 };
	kernel_settings.dimGrid = { 1, 1 ,1 };

	tc.kernel_settings = kernel_settings;

	// VYTVOR ARCHITEKTURU NEURONOVE SITE
	// TODO VETSI VELIKOST 
	create_dense(tc, 2, ActivationFunction::NONE, LayerLogicalType::INPUT);
	//create_dense(tc, 300, ActivationFunction::RELU, LayerLogicalType::OTHER);
	create_dense(tc, 8, ActivationFunction::RELU, LayerLogicalType::OTHER);
	create_dense(tc, 1, ActivationFunction::SIGMOID, LayerLogicalType::OUTPUT);

	build_network(tc);
	
	// VYBER DATASET A ALOKUJ DATA
	tc.dataset = getDatasetByName("dataset1");
	allocate_input_data(tc);


	// HLAVNI TRENOVACI SMYCKA
	train(tc, false);

	cout << "That is all ..." << endl;
}
