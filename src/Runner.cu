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


void train(TrainingContext& tc, int learning_rate, bool enable_logging) {
	float* d_calculated_loss;
	checkCudaErrors(cudaMalloc(&d_calculated_loss, sizeof(float)));


	setNumSamplesConstant(tc.input_size);

	setLearningRateConstant(learning_rate);


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

	int learning_rate = 0.01;

	
	TrainingContext tc;
	tc.input_size = 4;
	tc.input_dim = 2;
	tc.output_size = 4;
	tc.output_dim = 1;
	tc.n_of_iterations = 2000;

	KernelSettings kernel_settings;
	kernel_settings.x_thread_count = 16;
	kernel_settings.y_thread_count = 16;
	kernel_settings.dimBlock = { kernel_settings.x_thread_count, kernel_settings.y_thread_count ,1 };
	kernel_settings.dimGrid = { 1, 1 ,1 };

	tc.kernel_settings = kernel_settings;

	// VYTVOR ARCHITEKTURU NEURONOVE SITE
	// TODO VETSI VELIKOST 
	create_dense(tc, 2, ActivationFunction::NONE, LayerLogicalType::INPUT);
	create_dense(tc, 300, ActivationFunction::RELU, LayerLogicalType::OTHER);
	create_dense(tc, 300, ActivationFunction::RELU, LayerLogicalType::OTHER);
	create_dense(tc, 1, ActivationFunction::SIGMOID, LayerLogicalType::OUTPUT);

	build_network(tc);
	
	// VYBER DATASET A ALOKUJ DATA
	tc.dataset = getDatasetByName("dataset1");
	allocate_input_data(tc);


	// HLAVNI TRENOVACI SMYCKA
	train(tc, learning_rate, false);

	cout << "That is all ..." << endl;
}
