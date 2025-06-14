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
	checkCudaErrors(cudaMalloc(&tc.d_input, tc.batch_size * tc.input_dim * sizeof(float)));

	checkCudaErrors(cudaMalloc(&tc.d_target, tc.batch_size * tc.output_dim * sizeof(float)));
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
	tc.total_batch_count = static_cast<int>(std::ceil(tc.num_samples*1.0 / tc.batch_size));

	// Naplnění TrainingContext vrstev
	for (int i = 0; i < layer_specifications.size() - 1;i++) {
		auto& first = layer_specifications[i];
		auto& next = layer_specifications[i + 1];

		tc.layers.push_back(createDenseLayer(first.first, next.first, next.second));
	}

	// GPU ALOKACE
	for (auto& layer : tc.layers) {
		initLayer(layer, tc.batch_size);
	}

	cout << "Network architecture:" << endl;
	cout << "Input: " << tc.input_dim << " neurons" << endl;
	for (int i = 0; i < tc.layers.size(); i++) {
		auto& layer = tc.layers[i];
		cout << "Hidden layer " << i << ": " << layer.out << " neurons | activation: " << getActivationFunction(layer.activation) << endl;
	}
}

void reset_print_statistics(TrainingContext& tc, int iteration) {
	float batch_loss = 0.0;
	float batch_accuracy = 0.0;

	for (int i = 0; i < tc.total_batch_count; i++) {
		batch_loss += tc.batch_loss[i];
		batch_accuracy += tc.batch_accuracy[i];
	}
	batch_loss /= tc.batch_size;
	batch_accuracy /= (tc.batch_size * tc.total_batch_count);

	// VYPSANI CELKOVE categorical crossentropy LOSS
	cout << "Iteration: " << iteration << " -- batch loss: " << batch_loss << " Batch accuracy: " << batch_accuracy << std::endl;

	// Resetuj batch pole
	tc.batch_accuracy.clear();
	tc.batch_loss.clear();
}

void train(TrainingContext& tc, bool enable_logging) {
	float* d_calculated_loss;
	checkCudaErrors(cudaMalloc(&d_calculated_loss, sizeof(float)));


	setNumSamplesConstant(tc.batch_size);

	setLearningRateConstant(tc.learning_rate);


	// Definice loss pole
	const int gradient_size = tc.batch_size * tc.output_dim;

	float* d_gradient;
	checkCudaErrors(cudaMalloc(&d_gradient, gradient_size * sizeof(float)));

	float* h_gradient = new float[gradient_size];

	checkCudaErrors(cudaMalloc(&tc.d_loss, sizeof(float)));
	checkCudaErrors(cudaMalloc(&tc.d_gradient, gradient_size * sizeof(float)));
	checkCudaErrors(cudaMalloc(&tc.d_accuracy, sizeof(float)));



	//******************************************************************************************|
	//								         MAIN TRAINING LOOP						            |					
	//******************************************************************************************|
	for (int iteration = 0; iteration < tc.n_of_iterations; iteration++) {

		std::mt19937 generator(iteration);

		for (int batch_id = 0; batch_id < tc.total_batch_count; batch_id++) {

			// Resetovani loss
			checkCudaErrors(cudaMemset(d_calculated_loss, 0.0, sizeof(float)));
			checkCudaErrors(cudaMemset(tc.d_accuracy, 0.0, sizeof(float)));

			Dataset batch = get_batch(tc.dataset, tc.batch_size, batch_id);

			int actual_batch_size = batch.target.size();
			setNumSamplesConstant(actual_batch_size);

			// Kopírování dat na GPU

			checkCudaErrors(cudaMemcpy(tc.d_input, batch.input.data(), actual_batch_size * tc.input_dim * sizeof(float), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(tc.d_target, batch.target.data(), actual_batch_size * tc.output_dim * sizeof(float), cudaMemcpyHostToDevice));


			// Forward fáze
			forward_phase(tc, enable_logging);

			// LOSS A GRADIENT FAZE
			loss_and_gradient_phase(tc, iteration, actual_batch_size, enable_logging);

			// Backward fáze
			backward_phase(tc, enable_logging);

		}
		//shuffle_batches(t)
		
		// Zamíchej vstupní data - batche
		//std::shuffle(tc.dataset.input.begin(), tc.dataset.input.end(), generator);
		//std::shuffle(tc.dataset.target.begin(), tc.dataset.target.end(), generator);
		
		reset_print_statistics(tc, iteration);

	}
}
int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	TrainingContext tc;

	// VYBER DATASET
	tc.dataset.dimensions = 2;
	tc.dataset = getDatasetByName("dataset3");
	//tc.num_samples = 4;
	tc.num_samples = tc.dataset.target.size();


	tc.n_of_iterations = 1000;
	tc.learning_rate = 0.001;
	tc.batch_size = 64;

	KernelSettings kernel_settings;
	kernel_settings.x_thread_count = 32;
	kernel_settings.y_thread_count = 32;
	kernel_settings.dimBlock = { kernel_settings.x_thread_count, kernel_settings.y_thread_count ,1 };
	kernel_settings.dimGrid = { 1, 1 ,1 };

	tc.kernel_settings = kernel_settings;

	// VYTVOR ARCHITEKTURU NEURONOVE SITE
	create_dense(tc, tc.dataset.dimensions, ActivationFunction::NONE, LayerLogicalType::INPUT);
	create_dense(tc, 50, ActivationFunction::RELU, LayerLogicalType::OTHER);
	//create_dense(tc, 50, ActivationFunction::RELU, LayerLogicalType::OTHER);
	create_dense(tc, 1, ActivationFunction::SIGMOID, LayerLogicalType::OUTPUT);

	build_network(tc);

	// Alokuj vstupní data na GPU
	allocate_input_data(tc);

	// HLAVNI TRENOVACI SMYCKA
	train(tc, false);

	cout << "That is all ..." << endl;
}
