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
	float batch_f1 = 0.0;

	for (int i = 0; i < tc.total_batch_count; i++) {
		batch_loss += tc.batch_loss[i];
		batch_accuracy += tc.batch_accuracy[i];
		batch_f1 += tc.batch_f1[i];
	}
	if (batch_accuracy > tc.dataset.target.size()) {
		cout << "AA" << endl;
	}


	//batch_loss /= tc.dataset.target.size();
	batch_accuracy /= tc.num_samples;
	batch_f1 /= tc.total_batch_count;
	// VYPSANI CELKOVE categorical crossentropy LOSS
	cout << "Iteration: " << iteration << " -- batch loss: " << batch_loss << " Batch accuracy: " << batch_accuracy << " Batch F1: " << batch_f1 <<std::endl;

	// Resetuj batch pole
	tc.batch_accuracy.clear();
	tc.batch_loss.clear();
	tc.batch_f1.clear();
}
void shuffle_indexes(std::vector<int> &indexes, int iteration) {
	std::mt19937 generator(iteration);
	std::shuffle(indexes.begin(), indexes.end(), generator);
}
void handle_f1_calculation(TrainingContext& tc) {
	float d_tp, d_fp, d_fn;
	cudaMemcpy(&d_tp, tc.d_tp, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&d_fp, tc.d_fp, sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(&d_fn, tc.d_fn, sizeof(float), cudaMemcpyDeviceToHost);

	// Vypočítání F1 score
	float precision = d_tp / (d_tp + d_fp + 1e-7f);
	float recall = d_tp / (d_tp + d_fn + 1e-7f);

	float f1 = (2.0f * precision * recall) / (precision + recall + 1e-7f);

	tc.batch_f1.push_back(f1);
}

void train(TrainingContext& tc, bool enable_logging, bool enable_results) {
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
	checkCudaErrors(cudaMalloc(&tc.d_tp, sizeof(float)));
	checkCudaErrors(cudaMalloc(&tc.d_fp, sizeof(float)));
	checkCudaErrors(cudaMalloc(&tc.d_fn, sizeof(float)));

	// Pomocné pole pro shuffling jednotlivlivých dávek
	std::vector<int> indexes;
	for (int i = 0; i < tc.dataset.target.size(); i++) {
		indexes.push_back(i);
	}
	//shuffle_indexes(indexes, 0);

	//******************************************************************************************|
	//								         MAIN TRAINING LOOP						            |					
	//******************************************************************************************|
	for (int iteration = 0; iteration < tc.n_of_iterations; iteration++) {

		// Zamíchej indexy pro batche
		shuffle_indexes(indexes, iteration);

		for (int batch_id = 0; batch_id < tc.total_batch_count; batch_id++) {

			// Resetovani loss
			checkCudaErrors(cudaMemset(d_calculated_loss, 0.0, sizeof(float)));
			checkCudaErrors(cudaMemset(tc.d_accuracy, 0.0, sizeof(float)));
			checkCudaErrors(cudaMemset(tc.d_tp, 0, sizeof(float)));
			checkCudaErrors(cudaMemset(tc.d_fp, 0, sizeof(float)));
			checkCudaErrors(cudaMemset(tc.d_fn, 0, sizeof(float)));

			Dataset batch = get_batch(tc.dataset, tc.batch_size, batch_id, indexes);

			int actual_batch_size = batch.target.size();
			setNumSamplesConstant(actual_batch_size);

			// Kopírování dat na GPU

			checkCudaErrors(cudaMemcpy(tc.d_input, batch.input.data(), actual_batch_size * tc.input_dim * sizeof(float), cudaMemcpyHostToDevice));
			checkCudaErrors(cudaMemcpy(tc.d_target, batch.target.data(), actual_batch_size * tc.output_dim * sizeof(float), cudaMemcpyHostToDevice));


			// Forward fáze
			forward_phase(tc, enable_logging, enable_results, actual_batch_size, batch.target);

			// LOSS A GRADIENT FAZE
			loss_and_gradient_phase(tc, iteration, actual_batch_size, enable_logging);

			// Backward fáze
			backward_phase(tc, enable_logging);


			handle_f1_calculation(tc);

		}
		reset_print_statistics(tc, iteration);

	}
}

void run_dataset1(TrainingContext& tc) {
	tc.dataset = getDatasetByName("dataset1");
	tc.n_of_iterations = 1000;
	tc.learning_rate = 0.1;
	tc.batch_size = 2;

	create_dense(tc, tc.dataset.dimensions, ActivationFunction::NONE, LayerLogicalType::INPUT);
	create_dense(tc, 50, ActivationFunction::RELU, LayerLogicalType::OTHER);
	create_dense(tc, 1, ActivationFunction::SIGMOID, LayerLogicalType::OUTPUT);
}
void run_dataset3(TrainingContext &tc) {
	tc.dataset = getDatasetByName("dataset3");
	tc.n_of_iterations = 500;
	tc.learning_rate = 0.01;
	tc.batch_size = 30;

	create_dense(tc, tc.dataset.dimensions, ActivationFunction::NONE, LayerLogicalType::INPUT);
	create_dense(tc, 50, ActivationFunction::RELU, LayerLogicalType::OTHER);
	create_dense(tc, 50, ActivationFunction::RELU, LayerLogicalType::OTHER);
	create_dense(tc, 1, ActivationFunction::SIGMOID, LayerLogicalType::OUTPUT);
}
void run_dataset4(TrainingContext& tc) {
	tc.dataset = getDatasetByName("dataset3");
	tc.n_of_iterations = 1500;
	tc.learning_rate = 0.001;
	tc.batch_size = 128;

	create_dense(tc, tc.dataset.dimensions, ActivationFunction::NONE, LayerLogicalType::INPUT);
	create_dense(tc, 50, ActivationFunction::RELU, LayerLogicalType::OTHER);
	create_dense(tc, 50, ActivationFunction::RELU, LayerLogicalType::OTHER);
	create_dense(tc, 1, ActivationFunction::SIGMOID, LayerLogicalType::OUTPUT);
}
int main(int argc, char* argv[])
{
	initializeCUDA(deviceProp);

	TrainingContext tc;
	
	//tc.dataset = getDatasetByName("dataset3");
	tc.num_samples = tc.dataset.target.size();


	KernelSettings kernel_settings;
	kernel_settings.x_thread_count = 32;
	kernel_settings.y_thread_count = 32;
	kernel_settings.dimBlock = { kernel_settings.x_thread_count, kernel_settings.y_thread_count ,1 };
	kernel_settings.dimGrid = { 1, 1 ,1 };

	tc.kernel_settings = kernel_settings;

	// VYTVOR ARCHITEKTURU NEURONOVE SITE
	run_dataset4(tc);
	tc.num_samples = tc.dataset.target.size();

	build_network(tc);

	// Alokuj vstupní data na GPU
	allocate_input_data(tc);

	// HLAVNI TRENOVACI SMYCKA
	train(tc, false, false);

	cout << "That is all ..." << endl;
}
