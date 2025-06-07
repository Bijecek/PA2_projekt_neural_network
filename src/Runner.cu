#include <cudaDefs.h>
#include <random>
#include <algorithm>
#include <iostream>

#include "NeuralNetwork.cuh"
#include "Layer.cuh"
#include "ActivationFunctions.cuh"
#include "Datasets.cuh"

using std::cout;
using std::endl;

//constexpr unsigned int THREADS_PER_BLOCK = 128;

cudaError_t error = cudaSuccess;
cudaDeviceProp deviceProp = cudaDeviceProp();


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



//******************************************************************************************|
//								    INPUT DATASETS										    |						
//******************************************************************************************|

	Dataset ds = getDatasetByName("dataset1");

	const std::vector<float> X = ds.input;
	const std::vector<float> y = ds.target;


//******************************************************************************************|
//							   INPUT DATA ALLOCATION ON GPU									|						
//******************************************************************************************|

	float* d_input;
	checkCudaErrors(cudaMalloc(&d_input, input_size * input_dimension * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_input, X.data(), input_size * input_dimension * sizeof(float), cudaMemcpyHostToDevice));

	float* d_target;
	checkCudaErrors(cudaMalloc(&d_target, output_size * output_dimension * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_target, y.data(), output_size * output_dimension * sizeof(float), cudaMemcpyHostToDevice));


//******************************************************************************************|
//								      CREATE LAYERS											|						
//******************************************************************************************|

	std::vector<Layer> layers;

	// Vstupní -> Hidden1
	layers.push_back(createDenseLayer(input_dimension, hidden_size, ActivationFunction::RELU));

	// Hidden1 -> HiddenN
	for (int i = 0; i < num_hidden; i++) {
		layers.push_back(createDenseLayer(hidden_size, hidden_size, ActivationFunction::RELU));
	}

	// HiddenN -> Výstupní
	layers.push_back(createDenseLayer(hidden_size, output_dimension, ActivationFunction::SIGMOID));

//******************************************************************************************|
//								  LAYERS ALLOCATION ON GPU								    |					
//******************************************************************************************|
	
	for (auto &layer : layers) {
		initLayer(layer, input_size);
	}

	float* d_calculated_loss;
	checkCudaErrors(cudaMalloc(&d_calculated_loss, sizeof(float)));


	setNumSamplesConstant(input_size);


	// Definice loss pole
	const int gradient_size = output_size * output_dimension;

	float* d_gradient;
	checkCudaErrors(cudaMalloc(&d_gradient, gradient_size * sizeof(float)));

	float* h_gradient = new float[gradient_size];

	const int x_thread_count = 16;
	const int y_thread_count = 16;
	

	dim3 dimBlock{ x_thread_count, y_thread_count ,1 };
	dim3 dimGrid{ 1,1 ,1 };


//******************************************************************************************|
//								         MAIN TRAINING LOOP						            |					
//******************************************************************************************|
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
