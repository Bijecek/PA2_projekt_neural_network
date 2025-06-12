#include "NeuralNetwork.cuh"

using std::cout;
using std::endl;

std::vector<std::pair<int, ActivationFunction>> layer_specifications;


void setNumSamplesConstant(int input_size) {
	checkCudaErrors(cudaMemcpyToSymbol((const void*)&num_samples, &input_size, sizeof(int)));
}
void setLearningRateConstant(int learning_r) {
	checkCudaErrors(cudaMemcpyToSymbol((const void*)&learning_rate, &learning_r, sizeof(float)));
}

// Jedno vlákno = jeden sample - jeden neuron
__global__ void forward(const float* __restrict__ input_data, int input_size, const float* __restrict__ weight_matrix, const float* __restrict__ bias, float* __restrict__ output_data, int output_size, int activation_type) {
	int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
	int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

	if (sample_id < num_samples && neuron_id < output_size) {
		float sum = 0.0;

		for (int i = 0; i < input_size; i++) {
			sum += input_data[sample_id * input_size + i] * weight_matrix[neuron_id * input_size + i];
		}

		// Pøidej bias (input 0,0 -> target 0 by jinak nefungovalo)
		sum += bias[neuron_id];

		output_data[sample_id * output_size + neuron_id] = activation_functions[activation_type](sum);

	}
}

//256 velikost -> pracuju s 128 THREADS_PER_BLOCK
__global__ void compute_loss(float* y_predicted, float* y_true, float* loss, int size) {
	__shared__ float s_loss[THREADS_PER_BLOCK];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int next = size / 2;

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
__global__ void backward(float* activations, int input_size, float* weight_matrix, bool first, float* gradient_in
	, float* gradient_out, int output_size, int next_out_size, int activation_type) {

	int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
	int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

	if (sample_id < num_samples && neuron_id < output_size) {

		if (first) {
			gradient_out[sample_id * output_size + neuron_id] = gradient_in[sample_id * output_size + neuron_id] * derivate_activation_functions[activation_type](activations[sample_id * output_size + neuron_id]);
		}
		else {

			float sum = 0.0;
			for (int j = 0; j < next_out_size; j++) {
				sum += weight_matrix[j * output_size + neuron_id] * gradient_in[sample_id * next_out_size + j];
			}

			gradient_out[sample_id * output_size + neuron_id] = sum * derivate_activation_functions[activation_type](activations[sample_id * output_size + neuron_id]);

		}
	}
}
// Funkce pro aktualizaci vah a biasu
// TODO:
//      Pøepoèítat si zmìnu váhy pro N vzorkù a až potom volat tuhle funkci
__global__ void update_parameters(float* input, float* gradient, float* weight_matrix, float* biases, int input_size, int output_size) {
	int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
	int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

	if (sample_id < num_samples && neuron_id < output_size) {
		// Learning rate
		float learning_rate = 0.01f;


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

// Forward fáze
void forward_phase(TrainingContext& tc, bool enable_logging) {
	float* current_input = tc.d_input;
	for (int i = 0; i < tc.layers.size(); i++) {
		Layer& current_layer = tc.layers[i];

		// Nastavení rozmìrù gridu - dynamicky ho upravujeme podle rozmìrù <input_size; layers[i].out>

		// (4 + 16 - 1) / 16
		unsigned int x_grid_dim = (tc.input_size + tc.kernel_settings.x_thread_count - 1) / tc.kernel_settings.x_thread_count;
		// (20 + 16 - 1) / 16
		unsigned int y_grid_dim = (tc.layers[i].out + tc.kernel_settings.y_thread_count - 1) / tc.kernel_settings.y_thread_count;

		tc.kernel_settings.dimGrid.x = x_grid_dim;
		tc.kernel_settings.dimGrid.y = y_grid_dim;
		tc.kernel_settings.dimGrid.z = 1;

		if (enable_logging) {
			cout << "Forward kernel executed with: " << x_grid_dim << " " << y_grid_dim << endl;
		}

		// LOGOVANI
		if (enable_logging) {
			checkDeviceMatrix<float>(current_layer.activations, tc.input_size * current_layer.out * sizeof(float), 1,
				tc.input_size * current_layer.out, "%f ", "Before: ");
		}

		forward << <tc.kernel_settings.dimGrid, tc.kernel_settings.dimBlock >> > (current_input, current_layer.in, current_layer.weights,
			current_layer.biases, current_layer.activations, current_layer.out, static_cast<int>(current_layer.activation));


		// Zmìna vstupu
		current_input = current_layer.activations;

		// LOGOVANI
		if (enable_logging) {
			checkDeviceMatrix<float>(current_layer.activations, tc.input_size * current_layer.out * sizeof(float), 1,
				tc.input_size * current_layer.out, "%f ", "After: ");
		}
	}

	// LOGOVANI - vypis výstupu poslední vrstvy pro všechny vstupy
	
	checkDeviceMatrix<float>(tc.layers[tc.layers.size() - 1].activations, tc.input_size * tc.layers[tc.layers.size() - 1].out *
			sizeof(float), 1, tc.input_size * tc.layers[tc.layers.size() - 1].out, "%f ", "Activations: ");
	

	if (enable_logging) {
		std::cout << "Forward ok" << std::endl;
	}

}
void loss_and_gradient_phase(TrainingContext& tc, int iteration, bool enable_logging) {

	checkCudaErrors(cudaMemset(tc.d_loss, 0, sizeof(float)));


	// TODO Nastavit dle batch_size -- momentálnì je to na 128
	// S timto pocitame LOSS a GRADIENTy
	tc.kernel_settings.dimBlock.x = 128;
	tc.kernel_settings.dimBlock.y = 1;
	tc.kernel_settings.dimBlock.z = 1;

	tc.kernel_settings.dimGrid.x = 1;
	tc.kernel_settings.dimGrid.y = 1;
	tc.kernel_settings.dimGrid.z = 1;


	// Poèítání loss -- jako vstup je output z pøedposlední do poslední vrstvy (proto size() - 1)
	compute_loss << <tc.kernel_settings.dimGrid, tc.kernel_settings.dimBlock >> > (tc.layers[tc.layers.size() - 1].activations, tc.d_target, tc.d_loss, tc.output_size * tc.output_dim);

	// Pøesun celkové loss zpátky na host 
	if (enable_logging) {
		checkDeviceMatrix<float>(tc.d_loss, sizeof(float), 1, 1, "%f ", "Loss: ");
	}
	float* tmp_loss = new float[1];
	checkCudaErrors(cudaMemcpy(tmp_loss, tc.d_loss, sizeof(float), cudaMemcpyDeviceToHost));

	// VYPSANI CELKOVE categorical crossentropy LOSS
	cout << "Iteration: " << iteration << " -- loss: " << tmp_loss[0] << std::endl;

	if (enable_logging) {
		std::cout << "Loss ok" << std::endl;
	}

	compute_gradient << <tc.kernel_settings.dimGrid, tc.kernel_settings.dimBlock >> > (tc.layers[tc.layers.size() - 1].activations, tc.d_target, tc.d_gradient, tc.output_size * tc.output_dim);
	
	// LOGOVANI
	if (enable_logging) {
		checkDeviceMatrix<float>(tc.d_gradient, tc.output_size * tc.output_dim * sizeof(float), 1, tc.output_size * tc.output_dim, "%f ", "Gradient: ");
	}
}

void backward_phase(TrainingContext& tc, bool enable_logging) {
	// Resetovani velikosti kernelu
	tc.kernel_settings.dimBlock.x = tc.kernel_settings.x_thread_count;
	tc.kernel_settings.dimBlock.y = tc.kernel_settings.y_thread_count;
	tc.kernel_settings.dimBlock.z = 1;

	for (int i = tc.layers.size() - 1; i >= 0; i--) {
		float* activation = tc.layers[i].activations;
		int in_size = tc.layers[i].in;
		int out_size = tc.layers[i].out;
		float* weight_matrix = (i == tc.layers.size() - 1) ? nullptr : tc.layers[i + 1].weights;
		bool first = (i == tc.layers.size() - 1) ? true : false;
		float* gradient_in = (i == tc.layers.size() - 1) ? tc.d_gradient : tc.layers[i + 1].gradients;
		float* gradient_out = tc.layers[i].gradients;


		// (4 + 16 - 1) / 16
		// (20 + 16 - 1) / 16
		unsigned int x_grid_dim = (tc.input_size + tc.kernel_settings.x_thread_count - 1) / tc.kernel_settings.x_thread_count;
		unsigned int y_grid_dim = (tc.layers[i].out + tc.kernel_settings.y_thread_count - 1) / tc.kernel_settings.y_thread_count;

		tc.kernel_settings.dimGrid.x = x_grid_dim;
		tc.kernel_settings.dimGrid.y = y_grid_dim;
		tc.kernel_settings.dimGrid.z = 1;


		if (enable_logging) {
			cout << "Backward kernel executed with: " << x_grid_dim << " " << y_grid_dim << endl;
		}


		if (i == tc.layers.size() - 1) {
			backward << <tc.kernel_settings.dimGrid, tc.kernel_settings.dimBlock >> > (activation, in_size, weight_matrix, first, gradient_in, gradient_out,
				out_size, 0, static_cast<int>(tc.layers[i].activation));
		}
		else {
			backward << <tc.kernel_settings.dimGrid, tc.kernel_settings.dimBlock >> > (activation, in_size, weight_matrix, first, gradient_in, gradient_out,
				out_size, tc.layers[i + 1].out, static_cast<int>(tc.layers[i].activation));
		}




		// LOGOVANI
		//checkDeviceMatrix<float>(layers[i].gradients, input_size * layers[i].out * sizeof(float), 1, input_size * layers[i].out, "%f ", "Gradient calc: ");


		// AKTUALIZACE VAH
		float* input_activations = (i == 0) ? tc.d_input : tc.layers[i - 1].activations;

		update_parameters << <tc.kernel_settings.dimGrid, tc.kernel_settings.dimBlock >> > (input_activations, tc.layers[i].gradients, tc.layers[i].weights
			, tc.layers[i].biases, tc.layers[i].in, tc.layers[i].out);


	}

	if (enable_logging) {
		std::cout << "Backward ok" << std::endl;
	}
}