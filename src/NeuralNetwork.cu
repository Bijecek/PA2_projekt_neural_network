#include "NeuralNetwork.cuh"

using std::cout;
using std::endl;

__constant__ int num_samples;
__constant__ float learning_rate;

std::vector<OneLayer> layer_specifications;


void setNumSamplesConstant(int input_size) {
	checkCudaErrors(cudaMemcpyToSymbol((const void*)&num_samples, &input_size, sizeof(int)));
}
void setLearningRateConstant(float learning_r) {
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
__global__ void dropout_forward(float* input, float* output, bool* mask, float dropout_rate, int size, curandState* curand_state) {
	int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
	int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (sample_id < num_samples && neuron_id < size) {
		curand_init(1, sample_id * size + neuron_id, 0, &curand_state[sample_id * size + neuron_id]);

		float random_val = curand_uniform(&curand_state[sample_id * size + neuron_id]);

		mask[sample_id * size + neuron_id] = random_val > dropout_rate;

		// Inverze dropout rate -- nutné pro inferenci
		//output[idx] = mask[idx] ? input[idx] / (1.0f - dropout_rate) : 0.0f;

		output[sample_id * size + neuron_id] = mask[sample_id * size + neuron_id] ? input[sample_id * size + neuron_id] : 0.0f;
	}
}

//256 velikost -> pracuju s 128 THREADS_PER_BLOCK
__global__ void compute_loss(const float* __restrict__ y_predicted, const float* __restrict__ y_true, float* loss, const int size) {
	__shared__ float s_loss[THREADS_PER_BLOCK];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int next = size / 2;

	float epsilon = 1e-7;

	if (idx < size) {
		s_loss[idx] = -y_true[idx] * logf(fmax(y_predicted[idx], epsilon)) - (1.0f - y_true[idx]) * logf(fmax(1.0f - y_predicted[idx], epsilon));;
		__syncthreads();

		if (idx + next >= size) {
			return;
		}

		while (next > 0) {
			s_loss[idx] += s_loss[idx + next];

			__syncthreads();

			next >>= 1;

			if (idx >= next) {
				break;
			}
		}
		if (idx == 0) {
			*loss = s_loss[0];
		}
	}
}
// Výpoèet gradientu - derivace BCE loss (y_pred je po sigmoidu)
__global__ void compute_gradient(float* y_pred, float* y_true, float* gradient, int size, float* accuracy, float* d_tp, float* d_fp, float* d_fn) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {
		gradient[idx] = y_pred[idx] - y_true[idx];

		if ((y_pred[idx] >= 0.5f && y_true[idx] == 1.0f) || (y_pred[idx] < 0.5f && y_true[idx] == 0.0f)) {
			atomicAdd(accuracy, 1.0f);
		}

		// Calculate confusion matrix
		if (y_true[idx] == 1.0f) {
			if (y_pred[idx] >= 0.5f) {
				atomicAdd(d_tp, 1.0f);
			}
			else {
				atomicAdd(d_fn, 1.0f);
			}
		}
		else {
			if (y_pred[idx] >= 0.5f) {
				atomicAdd(d_fp, 1.0f);
			}
		}

	}
}
__global__ void backward(float* activations, int input_size, float* weight_matrix, bool first, float* gradient_in
	, float* gradient_out, int output_size, int next_out_size, int activation_type, bool* dropout_mask, float dropout_rate) {

	int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
	int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

	if (sample_id < num_samples && neuron_id < output_size) {
		float final_gradient;
		if (first) {
			float g = gradient_in[sample_id * output_size + neuron_id];
			final_gradient = g * derivate_activation_functions[activation_type](activations[sample_id * output_size + neuron_id]);
		}
		else {

			float sum = 0.0;
			for (int j = 0; j < next_out_size; j++) {
				sum += weight_matrix[j * output_size + neuron_id] * gradient_in[sample_id * next_out_size + j];
			}

			final_gradient = sum * derivate_activation_functions[activation_type](activations[sample_id * output_size + neuron_id]);

		}

		if (dropout_rate > 0.0f) {
			final_gradient = dropout_mask[sample_id * output_size + neuron_id] ? final_gradient : 0.0f;
		}

		gradient_out[sample_id * output_size + neuron_id] = final_gradient;
	}
}
__global__ void dropout_backward(float* gradient_in, float* gradient_out, int output_size, bool* mask) {
	int sample_id = blockIdx.x * blockDim.x + threadIdx.x;
	int neuron_id = blockIdx.y * blockDim.y + threadIdx.y;

	if (sample_id < num_samples && neuron_id < output_size) {
		gradient_out[sample_id * output_size + neuron_id] = mask[sample_id * output_size + neuron_id] ? gradient_in[sample_id * output_size + neuron_id] : 0.0f;
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
		//float learning_rate = 0.01f;


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
void forward_phase(TrainingContext& tc, bool enable_logging, bool enable_results, int actual_batch_size, std::vector<float> target_data) {
	float* current_input = tc.d_input;
	for (int i = 0; i < tc.layers.size(); i++) {
		Layer& current_layer = tc.layers[i];

		// Nastavení rozmìrù gridu - dynamicky ho upravujeme podle rozmìrù <input_size; layers[i].out>

		// (4 + 16 - 1) / 16
		unsigned int x_grid_dim = (tc.batch_size + tc.kernel_settings.x_thread_count - 1) / tc.kernel_settings.x_thread_count;
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
			checkDeviceMatrix<float>(current_layer.activations, tc.batch_size * current_layer.out * sizeof(float), 1,
				tc.batch_size * current_layer.out, "%f ", "Before: ");
		}

		if (current_layer.type == LayerType::DENSE) {
			forward << <tc.kernel_settings.dimGrid, tc.kernel_settings.dimBlock >> > (current_input, current_layer.in, current_layer.weights,
				current_layer.biases, current_layer.activations, current_layer.out, static_cast<int>(current_layer.activation));
		}
		if (current_layer.dropout_rate > 0.0f) {
			cudaMalloc((void**)&tc.d_curand_state, tc.num_samples * current_layer.out * sizeof(curandState));

			// Proveï inplace dropout
			dropout_forward << <tc.kernel_settings.dimGrid, tc.kernel_settings.dimBlock >> > (current_layer.activations, current_layer.activations, 
				current_layer.mask, current_layer.dropout_rate, current_layer.out, tc.d_curand_state);

			cudaFree(tc.d_curand_state);
		}

		// Zmìna vstupu
		current_input = current_layer.activations;

		// LOGOVANI
		if (enable_logging) {
			checkDeviceMatrix<float>(current_layer.activations, tc.batch_size * current_layer.out * sizeof(float), 1,
				tc.batch_size * current_layer.out, "%f ", "After: ");
		}
	}

	// LOGOVANI - vypis výstupu poslední vrstvy pro všechny vstupy
	if (enable_results || enable_logging) {
		cout << "Expected values: " << endl;
		for (int i = 0; i < target_data.size(); i++) {
			cout << target_data[i] << " ";
		}
		cout << endl;
		checkDeviceMatrix<float>(tc.layers[tc.layers.size() - 1].activations, actual_batch_size * tc.layers[tc.layers.size() - 1].out *
			sizeof(float), 1, actual_batch_size * tc.layers[tc.layers.size() - 1].out, "%f ", "Activations: ");
	}

	if (enable_logging) {
		std::cout << "Forward ok" << std::endl;
	}

}
void loss_and_gradient_phase(TrainingContext& tc, int iteration, int actual_batch_size, bool enable_logging) {

	checkCudaErrors(cudaMemset(tc.d_loss, 0.0, sizeof(float)));
	checkCudaErrors(cudaMemset(tc.d_accuracy, 0.0, sizeof(float)));


	// TODO Nastavit dle batch_size -- momentálnì je to na 128
	// S timto pocitame LOSS a GRADIENTy
	tc.kernel_settings.dimBlock.x = tc.batch_size;
	tc.kernel_settings.dimBlock.y = 1;
	tc.kernel_settings.dimBlock.z = 1;

	tc.kernel_settings.dimGrid.x = 1;
	tc.kernel_settings.dimGrid.y = 1;
	tc.kernel_settings.dimGrid.z = 1;


	// Poèítání loss -- jako vstup je output z pøedposlední do poslední vrstvy (proto size() - 1)
	compute_loss << <tc.kernel_settings.dimGrid, tc.kernel_settings.dimBlock >> > (tc.layers[tc.layers.size() - 1].activations, tc.d_target, tc.d_loss, actual_batch_size * tc.output_dim);

	// Pøesun celkové loss zpátky na host 
	if (enable_logging) {
		checkDeviceMatrix<float>(tc.d_loss, sizeof(float), 1, 1, "%f ", "Loss: ");
	}

	float tmp_loss;
	checkCudaErrors(cudaMemcpy(&tmp_loss, tc.d_loss, sizeof(float), cudaMemcpyDeviceToHost));

	// Batch loss pole
	tc.batch_loss.push_back(tmp_loss);
	


	compute_gradient << <tc.kernel_settings.dimGrid, tc.kernel_settings.dimBlock >> > (tc.layers[tc.layers.size() - 1].activations, 
		tc.d_target, tc.d_gradient, actual_batch_size * tc.output_dim, tc.d_accuracy, tc.d_tp, tc.d_fp, tc.d_fn);
	
	float tmp_accuracy;
	checkCudaErrors(cudaMemcpy(&tmp_accuracy, tc.d_accuracy, sizeof(float), cudaMemcpyDeviceToHost));

	// Batch accuracy pole
	tc.batch_accuracy.push_back(tmp_accuracy);

	//cout << "Current batch Loss: " << tmp_loss << endl;
	//cout << "Current batch Accuracy: " << tmp_accuracy << endl;

	// LOGOVANI
	if (enable_logging) {
		cout << "Current batch Loss: " << tmp_loss << endl;
		cout << "Current batch Accuracy: " << tmp_accuracy << endl;

		checkDeviceMatrix<float>(tc.d_gradient, tc.batch_size * tc.output_dim * sizeof(float), 1, tc.batch_size * tc.output_dim, "%f ", "Gradient: ");
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
		unsigned int x_grid_dim = (tc.batch_size + tc.kernel_settings.x_thread_count - 1) / tc.kernel_settings.x_thread_count;
		unsigned int y_grid_dim = (tc.layers[i].out + tc.kernel_settings.y_thread_count - 1) / tc.kernel_settings.y_thread_count;

		tc.kernel_settings.dimGrid.x = x_grid_dim;
		tc.kernel_settings.dimGrid.y = y_grid_dim;
		tc.kernel_settings.dimGrid.z = 1;


		if (enable_logging) {
			cout << "Backward kernel executed with: " << x_grid_dim << " " << y_grid_dim << endl;
		}


		if (tc.layers[i].type == LayerType::DENSE) {
			if (i == tc.layers.size() - 1) {
				backward << <tc.kernel_settings.dimGrid, tc.kernel_settings.dimBlock >> > (activation, in_size, weight_matrix, first, gradient_in, gradient_out,
					out_size, 0, static_cast<int>(tc.layers[i].activation), tc.layers[i].mask, tc.layers[i].dropout_rate);
			}
			else {
				backward << <tc.kernel_settings.dimGrid, tc.kernel_settings.dimBlock >> > (activation, in_size, weight_matrix, first, gradient_in, gradient_out,
					out_size, tc.layers[i + 1].out, static_cast<int>(tc.layers[i].activation), tc.layers[i].mask, tc.layers[i].dropout_rate);
			}
		}
	

		if (enable_logging) {
			checkDeviceMatrix<float>(tc.layers[i].gradients, tc.batch_size * tc.layers[i].out * sizeof(float), 1, tc.batch_size * tc.layers[i].out, "%f ", "Gradient calc: ");
		}


		// AKTUALIZACE VAH
		float* input_activations = (i == 0) ? tc.d_input : tc.layers[i - 1].activations;

		
		update_parameters << <tc.kernel_settings.dimGrid, tc.kernel_settings.dimBlock >> > (input_activations, tc.layers[i].gradients, tc.layers[i].weights
				, tc.layers[i].biases, tc.layers[i].in, tc.layers[i].out);
		

		
	}

	if (enable_logging) {
		std::cout << "Backward ok" << std::endl;
	}
}