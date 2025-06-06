#include "Loss_functions.cuh"

//256 velikost -> pracuju s 128 THREADS_PER_BLOCK
__global__ void compute_loss(float* y_predicted, float* y_true, float* loss, int size) {
	__shared__ float s_loss[THREADS_PER_BLOCK * 2];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int next = size / 2;

	float epsilon = 1e-7;

	// TODO SIZE/2
	if (idx < size) {
		s_loss[idx] = -y_true[idx] * logf(fmax(y_predicted[idx], epsilon));
		__syncthreads();

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