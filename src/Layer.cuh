#pragma once
#include <cudaDefs.h>
#include <iostream>
#include <random>


enum class LayerType {
	DENSE,
	DROPOUT
};

enum class ActivationFunction {
	RELU,
	SIGMOID,
	NONE
};

struct Layer {
	LayerType type;
	ActivationFunction activation;
	int in;
	int out;
	float* weights;
	float* biases;
	float* activations;
	float* gradients;
	float dropout_rate; // pravd�podobnost dropout
	bool* mask; // maska pro dropout (1 = neuron aktivn�, 0 = vypnut�)
};

Layer createDenseLayer(int in_size, int out_size, ActivationFunction act);

Layer createDropoutLayer(int size, float rate);

void initLayer(Layer& layer, int input_size);