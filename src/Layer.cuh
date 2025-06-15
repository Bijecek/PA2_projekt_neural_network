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

enum class LayerLogicalType {
	INPUT,
	OUTPUT,
	OTHER
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

	// Dropout parametry
	float dropout_rate = 0.0f;
	bool* mask = nullptr;
};

Layer createDenseLayer(int in_size, int out_size, ActivationFunction act);

Layer createDropoutLayer(int size, float rate);

void initLayer(Layer& layer, int input_size);

std::string getActivationFunction(ActivationFunction af);