#pragma once
#include <string>

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
    float dropout_rate; // pravdìpodobnost dropout
    bool* mask;         // maska pro dropout (1 = neuron aktivní, 0 = vypnutý)
};

Layer createDenseLayer(int in_size, int out_size, ActivationFunction act) {
    Layer layer;

    layer.type = LayerType::DENSE;
    layer.activation = act;
    layer.in = in_size;
    layer.out = out_size;
    layer.dropout_rate = 0.0f;
    layer.activations = nullptr;
    layer.mask = nullptr;
    layer.weights = nullptr;
    layer.biases = nullptr;

    return layer;
}

Layer createDropoutLayer(int size, float rate) {
    Layer layer;
    layer.type = LayerType::DROPOUT;
    layer.activation = ActivationFunction::NONE;
    layer.in = size;
    layer.out = size;
    
    layer.weights = nullptr;
    layer.biases = nullptr;

    layer.activations = new float[size];
    layer.gradients = new float[size];
    layer.dropout_rate = rate;
    layer.mask = new bool[size];
    return layer;
}

void initLayer(Layer& layer, int input_size) {

    if (layer.type == LayerType::DENSE) {
        // Alokace pamìti na GPU
        checkCudaErrors(cudaMalloc(&layer.weights, layer.in * layer.out * sizeof(float)));
        checkCudaErrors(cudaMalloc(&layer.biases, layer.out * sizeof(float)));
        checkCudaErrors(cudaMalloc(&layer.activations, input_size * layer.out * sizeof(float)));
        checkCudaErrors(cudaMalloc(&layer.gradients, input_size * layer.out * sizeof(float)));

        // Náhodné naplnìní vah (normální rozdìlení napøíklad)
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);

        std::vector<float> temporary_weights(layer.in * layer.out);
        std::vector<float> temporary_biases(layer.out);

        for (auto& v : temporary_weights) v = dist(gen);
        for (auto& v : temporary_biases) v = 0.1f;

        // Kopírování na GPU
        checkCudaErrors(cudaMemcpy(layer.weights, temporary_weights.data(), layer.in * layer.out * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(layer.biases, temporary_biases.data(), layer.out * sizeof(float), cudaMemcpyHostToDevice));
    }
    else {
        checkCudaErrors(cudaMalloc(&layer.activations, input_size * layer.out * sizeof(float)));
        checkCudaErrors(cudaMalloc(&layer.gradients, input_size * layer.out * sizeof(float)));
        checkCudaErrors(cudaMalloc(&layer.mask, input_size * layer.out * sizeof(bool)));
    }

    // LOGOVANI
    //checkDeviceMatrix<float>(layer.weights, layer.in * layer.out * sizeof(float), 1, layer.in * layer.out, "%f ", "Weights: ");
}