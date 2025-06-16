#include "Layer.cuh"

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

void initLayer(Layer& layer, int input_size) {

    if (layer.type == LayerType::DENSE) {
        // Alokace paměti na GPU
        checkCudaErrors(cudaMalloc(&layer.weights, layer.in * layer.out * sizeof(float)));
        checkCudaErrors(cudaMalloc(&layer.biases, layer.out * sizeof(float)));
        checkCudaErrors(cudaMalloc(&layer.activations, input_size * layer.out * sizeof(float)));
        checkCudaErrors(cudaMalloc(&layer.gradients, input_size * layer.out * sizeof(float)));

        // Náhodné naplnění vah
        std::mt19937 gen(42);
        //std::uniform_real_distribution<float> dist(-1.0, 1.0f);
        std::uniform_real_distribution<float> dist(-0.2, 0.2f);

        std::vector<float> temporary_weights(layer.in * layer.out);
        std::vector<float> temporary_biases(layer.out);

        for (auto& v : temporary_weights) v = dist(gen);
        for (auto& v : temporary_biases) v = dist(gen);

        // Kopírování na GPU
        checkCudaErrors(cudaMemcpy(layer.weights, temporary_weights.data(), layer.in * layer.out * sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(layer.biases, temporary_biases.data(), layer.out * sizeof(float), cudaMemcpyHostToDevice));

        if (layer.dropout_rate > 0.0f) {
            checkCudaErrors(cudaMalloc(&layer.mask, input_size * layer.out * sizeof(bool)));
        }
    }
}
std::string getActivationFunction(ActivationFunction af) {
    switch (af) {
    case ActivationFunction::RELU:
        return "ReLU";
    case ActivationFunction::SIGMOID:
        return "Sigmoid";
    case ActivationFunction::NONE:
        return "None";
    default:
        return "Unknown";
    }
}
std::string getLayerType(LayerType type) {
    switch (type) {
    case LayerType::DENSE:
        return "DENSE";
    case LayerType::DROPOUT:
        return "DROPOUT";
    default:
        return "Unknown";
    }
}