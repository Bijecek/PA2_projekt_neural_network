#include "Datasets.cuh"


Dataset dataset1 = {
    {0, 0, 0, 1, 1, 0, 1, 1},
    {0, 1, 1, 0}
};

Dataset dataset2 = {
    {1, 1, 0, 0, 1, 0, 0, 1},
    {1, 1, 0, 0}
};

Dataset dataset3 = {
    {0.5f, 0.3f, 0.8f, 0.1f, 0.9f, 0.4f, 0.2f, 0.7f},
    {0.6f, 0.3f, 0.7f, 0.2f}
};

Dataset dataset4 = {
    {0.0f, 0.8f, 0.0f, 0.6f, 0.3f, 0.0f, 1.0f, 0.2f},
    {0.0f, 1.0f, 0.0f, 0.9f}
};

Dataset getDatasetByName(std::string name) {
    if (name == "dataset1") return dataset1;
    else if (name == "dataset2") return dataset2;
    else if (name == "dataset3") return dataset3;
    else if (name == "dataset4") return dataset4;
    else throw std::invalid_argument("Dataset not found");
}