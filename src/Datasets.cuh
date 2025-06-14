#pragma once
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <sstream>

struct Dataset {
    int dimensions;
    std::vector<float> input;
    std::vector<float> target;
};

extern Dataset dataset1;
extern Dataset dataset2;
extern Dataset dataset3;
extern Dataset dataset4;
extern Dataset dataset5;

Dataset getDatasetByName(std::string name);
Dataset load_dataset(std::string name);
Dataset get_batch(const Dataset& data, int batch_size, int batch_index);