#pragma once
#include <vector>
#include <string>
#include <stdexcept>

struct Dataset {
    std::vector<float> input;
    std::vector<float> target;
};

extern Dataset dataset1;
extern Dataset dataset2;
extern Dataset dataset3;
extern Dataset dataset4;

Dataset getDatasetByName(std::string name);