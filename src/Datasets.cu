#include "Datasets.cuh"


Dataset dataset1 = {
    {2},
    {0, 0, 0, 1, 1, 0, 1, 1},
    {0, 1, 1, 0}
};



Dataset load_dataset(std::string name) {
    Dataset d;

    std::ifstream file(name);
    if (!file) {
        std::cout << "Unable to open file" << std::endl;
        return d;
    }

    std::string line;
    d.dimensions = -1;

    // Ètení souboru po øádcích
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;


        std::vector<float> input;
        float target;

        // Rozbití stringu na jednotlivé hodnoty
        while (std::getline(ss, item, ',')) {
            try {
                input.push_back(std::stof(item));
            }
            catch (int e) {
                std::cout << "Invalid data" << std::endl;
            }
        }
        target = input.back();
        input.pop_back();

        // Ulo data do hlavního vektoru
        d.input.insert(d.input.end(), input.begin(), input.end());
        d.target.push_back(target);

        if (d.dimensions == -1) {
            d.dimensions = input.size();
        }
    }
    std::cout << "Dataset loaded, number of samples: " << d.target.size() << std::endl;
    return d;
}

Dataset get_batch(const Dataset& data, int batch_size, int batch_index, std::vector<int> indexes) {
    Dataset batch;

    batch.dimensions = data.dimensions;

    int start_idx = batch_index * batch_size;
    int end_idx = std::min(start_idx+batch_size, (int)data.target.size());

    // Jeden záznam
    for (int i = start_idx; i < end_idx; i++) {
        int current_index = indexes[i];
        // Pro kadı element z øádku
        for (int j = 0; j < data.dimensions; j++) {
            batch.input.push_back(data.input[current_index * data.dimensions + j]);
        }
        batch.target.push_back(data.target[current_index]);
    }

    return batch;
}

Dataset getDatasetByName(std::string name) {
    if (name == "dataset1") return dataset1;
    else if (name == "dataset2") return load_dataset("sonar_reduced_more.data");
    else if (name == "dataset3") return load_dataset("sonar_reduced.data");
    else if (name == "dataset4") return load_dataset("sonar.data");
    else if (name == "dataset5") return load_dataset("spambase.data");
    else throw std::invalid_argument("Dataset not found");
}