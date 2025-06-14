#include "Datasets.cuh"


Dataset dataset1 = {
    {2},
    {0, 0, 0, 1, 1, 0, 1, 1},
    {0, 1, 1, 0}
};

Dataset dataset2 = {
    {2},
    {1, 1, 0, 0, 1, 0, 0, 1},
    {1, 1, 0, 0}
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

Dataset get_batch(const Dataset& data, int batch_size, int batch_index) {
    Dataset batch;

    int start_idx = batch_index * batch_size;
    int end_idx = std::min(start_idx + batch_size, (int)data.target.size());

    //std::cout << data.input.size() << std::endl;
    //std::cout << data.input[5347] << std::endl;
    // Jeden záznam
    for (int i = start_idx; i < end_idx; i++) {
        // Pro kadı element z øádku
        for (int j = 0; j < data.dimensions; j++) {
            //std::cout << i << " " << j << std::endl;
            //std::cout << i * data.dimensions + j << std::endl;
            batch.input.push_back(data.input[i * data.dimensions + j]);
        }
        batch.target.push_back(data.target[i]);
    }

    return batch;
}

Dataset getDatasetByName(std::string name) {
    if (name == "dataset1") return dataset1;
    else if (name == "dataset2") return dataset2;
    else if (name == "dataset3") return load_dataset("sonar.all-data");
    else if (name == "dataset4") return load_dataset("spambase_reduced.data");
    else if (name == "dataset5") return load_dataset("spambase.data");
    else throw std::invalid_argument("Dataset not found");
}