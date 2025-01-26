#include "Hamerly.h"
#include "../utils/Utils.h"
#include <iostream>

using namespace Utils;

Hamerly::Hamerly(int max_iterations, double convergence_threshold)
    : KMeansBase(max_iterations, convergence_threshold) {}

void Hamerly::initParameters(int data_scale, int data_dimension, int k) {
    this->data_scale = data_scale;
    this->data_dimension = data_dimension;
    this->k = k;
    this->labels.assign(this->data_scale, -1);
    ub = std::vector<double>(data_scale, 0.0);
    lb = std::vector<double>(data_scale, 0.0);
    max_drift = 0.0;
}

void Hamerly::run() {
    int it = 0;     // iteration
    double start_time, end_time;
    start_time = clock();
    initializeCentroids();
    end_time = clock();
    init_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "initialize centroids in " << init_time << " s" << std::endl;

    // main loop
    do {
        start_time = clock();
        assignLabels();
        updateCentroids();

        end_time = clock();
        runtime[it] = double(end_time - start_time) / CLOCKS_PER_SEC;
        std::cout << "iter: " << it << ", runtime: " << runtime[it] << " s" << std::endl;
        it++;
    } while (/*!hasConverged() && */it < max_iterations);

    // show total runtime
    double total_runtime = init_time;
    for (size_t i = 0; i < max_iterations; i++) {
        total_runtime += runtime[i];
    }
    std::cout << "successfully run Hamerly in " << total_runtime << " s" << std::endl;

}

void Hamerly::assignLabels() {
    // clear all clusters firstly
    for (auto centroid : centroid_list) {
        centroid->cluster->clear();
    }

    for (int j = 0; j < data_scale; j++) {
        if (labels[j] != -1) {
            auto centroid = centroid_list[labels[j]];
            ub[j] += centroid->drift;
            lb[j] -= max_drift;
            if (ub[j] <= lb[j]) {
                centroid->cluster->addDataId(j);
                continue;
            }
        }

        auto data_point = dataset[j];
        int nearest_centroid_id = 0;
        double nearest_centroid_dis = HUGE_VAL;

        for (int i = 0; i < k; i++) {
            double dis = distance1(data_point, centroid_list[i]->coordinate);

            if (dis >= nearest_centroid_dis)
                continue;
            nearest_centroid_dis = dis;
            nearest_centroid_id = i;
        }

        labels[j] = nearest_centroid_id;
        centroid_list[nearest_centroid_id]->cluster->addDataId(j);
    }
}

void Hamerly::updateCentroids() {
    max_drift = 0.0;
    for (auto centroid : centroid_list) {
        std::vector<int> data_id_list = centroid->cluster->getDataIdList();
        int data_num = data_id_list.size();

        std::vector<double> new_coordinate(data_dimension, 0);
        for (int data_id : data_id_list) {
            new_coordinate = addVector(new_coordinate, dataset[data_id]);
        }
        if (data_num != 0) {
            new_coordinate = divideVector(new_coordinate, data_num);
            centroid->updateCoordinate(new_coordinate);
        }

        max_drift = std::max(max_drift, centroid->max_drift);
    }
}