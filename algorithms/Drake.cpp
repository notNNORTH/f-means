#include "Drake.h"
#include "../utils/Utils.h"
#include <iostream>
#include <algorithm>
#include <cmath>

using namespace Utils;

Drake::Drake(int max_iterations, double convergence_threshold)
    : KMeansBase(max_iterations, convergence_threshold) {
        max_drift = 0.0;
        max_b = 0;
    }

void Drake::initParameters(int data_scale, int data_dimension, int k) {
    this->data_scale = data_scale;
    this->data_dimension = data_dimension;
    this->k = k;
    this->labels.assign(this->data_scale, -1);
    b = k / 4;
    ub = std::vector<double>(data_scale, 0.0);
    a = std::vector<std::vector<KnnRes*>>(data_scale, std::vector<KnnRes*>());
}

void Drake::run() {
    int it = 0;
    double start_time, end_time;
    start_time = clock();
    initializeCentroids();
    for (size_t i = 0; i < data_scale; i++) {
        sortCentroids(i, b + 1, centroid_list);
    }
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
    std::cout << "successfully run Drake in " << total_runtime << " s" << std::endl;

}

void Drake::assignLabels() {
    for (auto centroid : centroid_list) {
        centroid->cluster->clear();
    }

    for (int j = 0; j < data_scale; j++) {
        if (labels[j] != 1) {
            // 1.update upper bound and lower bound
            ub[j] += centroid_list[a[j][0]->id]->drift;
            for (int i = 1; i < b + 1; i++) {
                a[j][i]->dis -= centroid_list[a[j][i]->id]->drift;
            }
        }
        bool flag = false;
        for (int z = 1; z < b + 1; z++) {
            // 2.check whether the node can be pruned
            if (ub[j] <= a[j][z]->dis) {
                sortCentroids(j, z + 1, centroid_list, a[j]);
                centroid_list[a[j][0]->id]->cluster->addDataId(j);
                labels[j] = a[j][0]->id;
                max_b = std::max(max_b, z);
                flag = true;
                break;
            }
        }
        if (!flag) {
            sortCentroids(j, b + 1, centroid_list);
            centroid_list[a[j][0]->id]->cluster->addDataId(j);
            labels[j] = a[j][0]->id;
        }
    }
    this->b = std::max(k / 4, max_b);
}

void Drake::updateCentroids() {
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

// given point_id, sort and get q nearest centroids
void Drake::sortCentroids(int point_id, int q, std::vector<Centroid*> cent_list) {
    
    a[point_id].clear();
    std::vector<double> point = dataset[point_id];
    std::partial_sort(cent_list.begin(), cent_list.begin() + q, cent_list.end(), 
            [this, point](const Centroid* a, const Centroid* b) {
        double dist_a = distance1(point, a->coordinate);
        double dist_b = distance1(point, b->coordinate);
        return dist_a < dist_b;
    });

    ub[point_id] = distance1(cent_list[0]->coordinate, point);
    for (size_t i = 0; i < q; i++) {
        KnnRes* nearest_cent = new KnnRes();
        nearest_cent->dis = distance1(cent_list[i]->coordinate, point);
        nearest_cent->id = cent_list[i]->centroid_id;
        a[point_id].push_back(nearest_cent);
    }
}

void Drake::sortCentroids(int point_id, int q, std::vector<Centroid*> cent_list,
        std::vector<KnnRes*>& res) {
    std::vector<double> point = dataset[point_id];
    for (size_t i = 0; i < q; i++) {
        res[i]->dis = distance1(cent_list[res[i]->id]->coordinate, point);
    }
    
    std::partial_sort(res.begin(), res.begin() + q, res.end(), 
            [this](const KnnRes* a1, const KnnRes* b1) {
        return a1->dis < b1->dis;
    });

    ub[point_id] = res[0]->dis;
}