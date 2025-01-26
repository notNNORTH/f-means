#include "Yinyang.h"
#include "Lloyd.h"
#include "../utils/Utils.h"
#include <iostream>

using namespace Utils;

Yinyang::Yinyang(int max_iterations, double convergence_threshold)
    : KMeansBase(max_iterations, convergence_threshold) {}

void Yinyang::initParameters(int data_scale, int data_dimension, int k) {
    this->data_scale = data_scale;
    this->data_dimension = data_dimension;
    this->k = k;
    this->labels.assign(this->data_scale, -1);
    group_num = k < 10 ? k : k / 10;
    ub = std::vector<double>(data_scale, 0.0);
    global_lb = std::vector<double>(data_scale, 0.0);
    group_lb = std::vector<std::vector<double>>(data_scale, std::vector<double>(group_num, 0.0));
    for (size_t i = 0; i < group_num; i++) {
        Group* group = new Group();
        groups.push_back(group); 
    }
    max_drift = 0.0;
}

void Yinyang::run() {
    int it = 0;     // iteration
    double start_time, end_time;
    start_time = clock();
    initializeCentroids();
    end_time = clock();
    init_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "initialize centroids in " << init_time << " s" << std::endl;

    do {
        start_time = clock();
        classicKMeansAssign();
        classicKMeansUpdate();

        end_time = clock();
        runtime[it] = double(end_time - start_time) / CLOCKS_PER_SEC;
        std::cout << "iter: " << it << ", runtime: " << runtime[it] << " s" << std::endl;
        it++;
    } while (!hasConverged() && it < 5);

    // set groups
    start_time = clock();
    setGroups();
    do {
        if (it != 5) {
            start_time = clock();
        }        
        assignLabels();
        updateCentroids();

        end_time = clock();
        runtime[it] = double(end_time - start_time) / CLOCKS_PER_SEC;
        std::cout << "iter: " << it << ", runtime: " << runtime[it] << " s" << std::endl;
        it++;
    } while (/*!hasConverged() && */it < MAX_ITERATIONS);

    // show total runtime
    double total_runtime = init_time;
    for (size_t i = 0; i < max_iterations; i++) {
        total_runtime += runtime[i];
    }
    std::cout << "successfully run Yinyang in " << total_runtime << " s" << std::endl;
}

void Yinyang::assignLabels() {
    for (auto centroid : centroid_list) {
        centroid->cluster->clear();
    }

    for (int j = 0; j < data_scale; j++) {
        std::vector<int> pruned_group_id;
        if (labels[j] != -1) {
            auto centroid = centroid_list[labels[j]];

            // 1.update global upper bound and global lower bound
            ub[j] += centroid->drift;
            global_lb[j] -= max_drift;

            // 2.update group lower bound
            for (size_t i = 0; i < group_num; i++) {
                group_lb[j][i] -= groups[i]->max_drift;
            }

            // global filter
            if (ub[j] <= global_lb[j]) {
                centroid->cluster->addDataId(j);
                continue;
            }

            // group filter
            for (size_t i = 0; i < group_num; i++) {
                if (ub[j] <= group_lb[j][i]) {
                    pruned_group_id.push_back(i);
                }
            }
        }
        // local filter
        auto data_point = dataset[j];
        int nearest_centroid_id = labels[j];
        double nearest_centroid_dis = HUGE_VAL;

        for (size_t i = 0; i < group_num; i++) {
            auto it = std::find(pruned_group_id.begin(), pruned_group_id.end(), i);
            if (it != pruned_group_id.end()) {
                continue;
            }

            for (auto id : groups[i]->centroid_id_list) {
                double dis = distance1(data_point, centroid_list[id]->coordinate);

                if (dis >= nearest_centroid_dis)
                    continue;
                nearest_centroid_dis = dis;
                nearest_centroid_id = id;
            }
        }

        labels[j] = nearest_centroid_id;
        centroid_list[nearest_centroid_id]->cluster->addDataId(j);
    }
}

void Yinyang::updateCentroids() {
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

    for (auto group : groups) {
        group->max_drift = 0.0;
        for (auto id : group->centroid_id_list) {
            group->max_drift = std::max(group->max_drift, centroid_list[id]->max_drift);
        }
    }
}

void Yinyang::classicKMeansAssign() {
    // clear all clusters firstly
    for (auto centroid : centroid_list) {
        centroid->getCluster()->clear();
    }

    for (int j = 0; j < data_scale; j++) {
        auto data_point = dataset[j];
        int nearest_centroid_id = 0;
        double nearest_centroid_dis = HUGE_VAL;

        for (int i = 0; i < k; i++) {
            double dis = distance1(data_point, centroid_list[i]->getCoordinate());

            if (dis >= nearest_centroid_dis)
                continue;
            nearest_centroid_dis = dis;
            nearest_centroid_id = i;
        }

        labels[j] = nearest_centroid_id;
        centroid_list[nearest_centroid_id]->getCluster()->addDataId(j);
    }
}

void Yinyang::classicKMeansUpdate() {
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

void Yinyang::setGroups() {
    Lloyd* lloyd = new Lloyd();
    lloyd->initParameters(k, data_dimension, group_num);
    lloyd->load(centroid_list);
    lloyd->max_iterations = 10;     // set the iteration number for grouping all centroids
    lloyd->run();

    // init each group
    for (size_t i = 0; i < k; i++) {
        int centroid_id = i;
        int group_id = lloyd->labels[i];
        groups[group_id]->centroid_id_list.push_back(centroid_id);
    }

    delete lloyd;
}

