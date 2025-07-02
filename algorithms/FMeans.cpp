#include "FMeans.h"
#include "../structure/KnnRes.h"
#include "../structure/Node.h"
#include "../utils/Utils.h"
#include <fstream>
#include <sstream>
#include <iostream>

using namespace Utils;

FMeans::FMeans(int capacity, int max_iterations, double convergence_threshold)
    : KMeansBase(max_iterations, convergence_threshold) {
        this->capacity = capacity;
        data_index = nullptr;
        centroid_index = nullptr;
    }

FMeans::~FMeans() {
    delete data_index;
    delete centroid_index;
    data_index = nullptr;
    centroid_index = nullptr;
}

void FMeans::run() {
    int it = 0;     // iteration
    double start_time, end_time;
    start_time = clock();

    buildDataIndex(this->capacity);
    initializeCentroids();

    end_time = clock();
    init_time = double(end_time - start_time) / CLOCKS_PER_SEC;
    std::cout << "build index and initialize in " << init_time << " s" << std::endl;

    // main loop
    do {
        start_time = clock();
        buildCentroidIndex();
        if (it == 0) {
            ub = std::vector<double>(k, std::numeric_limits<double>::max());
        } else {
            for (int i = 0; i < k; i++) {
                ub[i] = inner_bound[i] + centroid_list[i]->drift + centroid_list[i]->max_drift;
            }
        }
        setInnerBound();
        assignLabels(*data_index->root, std::numeric_limits<double>::max());
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
    // std::cout << "iter: " << it << ", pruned totally: " << pruned_point / 1000000 << std::endl;
    std::cout << "successfully run F-means in " << total_runtime << " s" << std::endl;
}

void FMeans::output(const std::string& file_path) {
    std::ofstream file(file_path, std::ios::app);
    // std::ofstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }

    for (auto centroid : centroid_list) {
        Cluster* clus = centroid->getCluster();
        file << "{cluster_id: " << clus->getClusterId() << ", point_id_list: [";

        std::vector<int> data_id_list = clus->getAllDataId();
        for (size_t i = 0; i < data_id_list.size(); ++i) {
            file << data_id_list[i];
            if (i < data_id_list.size() - 1) {
                file << ", ";
            }
        }
        file << "]}" << std::endl;
    }

    file.close();
}

void FMeans::rewriteDataInCentroids() {
    for (int i = 0; i < k; i++) {
        centroid_list[i]->getCluster()->clear();
    }
    for (int i = 0; i < data_scale; i++) {
        centroid_list[labels[i]]->getCluster()->addDataId(i);
    }
}

void FMeans::buildDataIndex(int capacity) {
    data_index = new BallTree(capacity, data_scale);
    data_index->buildBallTree(dataset, data_scale);
}

void FMeans::buildCentroidIndex(int capacity) {
    if (centroid_index != nullptr) {
        delete centroid_index;
        centroid_index = nullptr;
    }
    centroid_index = new BallTree(capacity);
    centroid_index->buildBallTree(centroid_list, k);
}

void FMeans::setInnerBound() {
    // using index ball-tree knn to set inner bound
    this->inner_bound = std::vector<double>(k, -1.0);
    this->inner_id = std::vector<double>(k, -1.0);
    for (int i = 0; i < k; i++) {
        std::vector<KnnRes*> res(2);
        for (int j = 0; j < 2; j++) {
            res[j] = new KnnRes(ub[i]);
        }
        ballTree2nn(centroid_list[i]->coordinate, *(centroid_index->root), res, centroid_list);
        inner_bound[i] = res[1]->dis;
        inner_id[i] = res[1]->id;
        for (int j = 0; j < 2; j++) {
            delete res[j];
        }
    }
}

void FMeans::assignLabels(Node& node, double ub) {
    // 1. if the node is assigned before (pruning 1)
    if (node.centroid_id != -1  && 
        distance1(node.pivot, centroid_list[node.centroid_id]->getCoordinate()) 
        + node.radius < inner_bound[node.centroid_id] / 2) 
        { return; }

    // 2. find two nearest centroid for the node (pruning 2)
    std::vector<KnnRes*> res(2);
    for (int i = 0; i < 2; i++) {
        res[i] = new KnnRes(ub);
    }
    ballTree2nn(node.pivot, *(centroid_index->root), res, centroid_list);
    if (res[1]->dis - res[0]->dis > 2 * node.radius) {
        if (res[0]->id == node.centroid_id) {
            // pruned_point += node.point_number;
            return;
        }
        assignToCluster(node, res[0]->id);
        Cluster* new_cluster = centroid_list[node.centroid_id]->cluster;
        new_cluster->dataIn(node.point_number, node.sum_vector);
        // pruned_point += node.point_number;
        return;
    }

    if (!node.leaf) {
        // 3. split the node into two child node
        assignToCluster(node, -1);
        if (node.leftChild != nullptr) {
            assignLabels(*node.leftChild, res[1]->dis + node.radius);
        }
        if (node.rightChild != nullptr) {
            assignLabels(*node.rightChild, res[1]->dis + node.radius);
        }
    } else {
        if (node.centroid_id != -1) {
            Cluster* old_cluster = centroid_list[node.centroid_id]->cluster;
            old_cluster->dataOut(node.point_number, node.sum_vector);
        }
        // 4. assign each point in the leaf node
        for (int i = 0; i < node.point_number; i++) {
            std::vector<double> data = dataset[node.data_id_list[i]];
            int centroid_id = node.centroid_id_for_data[i];
            // if the point is assigned before
            if (centroid_id != -1 && distance1(data, centroid_list[centroid_id]->coordinate) 
                < inner_bound[centroid_id] / 2 ) {
                    // pruned_point += 1;
                    return;
                }

            // use 1nn
            KnnRes* res = new KnnRes(ub);
            ballTree1nn(data, *(centroid_index->root), *res, centroid_list);
            if (centroid_id != -1) {
                Cluster* old_cluster = centroid_list[centroid_id]->cluster;
                old_cluster->dataOut(1, data);
            }
            centroid_id = res[0].id;
            labels[node.data_id_list[i]] = centroid_id;
            // node.centroid_id_for_data[i] = centroid_id;
            Cluster* new_cluster = centroid_list[centroid_id]->cluster;
            new_cluster->dataIn(1, data);

            delete res;
        }
    }
    for (int i = 0; i < 2; i++) {
        delete res[i];
    }
}

void FMeans::updateCentroids() {
    for (auto centroid : centroid_list) {
        Cluster* cluster = centroid->cluster;
        std::vector<double> new_coordinate;
        if (cluster->point_number != 0) {
            new_coordinate = divideVector(cluster->sum_vec, cluster->point_number);
            centroid->updateCoordinate(new_coordinate);
        }
    }
}

// assign a node and its child node to a centroid,
// and remove all nodes from previous centroids is exist
void FMeans::assignToCluster(Node& node, int centroid_id) {
    if (node.centroid_id != -1) {
        Cluster* old_cluster = centroid_list[node.centroid_id]->cluster;
        old_cluster->dataOut(node.point_number, node.sum_vector);
    }

    node.centroid_id = centroid_id;
    if (node.leaf) {
        for (int i = 0; i < node.point_number; i++) {
            int clu_id = node.centroid_id_for_data[i];
            if (clu_id != -1) {
                Cluster* old_cluster = centroid_list[clu_id]->cluster;
                old_cluster->dataOut(1, dataset[node.data_id_list[i]]);
            }
            labels[node.data_id_list[i]] = centroid_id;
            // node.centroid_id_for_data[i] = centroid_id;
        }
    } else {
        if (node.leftChild != nullptr) {
            assignToCluster(*node.leftChild, centroid_id);
        }
        if (node.rightChild != nullptr) {
            assignToCluster(*node.rightChild, centroid_id);
        }
    }
}