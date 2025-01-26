#include "KMeansBase.h"
#include "Centroid.h"
#include <iostream>
#include <random>
#include <set>
#include <fstream>
#include <sstream>
#include <Utils.h>

using namespace Utils;
using namespace std;

KMeansBase::KMeansBase(int max_iterations, double convergence_threshold)
    : max_iterations(max_iterations), convergence_threshold(convergence_threshold) {
        runtime = std::vector(MAX_ITERATIONS, 0.0);
    }

KMeansBase::~KMeansBase() {
    for (Centroid* centroid : centroid_list) { delete centroid; }
}

void KMeansBase::initParameters(int data_scale, int data_dimension, int k) {
    this->data_scale = data_scale;
    this->data_dimension = data_dimension;
    this->k = k;
    this->labels.assign(this->data_scale, -1);
    this->distances.assign(this->data_scale, -1);
    this->group_labels.assign(this->data_scale, -1);
}

void KMeansBase::setK(int k) {
    this->k = k;
}

// load data into para::dataset
void KMeansBase::load(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << file_path << std::endl;
        return;
    }

    dataset.clear();
    std::string line;
    int count = 0;

    while (std::getline(file, line)) {
        if (count >= data_scale) break;

        std::stringstream ss(line);
        std::vector<double> data_point;
        double value;

        while (ss >> value) {
            data_point.push_back(value);
            if (ss.peek() == ',' || ss.peek() == ' ') ss.ignore();
        }

        if (data_point.size() == data_dimension) {
            dataset.push_back(data_point);
            count++;
        } else {
            std::cerr << "Warning: Ignoring line with incorrect dimension: " << line << std::endl;
        }
    }

    file.close();
}

void KMeansBase::load(std::vector<Centroid*>& centroid_list) {
    if (dataset.size() != 0) {
        dataset.clear();
    }
    for (auto centroid : centroid_list) {
        dataset.push_back(centroid->coordinate);
    }
}

// output k cluster with the schema as {cluster_id: int, point_id_list: list()}
void KMeansBase::output(const std::string& file_path) {
    // std::ofstream file(file_path, std::ios::app);
    std::ofstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + file_path);
    }

    for (auto centroid : centroid_list) {
        Cluster* clus = centroid->getCluster();
        file << "{cluster_id: " << clus->getClusterId() << ", point_id_list: [";

        std::vector<int> data_id_list = clus->getDataIdList();
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

// update centroids by labels, and then calculate the sum of squared errors using labels
void KMeansBase::measure() {
    // 0. update distance and get max disatance
    double max_distance = 0.0;
    for (int i = 0; i < data_scale; i++) {
        int label = labels[i];
        std::vector<double> data_point = dataset[i];
        std::vector<double> centroid_coordinate = centroid_list[label]->getCoordinate();
        distances[i] = distance1(data_point, centroid_coordinate);
        if (distances[i] > max_distance) {
            max_distance = distances[i];
        }
    }
    std::cout << "max distance: " << max_distance << std::endl;

    // 1. get sse
    double sse = 0.0;
    for (int i = 0; i < data_scale; i++) {
        sse += distances[i] * distances[i];
    }
    std::cout << "SSE: " << sse << std::endl;

    // 2. get variance
    // divide the distances into k groups based on labels
    std::vector<std::vector<double>> groups(k);
    for (int i = 0; i < data_scale; i++) {
        int label = labels[i];
        groups[label].push_back(distances[i]);
    }
    // calculate the mse for each group
    std::vector<double> mses(k);
    for (int i = 0; i < k; i++) {
        double mean_dis = std::accumulate(groups[i].begin(), groups[i].end(), 0.0) / groups[i].size();
        double sum = 0.0;
        for (double distance : groups[i]) {
            sum += (distance - mean_dis) * (distance - mean_dis);
        }
        if (groups[i].size() > 0) {
            mses[i] = sum / groups[i].size();
        }
    }
    double variance = 0.0;
    for (int i = 0; i < k; i++) {
        variance += mses[i];
    }
    std::cout << "variance: " << variance << std::endl;
}

// set group_labels by distances where each group has the same number of points
void KMeansBase::equalPointsGrouping() {
    std::vector<double> sorted_distances = distances;
    std::sort(sorted_distances.begin(), sorted_distances.end());

    int group_size = data_scale / g;
    int remainder = data_scale % g;
    std::vector<int> group_sizes(g, group_size);
    for (int i = 0; i < remainder; i++) {
        group_sizes[i]++;
    }

    int start_index = 0;
    for (int i = 0; i < g; i++) {
        int end_index = start_index + group_sizes[i];
        double threshold = sorted_distances[end_index - 1];
        for (int k = 0; k < data_scale; k++) {
            if (distances[k] <= threshold) {
                group_labels[k] = g - i;
            }
        }

        start_index = end_index;
    }
}

// set group_labels by distances where each group has the same distance
void KMeansBase::equalDistanceGrouping() {
    double max_distance = *std::max_element(distances.begin(), distances.end());
    for (size_t i = 0; i < data_scale; i++) {
        group_labels[i] = static_cast<int>((1 - distances[i] / max_distance) * g + 1);
    }
}

// write the runtime to the given file
void KMeansBase::writeRuntime(const std::string& file_path, string name) {
    std::ofstream outFile(file_path, std::ios::out | std::ios::app);
    if (!outFile) {
        throw std::runtime_error("Unable to open file: " + file_path);
    }
    // write init time
    outFile << name;
    outFile << ",";
    runtime[0] += init_time;
    // write runtime in main loop
    for (size_t i = 0; i < MAX_ITERATIONS; i++) {
        outFile << runtime[i];
        if (i != MAX_ITERATIONS - 1) {
            outFile << ",";
        }
    }
    outFile << std::endl;
    outFile.close();
}

// initialize all centroids randomly
void KMeansBase::initializeCentroids() {
    centroid_list.clear();
    std::set<int> selected_ids;

    for (int i = 0; i < k; i ++) {
        unsigned seed = 6;      // fixed seed 6
        std::mt19937 gen(seed);
        // std::random_device seed;    // random seed
        // std::mt19937 gen(seed());
        std::uniform_int_distribution<> distrib(0, data_scale - 1);

        int point_id;
        do {
            point_id = distrib(gen);
        } while (selected_ids.find(point_id) != selected_ids.end());

        selected_ids.insert(point_id);
        Centroid* centroid = new Centroid(point_id, dataset[point_id], i);

        centroid_list.push_back(centroid);
    }
}

// assign all data points to a cluster with an intiger label
void KMeansBase::assignLabels() {}

// update each centroid as the mean of each cluster
void KMeansBase::updateCentroids() {
    for (auto centroid : centroid_list) {
        std::vector<int> data_id_list = centroid->getCluster()->getDataIdList();
        int data_num = data_id_list.size();

        std::vector<double> new_coordinate(data_dimension, 0);
        for (int data_id : data_id_list) {
            new_coordinate = addVector(new_coordinate, dataset[data_id]);
        }
        if (data_num != 0) {
            new_coordinate = divideVector(new_coordinate, data_num);
            centroid->updateCoordinate(new_coordinate);
        }
    }
}

// update each centroid as the mean of each cluster using labels
void KMeansBase::updateCentroidsFairly() {
    for (int i = 0; i < k; i++) {
        std::vector<double> new_coordinate(data_dimension, 0);
        double sum_weight = 0.0;
        for (int j = 0; j < data_scale; j++) {
            if (labels[j] == i) {
                double weight = pow(1.0 / group_labels[j], theta);
                std::vector<double> weighted_point = multiplyVector(dataset[j], weight);
                new_coordinate = addVector(new_coordinate, weighted_point);
                sum_weight += weight;
            }
        }
        if (sum_weight != 0.0) {
            new_coordinate = divideVector(new_coordinate, sum_weight);
            centroid_list[i]->updateCoordinate(new_coordinate);
        }
    }
}

// determine whether the algorithm has converged
bool KMeansBase::hasConverged() {
    double sum_drift = 0.0;
    
    for (auto centroid : centroid_list) {
        sum_drift += centroid->drift;
    }

    // cout << "sum drift: " << sum_drift << endl;
    return sum_drift <= convergence_threshold;
}
