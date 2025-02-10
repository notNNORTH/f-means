// KMeansBase.h
#ifndef KMEANSBASE_H
#define KMEANSBASE_H

#define MAX_ITERATIONS 20

#include <vector>
#include <algorithm>
#include <string>

#include "../structure/Centroid.h"
#include "../structure/Cluster.h"

class KMeansBase {
public:
    std::vector<std::vector<double>> dataset;
    std::vector<Centroid*> centroid_list;   // remember to release the memory
    std::vector<int> labels;        // noting which cluster the data point is assigned to
    int data_scale;
    int data_dimension;
    int k;
    int max_iterations;
    double convergence_threshold;
    std::vector<double> runtime;
    double init_time = 0.0;

    // for f-means
    std::vector<double> distances;  // the distance between the data point and its nearest centroid
    std::vector<int> group_labels;
    int g = 10;     // group number
    double theta = 0.05;

public:
    KMeansBase(int max_iterations = MAX_ITERATIONS, double convergence_threshold = 1e-4);

    virtual ~KMeansBase();

    virtual void initParameters(int data_scale, int data_dimension, int k);

    void setK(int k); 

    void load(const std::string& file_path);

    void load(std::vector<Centroid*>& centroid_list);

    virtual void run() {};

    virtual void runFairly() {};

    virtual void output(const std::string& file_path);

    void measure();

    void equalPointsGrouping();

    void equalDistanceGrouping();

    void writeRuntime(const std::string& file_path, std::string name = "null", std::string dataset_name = "null");

protected:
    virtual void initializeCentroids();

    virtual void assignLabels();

    virtual void updateCentroids();

    virtual void updateCentroidsFairly();

    virtual bool hasConverged();
};

#endif