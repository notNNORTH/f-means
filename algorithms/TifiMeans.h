// TifiMeans.h
#ifndef TifiMEANS_H
#define TifiMEANS_H

#include <vector>
#include "KMeansBase.h"
#include "../utils/Utils.h"
#include "../structure/BallTree.h"
#include "../structure/KnnRes.h"

using namespace Utils;

class TifiMeans : public KMeansBase {
protected:
    BallTree* data_index;
    BallTree* centroid_index;
    std::vector<double> inner_bound;
    std::vector<double> ub;
    int capacity;

    // for debug
    std::vector<double> inner_id;
    double pruned_point = 0.0;

public:
    TifiMeans(int capacity, int max_iterations = MAX_ITERATIONS, double convergence_threshold = 0.001);

    ~TifiMeans() override;

    void run() override;

    void output(const std::string& file_path) override;

    void rewriteDataInCentroids();

protected:
    void buildDataIndex(int capacity = 1);

    void buildCentroidIndex(int capacity = 5);

    void setInnerBound();

    void assignLabels(Node& node, double ub);
    
    void updateCentroids() override;

    void assignToCluster(Node& node, int centroid_id);   
};

#endif