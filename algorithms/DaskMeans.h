// DaskMeans.h
#ifndef DASKMEANS_H
#define DASKMEANS_H

#include <vector>
#include "KMeansBase.h"
#include "../utils/Utils.h"
#include "../structure/BallTree.h"
#include "../structure/KnnRes.h"

using namespace Utils;

class DaskMeans : public KMeansBase {
private:
    BallTree* data_index;
    BallTree* centroid_index;
    std::vector<double> inner_bound;
    std::vector<double> ub;
    int capacity;

public:
    DaskMeans(int capacity, int max_iterations = 30, double convergence_threshold = 0.001);

    ~DaskMeans() override;

    void run();

    void output(const std::string& file_path) override;

protected:
    void buildDataIndex(int capacity = 1);

    void buildCentroidIndex(int capacity = 1);

    void setInnerBound();

    void assignLabels(Node& node, double ub);
    
    void updateCentroids() override;

    void assignToCluster(Node& node, int centroid_id);

    void removeFromCluster(Node& node, bool flag);
    
};

#endif