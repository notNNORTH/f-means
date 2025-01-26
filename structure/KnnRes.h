#ifndef KNNRES_H
#define KNNRES_H

#include <limits>

class KnnRes {
public:
    double dis;
    int id;

    KnnRes() {
        dis = std::numeric_limits<double>::max();
        id = -1;
    }
    KnnRes(double dis): dis(dis) { id = -1; }
    KnnRes(double dis, int id): dis(dis), id(id) {}
    ~KnnRes() {}
};

#endif