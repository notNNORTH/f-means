#ifndef LLOYD_H
#define LLOYD_H

#include "KMeansBase.h"
#include "../utils/Utils.h"

class Lloyd : public KMeansBase {
public:
    Lloyd(int max_iterations = MAX_ITERATIONS, double convergence_threshold = 0.001);

    void run() override;

    void runFairly() override;

protected:
    void assignLabels() override;

    void updateCentroids() override;
};

#endif // LLOYD_H