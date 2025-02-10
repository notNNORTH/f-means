#include "algorithms/Lloyd.h"
#include "algorithms/DaskMeans.h"
#include "algorithms/DualTree.h"
#include "algorithms/Hamerly.h"
#include "algorithms/Drake.h"
#include "algorithms/Yinyang.h"
#include "algorithms/Elkan.h"
#include "structure/BallTree.h"

#include "utils/Utils.h"
using namespace Utils;
using namespace std;

class Experiment {
protected:
    std::string data_path;
    std::string output_path;
    std::string dataset_name;

    int leaf_capacity = 30; // change the leaf node capacity if needed
    int data_scale;
    int data_dimension;
    int k;

    /* tmp */
    std::vector<std::vector<double>> dataset;

public:
    Experiment() {}
    Experiment(int data_scale, int data_dimension, int k, string dataset_name = "null")
        : data_scale(data_scale), data_dimension(data_dimension), k(k), dataset_name(dataset_name) {}
    ~Experiment() {}

    void set_file_path(const std::string& data_path, const std::string& output_path);

    void test_Lloyd(double theta = 1.0);

    void test_dask_means();

    void test_DualTree();

    void test_Hamerly();

    void test_Drake();

    void test_Yinyang();

    void test_Elkan();
};

void Experiment::set_file_path(const std::string& data_path, const std::string& output_path) {
    this->data_path = data_path;
    this->output_path = output_path;
}

void Experiment::test_Lloyd(double theta) {
    cout << "=============starting Lloyd=============" << endl;
    Lloyd* lloyd = new Lloyd();
    lloyd->initParameters(data_scale, data_dimension, k);
    lloyd->load(data_path);
    lloyd->run();
    // lloyd->measure();

    // f-means
    // lloyd->theta = theta;
    // lloyd->equalDistanceGrouping();
    // lloyd->runFairly();
    // lloyd->measure();
    // lloyd->output("test.txt");
    lloyd->writeRuntime(output_path, "Lloyd", dataset_name);
    delete lloyd;
}

void Experiment::test_dask_means() {
    cout << "=============starting Dask-means=============" << endl;
    DaskMeans* dask_means = new DaskMeans(leaf_capacity);
    dask_means->initParameters(data_scale, data_dimension, k);
    dask_means->load(data_path);
    dask_means->run();
    // dask_means->rewriteDataInCentroids();
    // dask_means->output("test.txt");
    dask_means->writeRuntime(output_path, "DaskMeans", dataset_name);
    delete dask_means;
}

void Experiment::test_DualTree() {
    cout << "=============starting DualTree=============" << endl;
    DualTree* dual_tree = new DualTree(6);
    dual_tree->initParameters(data_scale, data_dimension, k);
    dual_tree->load(data_path);
    dual_tree->run();
    dual_tree->writeRuntime(output_path, "DualTree", dataset_name);
    delete dual_tree;
}

void Experiment::test_Hamerly() {
    cout << "=============starting Hamerly=============" << endl;
    Hamerly* hamerly = new Hamerly();
    hamerly->initParameters(data_scale, data_dimension, k);
    hamerly->load(data_path);
    hamerly->run();
    hamerly->writeRuntime(output_path, "Hamerly", dataset_name);
    delete hamerly;
}

void Experiment::test_Drake() {
    cout << "=============starting Drake=============" << endl;
    Drake* drake = new Drake();
    drake->initParameters(data_scale, data_dimension, k);
    drake->load(data_path);
    drake->run();
    drake->writeRuntime(output_path, "Drake", dataset_name);
    delete drake;
}

void Experiment::test_Yinyang() {
    cout << "=============starting Yinyang=============" << endl;
    Yinyang* yinyang = new Yinyang();
    yinyang->initParameters(data_scale, data_dimension, k);
    yinyang->load(data_path);
    yinyang->run();
    yinyang->writeRuntime(output_path, "Yinyang", dataset_name);
    delete yinyang;
}

void Experiment::test_Elkan() {
    cout << "=============starting Elkan=============" << endl;
    Elkan* elkan = new Elkan();
    elkan->initParameters(data_scale, data_dimension, k);
    elkan->load(data_path);
    elkan->run();
    elkan->writeRuntime(output_path, "Elkan", dataset_name);
    delete elkan;
}

