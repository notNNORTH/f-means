#include "algorithms/Lloyd.h"
#include "algorithms/FMeans.h"
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

    void setFilePath(const std::string& data_path, const std::string& output_path);

    void testLloyd();

    void testFairMeans(double theta = 1.0);

    void testFMeans();

    void testDualTree();

    void testHamerly();

    void testDrake();

    void testYinyang();

    void testElkan();
};

void Experiment::setFilePath(const std::string& data_path, const std::string& output_path) {
    this->data_path = data_path;
    this->output_path = output_path;
}

void Experiment::testLloyd() {
    cout << "=============starting Lloyd=============" << endl;
    Lloyd* lloyd = new Lloyd();
    lloyd->initParameters(data_scale, data_dimension, k);
    lloyd->load(data_path);
    lloyd->run();
    // lloyd->writeRuntime(output_path, "Lloyd", dataset_name);
    delete lloyd;
}

void Experiment::testFairMeans(double theta) {
    cout << "=============starting Lloyd=============" << endl;
    Lloyd* lloyd = new Lloyd();
    lloyd->initParameters(data_scale, data_dimension, k);
    lloyd->load(data_path);
    lloyd->run();
    lloyd->measure();

    // f-means
    cout << "=============starting individual fair=============" << endl;
    lloyd->theta = theta;
    lloyd->equalDistanceGrouping();
    lloyd->runFairly();
    lloyd->measure();
    delete lloyd;
}

void Experiment::testFMeans() {
    cout << "=============starting F-means=============" << endl;
    FMeans* F_means = new FMeans(leaf_capacity);
    F_means->initParameters(data_scale, data_dimension, k);
    F_means->load(data_path);
    F_means->run();
    // F_means->writeRuntime(output_path, "FMeans", dataset_name);
    delete F_means;
}

void Experiment::testDualTree() {
    cout << "=============starting DualTree=============" << endl;
    DualTree* dual_tree = new DualTree(6);
    dual_tree->initParameters(data_scale, data_dimension, k);
    dual_tree->load(data_path);
    dual_tree->run();
    // dual_tree->writeRuntime(output_path, "DualTree", dataset_name);
    delete dual_tree;
}

void Experiment::testHamerly() {
    cout << "=============starting Hamerly=============" << endl;
    Hamerly* hamerly = new Hamerly();
    hamerly->initParameters(data_scale, data_dimension, k);
    hamerly->load(data_path);
    hamerly->run();
    // hamerly->writeRuntime(output_path, "Hamerly", dataset_name);
    delete hamerly;
}

void Experiment::testDrake() {
    cout << "=============starting Drake=============" << endl;
    Drake* drake = new Drake();
    drake->initParameters(data_scale, data_dimension, k);
    drake->load(data_path);
    drake->run();
    // drake->writeRuntime(output_path, "Drake", dataset_name);
    delete drake;
}

void Experiment::testYinyang() {
    cout << "=============starting Yinyang=============" << endl;
    Yinyang* yinyang = new Yinyang();
    yinyang->initParameters(data_scale, data_dimension, k);
    yinyang->load(data_path);
    yinyang->run();
    // yinyang->writeRuntime(output_path, "Yinyang", dataset_name);
    delete yinyang;
}

void Experiment::testElkan() {
    cout << "=============starting Elkan=============" << endl;
    Elkan* elkan = new Elkan();
    elkan->initParameters(data_scale, data_dimension, k);
    elkan->load(data_path);
    elkan->run();
    // elkan->writeRuntime(output_path, "Elkan", dataset_name);
    delete elkan;
}

