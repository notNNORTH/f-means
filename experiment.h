#include "algorithms/Lloyd.h"
#include "algorithms/DaskMeans.h"
#include "algorithms/NoInB.h"
#include "algorithms/NoKnn.h"
#include "algorithms/NoBound.h"
#include "algorithms/DualTree.h"
#include "algorithms/Hamerly.h"
#include "algorithms/Drake.h"
#include "algorithms/Yinyang.h"
#include "algorithms/Elkan.h"
#include "structure/BallTree.h"

#include "utils/Utils.h"
using namespace Utils;

class Experiment {
protected:
    std::string data_path;
    std::string output_path;

    int leaf_capacity = 30; // change the leaf node capacity if needed
    int data_scale;
    int data_dimension;
    int k;

    /* tmp */
    std::vector<std::vector<double>> dataset;

public:
    Experiment() {}
    Experiment(int data_scale, int data_dimension, int k)
        : data_scale(data_scale), data_dimension(data_dimension), k(k) {}
    ~Experiment() {}

    void set_file_path(const std::string& data_path, const std::string& output_path);

    void test_Lloyd();

    void test_dask_means();

    void test_NoInB();

    void test_NoKnn();

    void test_NoBound();

    void test_DualTree();

    void test_Hamerly();

    void test_Drake();

    void test_Yinyang();

    void test_Elkan();

    /* tmp */
    void load(const std::string& file_path);
    void write_distance(const std::string& file_path);
};

void Experiment::set_file_path(const std::string& data_path, const std::string& output_path) {
    this->data_path = data_path;
    this->output_path = output_path;
}

void Experiment::test_Lloyd() {
    cout << "=============starting Lloyd=============" << endl;
    Lloyd* lloyd = new Lloyd();
    lloyd->initParameters(data_scale, data_dimension, k);
    lloyd->load(data_path);
    lloyd->run();
    lloyd->measure();

    // f-means
    lloyd->equalDistanceGrouping();
    lloyd->runFairly();
    lloyd->measure();
    // lloyd->output("test.txt");
    // lloyd->writeRuntime(output_path);
    delete lloyd;
}

void Experiment::test_dask_means() {
    cout << "=============starting Dask-means=============" << endl;
    DaskMeans* dask_means = new DaskMeans(leaf_capacity);
    dask_means->initParameters(data_scale, data_dimension, k);
    dask_means->load(data_path);
    dask_means->run();
    dask_means->rewriteDataInCentroids();
    dask_means->output("test.txt");
    // dask_means->writeRuntime(output_path, "DaskMeans");
    delete dask_means;
}

void Experiment::test_NoInB() {
    cout << "=============starting NoInB=============" << endl;
    NoInB* noInB = new NoInB(leaf_capacity);
    noInB->initParameters(data_scale, data_dimension, k);
    noInB->load(data_path);
    noInB->run();
    // noInB->writeRuntime(output_path);
    delete noInB;
}

void Experiment::test_NoKnn() {
    cout << "=============starting NoKnn=============" << endl;
    NoKnn* noKnn = new NoKnn(leaf_capacity);
    noKnn->initParameters(data_scale, data_dimension, k);
    noKnn->load(data_path);
    noKnn->run();
    // noKnn->writeRuntime(output_path);
    delete noKnn;
}

void Experiment::test_NoBound() {
    cout << "=============starting NoBound=============" << endl;
    NoBound* no_bound = new NoBound();
    no_bound->initParameters(data_scale, data_dimension, k);
    no_bound->run(data_path.c_str());
    no_bound->writeRuntime(output_path, "NoBound");
    delete no_bound;
}

void Experiment::test_DualTree() {
    cout << "=============starting DualTree=============" << endl;
    DualTree* dual_tree = new DualTree(6);
    dual_tree->initParameters(data_scale, data_dimension, k);
    dual_tree->load(data_path);
    dual_tree->run();
    dual_tree->writeRuntime(output_path, "DualTree");
    delete dual_tree;
}

void Experiment::test_Hamerly() {
    cout << "=============starting Hamerly=============" << endl;
    Hamerly* hamerly = new Hamerly();
    hamerly->initParameters(data_scale, data_dimension, k);
    hamerly->load(data_path);
    hamerly->run();
    hamerly->writeRuntime(output_path, "Hamerly");
    delete hamerly;
}

void Experiment::test_Drake() {
    cout << "=============starting Drake=============" << endl;
    Drake* drake = new Drake();
    drake->initParameters(data_scale, data_dimension, k);
    drake->load(data_path);
    drake->run();
    drake->writeRuntime(output_path, "Drake");
    delete drake;
}

void Experiment::test_Yinyang() {
    cout << "=============starting Yinyang=============" << endl;
    Yinyang* yinyang = new Yinyang();
    yinyang->initParameters(data_scale, data_dimension, k);
    yinyang->load(data_path);
    yinyang->run();
    yinyang->writeRuntime(output_path, "Yinyang");
    delete yinyang;
}

void Experiment::test_Elkan() {
    cout << "=============starting Elkan=============" << endl;
    Elkan* elkan = new Elkan();
    elkan->initParameters(data_scale, data_dimension, k);
    elkan->load(data_path);
    elkan->run();
    elkan->writeRuntime(output_path, "Elkan");
    delete elkan;
}

void Experiment::load(const std::string& file_path) {
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

void Experiment::write_distance(const std::string& file_path) {
    std::ofstream outFile(file_path, std::ios::out | std::ios::app);
    if (!outFile) {
        throw std::runtime_error("Unable to open file: " + file_path);
    }
    // write init time
    const size_t rows = data_scale;
    const size_t cols = data_scale;

    // 创建一个大小为 100000 x 100000 的全零二维数组
    std::vector<std::vector<double>> array(rows, std::vector<double>(cols, 0));

    // write runtime in main loop
    for (size_t i = 0; i < data_scale; i++) {
        for (size_t j = i; j < data_scale; j++) {
            double dis = Utils::distance1(dataset[i], dataset[j]);
            array[i][j] = dis;
            array[j][i] = dis;
            cout << i << "," << j << endl;
        }
    }

    for (size_t i = 0; i < data_scale; i++) {
        for (size_t j = 0; j < data_scale; j++) {
            outFile << array[i][j];
            if (j != data_scale - 1) {
                outFile << ",";
            }
        }
        cout << i << endl;
        outFile << std::endl;
    }
    outFile.close();
}