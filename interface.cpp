#include <iostream>
#include "experiment.h"
using namespace std;


int main(int argc, char* argv[]){
    // default parameters
    std::string data_path = "./dataset/7-19d-CO.csv";
    std::string output_path = "./output/1_T_drive_3.csv";
    int data_scale = 1000;
    int data_dimension = 19;
    int k = 10;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-data_path") {
            data_path = argv[++i];
        } else if (arg == "-output_path") {
            output_path = argv[++i];
        } else if (arg == "-data_scale") {
            data_scale = stoi(argv[++i]);
        } else if (arg == "-data_dimension") {
            data_dimension = stoi(argv[++i]);
        } else if (arg == "-k") {
            k = stoi(argv[++i]);
        }
    }

    cout << "running: " << data_path << endl;
    cout << "======================Strating experiment with k=" << k << "======================" << endl;

    Experiment* experiment = new Experiment(data_scale, data_dimension, k);
    experiment->set_file_path(data_path, output_path);

    // experiment->load("/mnt/e/f-means/dataset/7-19d-CO.csv");
    // experiment->write_distance("/mnt/e/f-means/dataset/dis7_COO.txt");

    // you can select your algorithms here
    experiment->test_Lloyd();
    // experiment->test_Elkan();
    // experiment->test_Hamerly();
    // experiment->test_Drake();
    // experiment->test_Yinyang();
    // experiment->test_DualTree();
    // experiment->test_NoBound();
    // experiment->test_NoKnn();
    // experiment->test_NoInB();
    // experiment->test_dask_means();
} 