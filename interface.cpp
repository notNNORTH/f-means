#include <iostream>
#include "experiment.h"
using namespace std;


int main(int argc, char* argv[]){
    // default parameters
    std::string data_path = "./dataset/1_2d_T_drive_100000.txt";
    std::string output_path = "../output/f-means.csv";
    std::string dataset = "null";
    int data_scale = 1000;
    int data_dimension = 2;
    int k = 10;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-data_path") {
            data_path = argv[++i];
        } else if (arg == "-output_path") {
            output_path = argv[++i];
        } else if (arg == "-data_scale") {
            data_scale = stoi(argv[++i]);
        } else if (arg == "-dataset") {
            dataset = argv[++i];
        } else if (arg == "-data_dimension") {
            data_dimension = stoi(argv[++i]);
        } else if (arg == "-k") {
            k = stoi(argv[++i]);
        }
    }

    cout << "running: " << data_path << endl;

    Experiment* experiment = new Experiment(data_scale, data_dimension, k, dataset);
    experiment->setFilePath(data_path, output_path);

    // for speedup
    experiment->testLloyd();
    experiment->testElkan();
    experiment->testHamerly();
    experiment->testDrake();
    experiment->testYinyang();
    experiment->testDualTree();
    experiment->testTifiMeans();

    // for fairness
    // experiment->testFairMeans();
} 