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
    // int k_list[5] = {10, 50, 100, 150, 200};
    // int k_list[1] = {10};

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


    // for (auto k : k_list) {
    //     cout << "======================Strating experiment with k=" << k << "======================" << endl;
    //     Experiment* experiment = new Experiment(data_scale, data_dimension, k);
    //     experiment->set_file_path(data_path, output_path);

    //     // experiment->load("/mnt/e/f-means/dataset/7-19d-CO.csv");
    //     // experiment->write_distance("/mnt/e/f-means/dataset/dis7_COO.txt");

    //     // you can select your algorithms here
    //     double theta_list[2] = {0.05, 3.0};
    //     for (auto theta : theta_list) {
    //         cout << "theta =" << theta << "======================" << endl;
    //         experiment->test_Lloyd(theta);
    //     }
    // }
    Experiment* experiment = new Experiment(data_scale, data_dimension, k, dataset);
    experiment->set_file_path(data_path, output_path);
    // experiment->test_Lloyd();
    experiment->test_DualTree();
    experiment->test_Elkan();
    // experiment->test_Hamerly();
    // experiment->test_Drake();
    // experiment->test_Yinyang();
    // experiment->test_dask_means();
} 