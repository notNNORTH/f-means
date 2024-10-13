#include <iostream>
#include "experiment.h"
using namespace std;


int main(){
    // T-drive
    // std::string data_path = "/mnt/e/torch-clus/dataset/1_2d_T_drive_11317142.txt";
    // std::string output_path = "/home/lzp/cs/dask-means-cpp/output/T-drive-AAA.csv";
    // Porto
    // std::string data_path = "/mnt/e/torch-clus/dataset/2_2d_Porto_1277582.txt";
    // std::string output_path = "/home/lzp/cs/dask-means-cpp/output/Porto-AAA.csv";
    // ShapeNet
    std::string data_path = "/mnt/e/torch-clus/dataset/4_3d_Shapenet_9999999.txt";
    std::string output_path = "/home/lzp/cs/dask-means-cpp/output/Shapenet-AAA.csv";
    
    int data_scale = 1000000;
    int data_dimension = 3;
    std::vector<int> k_list(2);
    k_list[0] = 1000;
    k_list[1] = 10000;

    // for (auto k : k_list) {
    //     size_t pos = output_path.find("AAA");
    //     if (pos != std::string::npos) {
    //         output_path.replace(pos, 3, std::to_string(k));
    //     }
    //     cout << "======================Strating experiment with k=" << k << "======================" << endl;

    //     Experiment* experiment = new Experiment(data_scale, data_dimension, k);
    //     experiment->set_file_path(data_path, output_path);

    //     experiment->test_Lloyd();
    //     experiment->test_DualTree();
    //     experiment->test_NoBound();

    //     // output_path = "/home/lzp/cs/dask-means-cpp/output/T-drive-AAA.csv";
    //     // output_path = "/home/lzp/cs/dask-means-cpp/output/Porto-AAA.csv";
    //     output_path = "/home/lzp/cs/dask-means-cpp/output/Shapenet-AAA.csv";
    // }

    for (auto k : k_list) {
        size_t pos = output_path.find("AAA");
        if (pos != std::string::npos) {
            output_path.replace(pos, 3, std::to_string(k));
        }
        // output_path = "/home/lzp/cs/dask-means-cpp/output/test.csv";
        cout << "======================Strating experiment with k=" << k << "======================" << endl;

        Experiment* experiment = new Experiment(data_scale, data_dimension, k);
        experiment->set_file_path(data_path, output_path);

        experiment->test_NoKnn();
        experiment->test_NoInB();
        experiment->test_dask_means();

        output_path = "/home/lzp/cs/dask-means-cpp/output/Porto-AAA.csv";
    }

}