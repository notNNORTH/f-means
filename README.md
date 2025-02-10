# Tifi-means
## 1 Introduction

***Tifi-means*** is a fast, memory-efficient, and cost-predictable <u>**da**</u>taset <u>**s**</u>implification **<u>$k$</u>-means** algorithm for large-scale spatial vectors. This repo holds the source code and scripts for reproducing the key experiments of our paper: *On Simplifying Large-Scale Spatial Vectors: Fast, Memory-Efficient, and Cost-Predictable k-means*.


## 2 Comparison Algorithms

#### [Least squares quantization in PCM](https://hal.science/hal-04614938/document)

- The most widely used $k$-means algorithm.
- The code is in `./algorithms/Lloyd.h` and `./algorithms/Lloyd.cpp`. 

#### [Using the Triangle Inequality to Accelerate k-Means](https://cdn.aaai.org/ICML/2003/ICML03-022.pdf)

- For each spatial vector, this algorithm maintains one global upper bound and $k$ lower bounds, and uses the triangle inequality to accelerate $k$-means.
- The code is in `./algorithms/Elkan.h` and `./algorithms/Elkan.cpp`.

#### [Making k-means even faster](https://epubs.siam.org/doi/pdf/10.1137/1.9781611972801.12)

- Using the triangle inequality. But for each spatial vector, only one upper bound and the minimum lower bound to any cluster centroids are maintained.
- The code is in `./algorithms/Hamerly.h` and `./algorithms/Hamerly.cpp`.

#### [Accelerated k-means with adaptive distance bounds](http://opt.kyb.tuebingen.mpg.de/papers/opt2012_paper_13.pdf)

- For each spatial vector, the algorithm sorts all centroids according to the distance to the spatial vector and then maintains $1<b<k$ nearest distances as lower bounds.
- The code is in `./algorithms/Drake.h` and `./algorithms/Drake.cpp`.

#### [Yinyang K-Means: A Drop-In Replacement of the Classic K-Means with Consistent Speedup](https://proceedings.mlr.press/v37/ding15.pdf)

- The algorithm groups the clusters using classic $k$-means first and maintains bounds for each group. Then it implements a pruning method with multiple filters to accelerate $k$-means.
- The code is in `./algorithms/Yinyang.h` and `./algorithms/Yinyang.cpp`.


#### [A Dual-Tree Algorithm for Fast k-means Clustering with Large k](https://epubs.siam.org/doi/pdf/10.1137/1.9781611974973.34)

- Organize spatial vectors using a dual-tree structure to accelerate the execution of the $k$-means algorithm.
- The code is in `./algorithms/DualTree.h` and `./algorithms/DualTree.cpp`.

## 3 How to Run Tifi-means

### 3.1 Code with comparison algorithms

You can find all source code of comparison algorithms in `./algorithms/`. 


**Compile**

Go to the project file `./Tifi-means-cpp/`, and then enter the following command in the terminal:

```sh
mkdir build
cd build
cmake ..
make
```

**Parameter Configuration and Run:**

To seccessfully run Tifi-means, several parameters are needed:

```sh
-data_path        # the path to your input dataset
-output_path      # the path to your output logs
-dataset          # (optional) an alias of your dataset
-data_scale       # the number of points involved in clustering
-data_dimension   # data dimension of your dataset
-k                # the number of clusters
```

Below is an example of an executable command. Make sure you are in the `./build/` directory before proceeding.

```sh
./TifiMeans -data_path ../dataset/your_dataset.csv -dataset an_alias -data_scale 1000000 -data_dimension 3 -k 200
```
