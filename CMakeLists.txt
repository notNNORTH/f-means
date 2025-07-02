cmake_minimum_required(VERSION 3.10)

project(FMeansCpp VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")

find_package(Eigen3 REQUIRED)

include_directories(
    ${PROJECT_SOURCE_DIR}/algorithms
    ${PROJECT_SOURCE_DIR}/structure
    ${PROJECT_SOURCE_DIR}/utils
    ${EIGEN3_INCLUDE_DIR}
)

set(SOURCES
    utils/Utils.cpp
    algorithms/KMeansBase.cpp
    algorithms/Lloyd.cpp
    algorithms/FMeans.cpp
    algorithms/DualTree.cpp
    algorithms/Hamerly.cpp
    algorithms/Drake.cpp
    algorithms/Yinyang.cpp
    algorithms/Elkan.cpp
    interface.cpp
    structure/BallTree.cpp
    structure/Node.cpp
    structure/Centroid.cpp
    structure/Cluster.cpp
    structure/KdTree.cpp
    structure/KdTreeNode.cpp
)

add_executable(FMeans ${SOURCES})
