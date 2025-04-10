cmake_minimum_required(VERSION 3.2)
project(cudaWrap)
add_subdirectory(pybind11)
pybind11_add_module(cudaWrap wraper.cpp)