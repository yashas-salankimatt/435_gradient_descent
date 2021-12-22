#ifndef HELPERS_H
#define HELPERS_H
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

void read_csv(std::string filename,
              std::vector<std::vector<float>> &features,
              std::vector<std::vector<float>> &labels,
              int dataset_size);

void printVectorMatrix(const std::vector<std::vector<float>> a);

void printFlatMatrix(const float *mat, int rows, int cols);

void flattenVectorMatrix(const std::vector<std::vector<float>> mat, float *flattenedMat);

#endif