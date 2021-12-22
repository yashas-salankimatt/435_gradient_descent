
#include "LogisticRegression.h"
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>

void read_csv(std::string filename,
              std::vector<std::vector<float>> &features,
              std::vector<std::vector<float>> &labels)
{
  std::ifstream dataset(filename);
  if (!dataset.is_open())
  {
    std::cout << "Could not open file." << std::endl;
    exit(1);
  }
  std::string row, column;
  while (std::getline(dataset, row))
  {
    std::vector<float> temp_row;
    std::stringstream ss(row);
    while (std::getline(ss, column, ','))
    {
      temp_row.push_back(std::stof(column));
    }
    std::vector<float> temp_back = {temp_row.back()};
    labels.push_back(temp_back);
    temp_row.pop_back();
    features.push_back(temp_row);
  }
}

void print_vectors(std::vector<std::vector<float>> features,
                   std::vector<std::vector<float>> labels)
{
  std::cout << "################Feature Vector:################" << std::endl;
  for (int i = 0; i < features.size(); i++)
  {
    for (int j = 0; j < 4; j++)
    {
      std::cout << features[i][j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << "################End Feature Vector:################" << std::endl;
  std::cout << "################Label Vector:################" << std::endl;
  for (int i = 0; i < labels.size(); i++)
  {
    std::cout << labels[i][0] << std::endl;
  }
  std::cout << "################End Label Vector:################" << std::endl;
}

int main(int argc, char *argv[])
{
  // for debugging
  bool DEBUG = false;

  // create "feature" vector
  std::vector<std::vector<float>> features;
  // create "label" vector
  std::vector<std::vector<float>> labels;

  // read in data
  read_csv("../../data/dataset.csv", features, labels);

  // debug
  if (DEBUG)
  {
    print_vectors(features, labels);
  }

  // main algorithm
  float learning_rate = 0.1;
  LogisticRegression *logreg = new LogisticRegression(features, labels, learning_rate);

  std::cout << "-----------------Training...-----------------" << std::endl;
  logreg->train();
  std::cout << "-----------------End Training-----------------" << std::endl;

  return 0;
}