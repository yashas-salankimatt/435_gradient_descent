#include "helpers.h"

void read_csv(std::string filename,
              std::vector<std::vector<float>> &features,
              std::vector<std::vector<float>> &labels,
              int dataset_size)
{
    std::ifstream dataset(filename);
    if (!dataset.is_open())
    {
        std::cout << "Could not open file." << std::endl;
        exit(1);
    }
    std::string row, column;
    int i = 0;
    while (std::getline(dataset, row) && i < dataset_size)
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
        i++;
    }
}

void printVectorMatrix(const std::vector<std::vector<float>> a)
{
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < a[i].size(); j++)
        {
            std::cout << a[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void printFlatMatrix(const float *mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; i++)
    {
        std::cout << mat[i] << " ";
        if ((i + 1) % cols == 0)
        {
            std::cout << std::endl;
        }
    }
}

void flattenVectorMatrix(const std::vector<std::vector<float>> mat, float *flattenedMat)
{
    for (int i = 0; i < mat.size(); i++)
    {
        int cols = mat.at(i).size();
        for (int j = 0; j < cols; j++)
        {
            flattenedMat[cols * i + j] = mat[i][j];
        }
    }
}