#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H

#include <vector>

class LogisticRegression
{
private:
    // Variables
    std::vector<std::vector<float>> _weight;    // column vector (n_featres x 1)
    std::vector<std::vector<float>> &_features; // n_samples x n_features
    std::vector<std::vector<float>> &_labels;   // column vector (n_samples X 1)
    float _learning_rate;

    int _n_samples;
    int _n_features;

    // Functions
    std::vector<std::vector<float>> _gradient(
        std::vector<std::vector<float>> &x,
        std::vector<std::vector<float>> &y,
        std::vector<std::vector<float>> &z);

    std::vector<std::vector<float>> forward(
        std::vector<std::vector<float>> &feature_T);

    void backward(
        std::vector<std::vector<float>> &feature_T,
        std::vector<std::vector<float>> &label,
        std::vector<std::vector<float>> &output);

    // loss function
    std::vector<std::vector<float>> bce_loss(
        std::vector<std::vector<float>> &y,
        std::vector<std::vector<float>> &z);

public:
    LogisticRegression(std::vector<std::vector<float>> &features,
                       std::vector<std::vector<float>> &labels,
                       float learning_rate);

    void train();
};

#endif