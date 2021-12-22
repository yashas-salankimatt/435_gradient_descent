/******************************************************************************
* FILE: mpi_logistic_regression.cpp
* DESCRIPTION:  
*   MPI Logistic Regression
*   This code performs logistic regression on matricies.
******************************************************************************/
#include "mpi_logistic_regression.h"
#include <math.h> /* exp */

// Matrix multiplication C = AB
template <class T>
std::vector<std::vector<T>> Multiply(
    std::vector<std::vector<T>> &a,
    std::vector<std::vector<T>> &b)
{
    const int n = a.size();    // a rows
    const int m = a[0].size(); // a cols
    const int p = b[0].size(); // b cols

    std::vector<std::vector<T>> c(n, std::vector<T>(p, 0));
    for (auto j = 0; j < p; ++j)
    {
        for (auto k = 0; k < m; ++k)
        {
            for (auto i = 0; i < n; ++i)
            {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    return c;
}

// Matrix subtraction C = A - B
template <class T>
std::vector<std::vector<T>> Subtract(
    std::vector<std::vector<T>> &a,
    std::vector<std::vector<T>> &b)
{
    const int m = a.size();
    const int n = a[0].size();

    std::vector<std::vector<T>> c(m, std::vector<T>(n));
    for (auto i = 0; i < n; i++)
    {
        for (auto j = 0; j < n; j++)
        {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    return c;
}

// Matrix addition C = A + B
template <class T>
std::vector<std::vector<T>> Add(
    std::vector<std::vector<T>> &a,
    std::vector<std::vector<T>> &b)
{
    const int m = a.size();
    const int n = a[0].size();

    std::vector<std::vector<T>> c(m, std::vector<T>(n));
    for (auto i = 0; i < n; i++)
    {
        for (auto j = 0; j < n; j++)
        {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    return c;
}

// Transpose A -> A^T
template <class T>
std::vector<std::vector<T>> Transpose(
    std::vector<std::vector<T>> &a)
{
    int n = a.size();
    int m = a[0].size();
    std::vector<std::vector<T>> a_T(m, std::vector<T>(n));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            a_T[j][i] = a[i][j];
        }
    }
    return a_T;
}

// Sigmoid function: f(x) = 1/(1 + e^-x)
template <class T>
std::vector<std::vector<T>> sigmoid(std::vector<std::vector<T>> &z)
{
    std::vector<std::vector<T>> output(z.size(), std::vector<T>(z[0].size()));

    for (int i = 0; i < z.size(); i++)
    {
        for (int j = 0; j < z[0].size(); j++)
        {
            output[i][j] = 1.0 / (1.0 + exp(-z[i][j]));
        }
    }
    return output;
}

// Scalar multiplies matrix
template <class T>
std::vector<std::vector<T>> SclarMultiply(
    T scalar, std::vector<std::vector<T>> &x)
{

    std::vector<std::vector<T>> res(x.size(), std::vector<T>(x[0].size()));
    for (int i = 0; i < x.size(); i++)
    {
        for (int j = 0; j < x[0].size(); j++)
        {
            res[i][j] = scalar * x[i][j];
        }
    }
    return res;
}

// matrix subtracts from scalar
template <class T>
std::vector<std::vector<T>> SclarSubtract(
    T scalar, std::vector<std::vector<T>> &x)
{

    std::vector<std::vector<T>> res(x.size(), std::vector<T>(x[0].size()));
    for (int i = 0; i < x.size(); i++)
    {
        for (int j = 0; j < x[0].size(); j++)
        {
            res[i][j] = scalar - x[i][j];
        }
    }
    return res;
}

// matrix adds to scalar
template <class T>
std::vector<std::vector<T>> SclarAdd(
    T scalar, std::vector<std::vector<T>> &x)
{

    std::vector<std::vector<T>> res(x.size(), std::vector<T>(x[0].size()));
    for (int i = 0; i < x.size(); i++)
    {
        for (int j = 0; j < x[0].size(); j++)
        {
            res[i][j] = scalar + x[i][j];
        }
    }
    return res;
}

// Log of matrix values
template <class T>
std::vector<std::vector<T>> MatLog(std::vector<std::vector<T>> &x)
{

    std::vector<std::vector<T>> res(x.size(), std::vector<T>(x[0].size()));
    for (int i = 0; i < x.size(); i++)
    {
        for (int j = 0; j < x[0].size(); j++)
        {
            res[i][j] = log(x[i][j]);
        }
    }
    return res;
}

// For debug/visualization purposes
template <class T>
void printMatrix(const std::vector<std::vector<T>> &a)
{
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < a[0].size(); j++)
        {
            std::cout << a[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// LogisticRegression class definition
LogisticRegression::LogisticRegression(
    std::vector<std::vector<float>> &features,
    std::vector<std::vector<float>> &labels,
    float learning_rate,
    int rank,
    std::vector<std::vector<float>> weights) : _features(features), _labels(labels), _learning_rate(learning_rate), _rank(rank),
                                               _n_samples(features.size()), _n_features(features[0].size()),
                                               _weight(weights)
{
}

std::vector<std::vector<float>> LogisticRegression::_gradient(
    std::vector<std::vector<float>> &x,
    std::vector<std::vector<float>> &y,
    std::vector<std::vector<float>> &z)
{
    // grad U = X^T (z - y)
    // where z = prediction,
    // and y = label
    auto delta = Subtract(z, y);
    return Multiply(x, delta);
}

std::vector<std::vector<float>> LogisticRegression::forward(
    std::vector<std::vector<float>> &feature_T)
{ // output is the prediction
    // h = (W^T) X
    auto _weight_T = Transpose(_weight);
    auto h = Multiply(_weight_T, feature_T);
    auto z = sigmoid(h);
    return z;
}

void LogisticRegression::backward(
    std::vector<std::vector<float>> &feature_T,
    std::vector<std::vector<float>> &label,
    std::vector<std::vector<float>> &output)
{

    // update _weight vector
    // W(t) = W(t-1) - alpha * (grad)
    auto grad = _gradient(feature_T, label, output);
    auto delta = SclarMultiply(_learning_rate, grad);
    _weight = Subtract(_weight, delta);
}

std::vector<std::vector<float>> LogisticRegression::bce_loss(
    std::vector<std::vector<float>> &y,
    std::vector<std::vector<float>> &z)
{
    // Avoid log of 0.
    float num = 1e-30;

    auto add_num_z = SclarAdd(num, z);
    auto log_y_anz = MatLog(add_num_z);
    auto sub_one_anz = SclarSubtract(1.0f, add_num_z);
    auto sub_one_y = SclarSubtract(1.0f, y);
    auto log_oanz = MatLog(sub_one_anz);
    auto mult_y_lyanz = Multiply(y, log_y_anz);
    auto mult_soy_loanz = Multiply(sub_one_y, log_oanz);
    auto add_ylyanz_sloanz = Add(mult_y_lyanz, mult_soy_loanz);
    return SclarMultiply(-1.0f, (add_ylyanz_sloanz));
}

std::vector<std::vector<float>> LogisticRegression::train(std::vector<std::vector<float>> weights)
{
    _weight = weights;
    const int m = _features.size();
    const int n = _features[0].size();
    std::vector<std::vector<float>> running_loss(m, std::vector<float>(n));
    for (int i = 0; i < _n_samples; i++)
    {
        std::vector<std::vector<float>> feature = {_features[i]}; // row vector
        std::vector<std::vector<float>> label = {_labels[i]};     // single value / classifier

        auto feature_T = Transpose(feature);
        auto output = forward(feature_T); // prediction
        auto loss = bce_loss(label, output);
        running_loss = Add(loss, running_loss);
        if (DEBUG)
        {
            std::cout << "Loss for rank = " << _rank << std::endl;
            printMatrix(running_loss);
        }
        backward(feature_T, label, output);
    }
    if (DEBUG_FINE && _rank == 2)
    {
        std::cout << "Final loss value for rank " << _rank << " = " << running_loss[0][0] << std::endl;
    }
    return _weight;
}
