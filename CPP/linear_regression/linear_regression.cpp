#include "linear_regression.h"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For std::vector bindings with pybind11

// Constructor
LinearRegression::LinearRegression(double learning_rate, int iterations)
    : slope_(0), intercept_(0), learning_rate_(learning_rate), iterations_(iterations) {}

// Fit the model to the data
void LinearRegression::fit(const std::vector<double>& X, const std::vector<double>& y) {
    size_t n = X.size();

    for (int i = 0; i < iterations_; ++i) {
        double slope_grad = 0;
        double int_grad = 0;

        // Calculate the gradient
        for (size_t j = 0; j < n; ++j) {
            double pred = slope_ * X[j] + intercept_;
            double error = pred - y[j];

            // propagate the error
            slope_grad += error * X[j];
            int_grad += error;
        }

        // Get the average error over the epoch
        slope_grad /= n;
        int_grad /= n;

        // Update the weights
        slope_ -= learning_rate_ * slope_grad;
        intercept_ -= learning_rate_ * int_grad;

        // Monitor the MSE after each epoch
        double mse = meanSquaredError(X, y);
        std::cout << "Epoch " << i + 1 << " - MSE: " << mse << std::endl;
    }
}

// Predict the target for a given input
std::vector<double> LinearRegression::predict(const std::vector<double>& X) const {
    std::vector<double> preds;
    for (const auto& x : X) {
        preds.push_back(slope_ * x + intercept_);
    }
    return preds;
}

// Get the slope
double LinearRegression::getSlope() const {
    return slope_;
}

// Get the intercept
double LinearRegression::getIntercept() const {
    return intercept_;
}

double LinearRegression::meanSquaredError(const std::vector<double>& X, const std::vector<double>& y) const {
    double mse = 0.0;
    size_t n = X.size();

    for (size_t i = 0; i < n; ++i) {
        double prediction = slope_ * X[i] + intercept_;
        mse += (prediction - y[i]) * (prediction - y[i]);
    }

    return mse / n;
}

// PYBIND11 bindings to expose C++ class and methods to Python
namespace py = pybind11;

PYBIND11_MODULE(linear_regression, m) {
    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<double, int>(), py::arg("learning_rate") = 0.01, py::arg("iterations") = 1000)  // Enable keyword args
        .def("fit", &LinearRegression::fit)
        .def("predict", &LinearRegression::predict)
        .def("getSlope", &LinearRegression::getSlope)
        .def("getIntercept", &LinearRegression::getIntercept);
}