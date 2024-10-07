#include "linear_regression.h"
#include <numeric> // For std::accumulate
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For std::vector bindings with pybind11

// Constructor
LinearRegression::LinearRegression() : slope_(0), intercept_(0) {}

// Fit the model to the data
void LinearRegression::fit(const std::vector<double>& X, const std::vector<double>& y) {
  // Get the means
  double X_mean = mean(X);
  double y_mean = mean(y);

  // Calculate slope and intercept
  slope_ = covariance(X, y) / variance(X);
  intercept_ = y_mean - slope_ * X_mean;
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

// Calculate the mean
double LinearRegression::mean(const std::vector<double>& vec) const {
  double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
  return sum / vec.size();
}

// Helper function to calculate covariance
double LinearRegression::covariance(const std::vector<double>& X, const std::vector<double>& y) const {
  double X_mean = mean(X);
  double y_mean = mean(y);
  double cov = 0.0;

  for (std::size_t i = 0; i < X.size(); ++i) {
    cov += (X[i] - X_mean) * (y[i] - y_mean);
  }

  return cov / X.size();
}

// Helper function to calculate variance
double LinearRegression::variance(const std::vector<double>& X) const {
  double X_mean = mean(X);
  double var = 0.0;

  for (const auto& x : X) {
    var += (x - X_mean) * (x - X_mean);
  }

  return var / X.size();
}

// PYBIND11 bindings to expose C++ class and methods to Python
namespace py = pybind11;

PYBIND11_MODULE(linear_regression, m) {
    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<>())  // Constructor binding
        .def("fit", &LinearRegression::fit)  // Binding fit method
        .def("predict", &LinearRegression::predict)  // Binding predict method
        .def("getSlope", &LinearRegression::getSlope)  // Binding getSlope method
        .def("getIntercept", &LinearRegression::getIntercept);  // Binding getIntercept method
}
