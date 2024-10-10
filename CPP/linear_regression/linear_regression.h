// Header file linear_regression.h
#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>

class LinearRegression {
public:
    // Constructor with learning rate and iterations
    LinearRegression(double learning_rate = 0.01, int iterations = 1000);

    // Fit the model to the data
    void fit(const std::vector<double>& X, const std::vector<double>& y);

    // Predict the target for a given input
    std::vector<double> predict(const std::vector<double>& X) const;

    // Get the coefficients
    double getSlope() const;
    double getIntercept() const;

private:
    // Coefficients for the linear regression line
    double slope_;
    double intercept_;

    double learning_rate_;
    int iterations_;

    double meanSquaredError(const std::vector<double>& X, const std::vector<double>& y) const;
};

#endif // LINEAR_REGRESSION_H