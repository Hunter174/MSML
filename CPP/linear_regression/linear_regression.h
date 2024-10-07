#ifndef LINEAR_REGRESSION_H
#define LINEAR_REGRESSION_H

#include <vector>

class LinearRegression {
public:
    // Constructor
    LinearRegression();

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

    // Helper function to calculate the mean
    double mean(const std::vector<double>& vec) const;

    // Helper function to calculate covariance
    double covariance(const std::vector<double>& X, const std::vector<double>& y) const;

    // Helper function to calculate variance
    double variance(const std::vector<double>& X) const;
};

#endif // LINEAR_REGRESSION_H