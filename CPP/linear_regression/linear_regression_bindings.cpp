#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "LinearRegression.h"

namespace py = pybind11;

PYBIND11_MODULE(linear_regression, m) {
    py::class_<LinearRegression>(m, "LinearRegression")
        .def(py::init<>())  // Constructor binding
        .def("fit", &LinearRegression::fit)  // Binding fit method
        .def("predict", &LinearRegression::predict)  // Binding predict method
        .def("getSlope", &LinearRegression::getSlope)  // Binding getSlope method
        .def("getIntercept", &LinearRegression::getIntercept);  // Binding getIntercept method
}

