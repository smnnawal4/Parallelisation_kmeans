#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kmeans.h"

namespace py = pybind11;

PYBIND11_MODULE(kmeans_cpp, m) {
    m.doc() = "K-Means parallèle avec OpenMP — projet teachcompute";

    m.def("assign_parallel",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> X,
           py::array_t<float, py::array::c_style | py::array::forcecast> centroids,
           py::array_t<int,   py::array::c_style | py::array::forcecast> labels,
           int n, int d, int K)
        {
            assign_parallel(
                X.data(),
                centroids.data(),
                labels.mutable_data(),
                n, d, K
            );
        },
        py::arg("X"), py::arg("centroids"), py::arg("labels"),
        py::arg("n"), py::arg("d"), py::arg("K"),
        "Phase assignment : assigne chaque point au centroïde le plus proche (OpenMP)"
    );

    m.def("update_centroids",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> X,
           py::array_t<int,   py::array::c_style | py::array::forcecast> labels,
           py::array_t<float, py::array::c_style | py::array::forcecast> centroids,
           int n, int d, int K)
        {
            update_centroids(
                X.data(),
                labels.data(),
                centroids.mutable_data(),
                n, d, K
            );
        },
        py::arg("X"), py::arg("labels"), py::arg("centroids"),
        py::arg("n"), py::arg("d"), py::arg("K"),
        "Phase update : recalcule les centroïdes (séquentiel)"
    );
}
