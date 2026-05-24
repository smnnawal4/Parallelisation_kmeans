#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kmeans.h"
#include "kmeans_v2.h"

namespace py = pybind11;

PYBIND11_MODULE(kmeans_cpp, m) {
    m.doc() = "K-Means parallèle avec OpenMP — projet teachcompute";

    // ── VERSION 1 : update séquentiel ──
    m.def("assign_parallel",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> X,
           py::array_t<float, py::array::c_style | py::array::forcecast> centroids,
           py::array_t<int,   py::array::c_style | py::array::forcecast> labels,
           int n, int d, int K) {
            assign_parallel(X.data(), centroids.data(),
                            labels.mutable_data(), n, d, K);
        },
        py::arg("X"), py::arg("centroids"), py::arg("labels"),
        py::arg("n"), py::arg("d"), py::arg("K"),
        "Phase assignment v1 : OpenMP parallel for"
    );

    m.def("update_centroids",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> X,
           py::array_t<int,   py::array::c_style | py::array::forcecast> labels,
           py::array_t<float, py::array::c_style | py::array::forcecast> centroids,
           int n, int d, int K) {
            update_centroids(X.data(), labels.data(),
                             centroids.mutable_data(), n, d, K);
        },
        py::arg("X"), py::arg("labels"), py::arg("centroids"),
        py::arg("n"), py::arg("d"), py::arg("K"),
        "Phase update v1 : sequentiel"
    );

    // ── VERSION 2 : les deux phases parallèles ──
    m.def("assign_parallel_v2",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> X,
           py::array_t<float, py::array::c_style | py::array::forcecast> centroids,
           py::array_t<int,   py::array::c_style | py::array::forcecast> labels,
           int n, int d, int K) {
            assign_parallel_v2(X.data(), centroids.data(),
                               labels.mutable_data(), n, d, K);
        },
        py::arg("X"), py::arg("centroids"), py::arg("labels"),
        py::arg("n"), py::arg("d"), py::arg("K"),
        "Phase assignment v2 : identique a v1"
    );

    m.def("update_parallel_v2",
        [](py::array_t<float, py::array::c_style | py::array::forcecast> X,
           py::array_t<int,   py::array::c_style | py::array::forcecast> labels,
           py::array_t<float, py::array::c_style | py::array::forcecast> centroids,
           int n, int d, int K) {
            update_parallel_v2(X.data(), labels.data(),
                               centroids.mutable_data(), n, d, K);
        },
        py::arg("X"), py::arg("labels"), py::arg("centroids"),
        py::arg("n"), py::arg("d"), py::arg("K"),
        "Phase update v2 : buffers locaux par thread, fusion via critical"
    );
}