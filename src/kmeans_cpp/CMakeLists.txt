cmake_minimum_required(VERSION 3.15)
project(kmeans_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP
find_package(OpenMP REQUIRED)

# Python + pybind11
find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Récupérer pybind11 depuis l'installation pip
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 REQUIRED)

# Module Python
pybind11_add_module(kmeans_cpp
    kmeans.cpp
    kmeans_v2.cpp
    binding.cpp
)

target_link_libraries(kmeans_cpp PRIVATE OpenMP::OpenMP_CXX)

# Optimisations
target_compile_options(kmeans_cpp PRIVATE
    -O3
    -march=native
    -ffast-math
)