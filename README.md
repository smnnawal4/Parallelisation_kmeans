# K-Means Parallèle sur CPU 

## Structure

```
projet_kmeans/
├── src/
│   ├── kmeans_numpy.py          # Implémentation 1 : NumPy séquentiel
│   ├── kmeans_parallel.py       # Implémentation 2 : wrapper C++ OpenMP
│   ├── kmeans_parallel_v2.py       # Implémentation 2 : wrapper C++ OpenMP+ // de la phase update
│   └── kmeans_cpp/
│       ├── kmeans.h             # Header C++
│       ├── kmeans.cpp           # Code C++ + OpenMP
│       ├── binding.cpp          # Binding pybind11
│       ├── CMakeLists.txt       # Build system
│       └── build/               # Module compilé (généré)
└── notebooks/
    ├── 01_implementations.ipynb  # Code + validation
    └── 02_benchmark.ipynb        # Benchmark + figures
```

## Installation

```bash
# 1. Compiler le module C++
cd src/kmeans_cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# 2. Installer les dépendances Python
pip install numpy scikit-learn matplotlib pandas jupyter

# 3. Lancer les notebooks
cd ../../..
jupyter notebook notebooks/
```

## Dépendances
- Python 3.10+
- numpy, scikit-learn, matplotlib, pandas
- g++ avec OpenMP (-fopenmp)
- cmake 3.15+
- pybind11
