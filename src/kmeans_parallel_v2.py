"""
K-Means C++ v2: wrapper Python.
Les DEUX phases sont parallèles :
  - Assignment : #pragma omp parallel for (identique à v1)
  - Update     : buffers locaux par thread + fusion via critical
"""
import sys
import os
import numpy as np

_BUILD_DIR = os.path.join(os.path.dirname(__file__), "kmeans_cpp", "build")
if _BUILD_DIR not in sys.path:
    sys.path.insert(0, _BUILD_DIR)

try:
    import kmeans_cpp as _cpp
    _CPP_AVAILABLE = True
except ImportError as e:
    _CPP_AVAILABLE = False
    print(f"Module C++ non disponible : {e}")


def init_centroids(X: np.ndarray, K: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), K, replace=False)
    return np.ascontiguousarray(X[idx], dtype=np.float32)


def kmeans_parallel_v2(
    X: np.ndarray,
    K: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
    n_threads: int = None,
    verbose: bool = False
) -> tuple:
    """
    K-Means C++ v2: les DEUX phases sont parallèles.

    Phase ASSIGNMENT : #pragma omp parallel for schedule(static)
    Phase UPDATE     : buffers locaux par thread → pas de race condition
                       fusion finale via #pragma omp critical

    L'ordre de fusion des buffers dépend du scheduling des threads,
    ce qui peut introduire des différences ~1e-6 en fp32 par rapport
    à la version séquentielle (non-associativité de l'addition flottante)    """
    if not _CPP_AVAILABLE:
        raise RuntimeError("Module C++ non compilé")

    if n_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(n_threads)

    X = np.ascontiguousarray(X, dtype=np.float32)
    n, d = X.shape

    centroids = init_centroids(X, K, seed)
    labels    = np.zeros(n, dtype=np.int32)

    for i in range(max_iter):
        old_centroids = centroids.copy()

        # Phase 1 : assignment parallèle (identique à v1)
        _cpp.assign_parallel_v2(X, centroids, labels, n, d, K)

        # Phase 2 : update parallèle avec buffers locaux
        new_centroids = np.ascontiguousarray(
            np.zeros((K, d), dtype=np.float32)
        )
        _cpp.update_parallel_v2(X, labels, new_centroids, n, d, K)
        centroids = new_centroids

        shift = np.linalg.norm(centroids - old_centroids)
        if verbose:
            print(f"  iter {i+1:3d} | shift = {shift:.6f}")

        if shift < tol:
            if verbose:
                print(f"  Convergence à l'itération {i+1}")
            return centroids, labels, i + 1

    return centroids, labels, max_iter
