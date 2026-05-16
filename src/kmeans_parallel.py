"""
K-Means C++ parallèle — wrapper Python.
Utilise le module compilé avec OpenMP pour la phase assignment.
"""
import sys
import os
import numpy as np

# Ajouter le répertoire du module compilé au path
_BUILD_DIR = os.path.join(os.path.dirname(__file__), "kmeans_cpp", "build")
if _BUILD_DIR not in sys.path:
    sys.path.insert(0, _BUILD_DIR)

try:
    import kmeans_cpp as _cpp
    _CPP_AVAILABLE = True
except ImportError as e:
    _CPP_AVAILABLE = False
    print(f"⚠️  Module C++ non disponible : {e}")
    print("   Compiler avec : cd src/kmeans_cpp/build && cmake .. && make")


def init_centroids(X: np.ndarray, K: int, seed: int = 42) -> np.ndarray:
    """Même initialisation que la version NumPy — même seed → mêmes centroïdes initiaux."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), K, replace=False)
    return np.ascontiguousarray(X[idx], dtype=np.float32)


def kmeans_parallel(
    X: np.ndarray,
    K: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
    n_threads: int = None,
    verbose: bool = False
) -> tuple:
    """
    K-Means complet — version C++ + OpenMP.

    Phase ASSIGNMENT : parallèle via #pragma omp parallel for
    Phase UPDATE     : séquentielle (< 0.5% du temps, Lavenier 2001)

    Paramètres
    ----------
    X         : (n, d) données d'entrée
    K         : nombre de clusters
    max_iter  : nombre maximum d'itérations
    tol       : seuil de convergence
    seed      : même seed que kmeans_numpy pour comparaison équitable
    n_threads : nombre de threads OpenMP (None = tous les cœurs dispo)
    verbose   : afficher les itérations

    Retourne
    --------
    centroids : (K, d)
    labels    : (n,)
    n_iter    : nombre d'itérations effectuées
    """
    if not _CPP_AVAILABLE:
        raise RuntimeError("Module C++ non compilé")

    # Configurer le nombre de threads OpenMP
    if n_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(n_threads)

    # Forcer float32 et C-contiguous (requis par pybind11)
    X = np.ascontiguousarray(X, dtype=np.float32)
    n, d = X.shape

    centroids = init_centroids(X, K, seed)
    labels    = np.zeros(n, dtype=np.int32)

    for i in range(max_iter):
        old_centroids = centroids.copy()

        # Phase 1 : assignment parallèle (C++ OpenMP)
        _cpp.assign_parallel(X, centroids, labels, n, d, K)

        # Phase 2 : update séquentiel (C++)
        new_centroids = np.ascontiguousarray(
            np.zeros((K, d), dtype=np.float32)
        )
        _cpp.update_centroids(X, labels, new_centroids, n, d, K)
        centroids = new_centroids

        shift = np.linalg.norm(centroids - old_centroids)
        if verbose:
            print(f"  iter {i+1:3d} | shift = {shift:.6f}")

        if shift < tol:
            if verbose:
                print(f"  Convergence à l'itération {i+1}")
            return centroids, labels, i + 1

    return centroids, labels, max_iter
