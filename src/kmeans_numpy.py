"""
K-Means NumPy — implémentation de référence (séquentielle)
Inspiré de l'approche GroupBy → Aggregate de Terrell (2017)
traduite en NumPy vectorisé.
"""
import numpy as np


def init_centroids(X: np.ndarray, K: int, seed: int = 42) -> np.ndarray:
    """Initialisation aléatoire : tire K points parmi X."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(X), K, replace=False)
    return X[idx].copy().astype(np.float32)


def assign_numpy(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Phase ASSIGNMENT — vectorisé NumPy.
    Pour chaque point, trouve le centroïde le plus proche.

    Équivalent de GetNearestCentroid (Terrell 2017) mais vectorisé.

    X          : (n, d)
    centroids  : (K, d)
    retourne   : labels (n,) dtype int32
    """
    # Broadcasting : (n, 1, d) - (1, K, d) → (n, K, d)
    diff = X[:, None, :] - centroids[None, :, :]
    # Distance euclidienne au carré : (n, K)
    dist_sq = (diff ** 2).sum(axis=2)
    return np.argmin(dist_sq, axis=1).astype(np.int32)


def update_numpy(X: np.ndarray, labels: np.ndarray, K: int) -> np.ndarray:
    """
    Phase UPDATE — GroupBy → moyenne, séquentiel.
    Équivalent de UpdateCentroids (Terrell 2017) en NumPy.

    X       : (n, d)
    labels  : (n,)
    retourne: centroids (K, d)
    """
    d = X.shape[1]
    centroids = np.zeros((K, d), dtype=np.float32)
    for k in range(K):
        mask = labels == k
        if mask.any():
            # Aggregate : somme des points du cluster k, divisée par count
            centroids[k] = X[mask].mean(axis=0)
    return centroids


def kmeans_numpy(
    X: np.ndarray,
    K: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: int = 42,
    verbose: bool = False
) -> tuple:
    """
    K-Means complet — version NumPy séquentielle.

    Paramètres
    ----------
    X        : (n, d) données d'entrée
    K        : nombre de clusters
    max_iter : nombre maximum d'itérations
    tol      : seuil de convergence (déplacement des centroïdes)
    seed     : graine aléatoire pour la reproductibilité
    verbose  : afficher les itérations

    Retourne
    --------
    centroids : (K, d)
    labels    : (n,)
    n_iter    : nombre d'itérations effectuées
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    centroids = init_centroids(X, K, seed)

    for i in range(max_iter):
        old_centroids = centroids.copy()

        labels    = assign_numpy(X, centroids)
        centroids = update_numpy(X, labels, K)

        shift = np.linalg.norm(centroids - old_centroids)
        if verbose:
            print(f"  iter {i+1:3d} | shift = {shift:.6f}")

        if shift < tol:
            if verbose:
                print(f"  Convergence à l'itération {i+1}")
            return centroids, labels, i + 1

    return centroids, labels, max_iter
