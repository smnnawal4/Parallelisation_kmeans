#pragma once

// Phase 1 : Assignment parallèle
// Pour chaque point i, trouve le centroïde le plus proche
// Parallélisme parfait : aucune dépendance entre les points
void assign_parallel(
    const float* X,           // (n, d) row-major
    const float* centroids,   // (K, d) row-major
    int*         labels,      // (n,)   sortie
    int n, int d, int K
);

// Phase 2 : Update des centroïdes
// Calcule la moyenne des points de chaque cluster
// Update séquentiel justifié : < 0.5% du temps total (Lavenier 2001)
void update_centroids(
    const float* X,           // (n, d)
    const int*   labels,      // (n,)
    float*       centroids,   // (K, d) sortie
    int n, int d, int K
);
