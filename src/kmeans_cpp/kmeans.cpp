#include "kmeans.h"
#include <omp.h>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <cstdlib>

// ─────────────────────────────────────────────────────────────
// PHASE 1 : ASSIGNMENT — parallélisme parfait
//
// Chaque point i est indépendant : on peut distribuer les n points
// entre les threads sans aucune synchronisation.
// #pragma omp parallel for schedule(static) : charge uniforme,
// chaque point fait exactement K calculs de distance → static optimal.
// ─────────────────────────────────────────────────────────────
void assign_parallel(
    const float* X,
    const float* centroids,
    int*         labels,
    int n, int d, int K
) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        float best_dist = FLT_MAX;
        int   best_k    = 0;

        for (int k = 0; k < K; k++) {
            // Distance euclidienne au carré (pas besoin de sqrt)
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = X[i*d + j] - centroids[k*d + j];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best_k    = k;
            }
        }
        labels[i] = best_k;
    }
}

// ─────────────────────────────────────────────────────────────
// PHASE 2 : UPDATE — séquentiel
//
// Justification : Lavenier (2001) montre que la phase update
// représente < 0.5% du temps total. On ne cherche pas à l'optimiser.
// La somme est faite séquentiellement pour éviter les race conditions
// sans complexité supplémentaire.
// ─────────────────────────────────────────────────────────────
void update_centroids(
    const float* X,
    const int*   labels,
    float*       centroids,
    int n, int d, int K
) {
    // Initialiser à zéro
    memset(centroids, 0, K * d * sizeof(float));
    int* counts = (int*)calloc(K, sizeof(int));

    // Accumulation séquentielle : pas de race condition
    for (int i = 0; i < n; i++) {
        int k = labels[i];
        counts[k]++;
        for (int j = 0; j < d; j++) {
            centroids[k*d + j] += X[i*d + j];
        }
    }

    // Diviser par les counts pour obtenir les moyennes
    for (int k = 0; k < K; k++) {
        if (counts[k] > 0) {
            for (int j = 0; j < d; j++) {
                centroids[k*d + j] /= (float)counts[k];
            }
        }
        // Si un cluster est vide : centroïde reste à 0
        // (cas rare, géré côté Python si besoin)
    }

    free(counts);
}
