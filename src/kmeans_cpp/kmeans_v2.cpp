#include "kmeans_v2.h"
#include <omp.h>
#include <cstring>
#include <cfloat>
#include <cstdlib>

// ─────────────────────────────────────────────────────────────
// PHASE 1 : ASSIGNMENT — identique à v1
// Reproduite ici pour que v2 soit autonome.
// ─────────────────────────────────────────────────────────────
void assign_parallel_v2(
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
            float dist = 0.0f;
            for (int j = 0; j < d; j++) {
                float diff = X[i*d + j] - centroids[k*d + j];
                dist += diff * diff;
            }
            if (dist < best_dist) { best_dist = dist; best_k = k; }
        }
        labels[i] = best_k;
    }
}

// ─────────────────────────────────────────────────────────────
// PHASE 2 : UPDATE — parallèle avec buffers locaux
//
// Stratégie (Terrell 2017, HOGWILD! pattern) :
//   Chaque thread alloue son propre tableau local (K×d floats + K ints).
//   Il accumule SANS verrou dans son buffer privé.
//   Une section #pragma omp critical fusionne les buffers à la fin.
//
// Pourquoi pas juste #pragma omp atomic ?
//   atomic protège chaque opération individuellement mais génère
//   de la contention si beaucoup de points tombent dans le même cluster.
//   Les buffers locaux éliminent complètement la contention pendant
//   l'accumulation — le critical ne couvre que la fusion finale,
//   qui est O(T × K × d) avec T = nb threads (négligeable).
//
// Note numérique : l'ordre de fusion des buffers dépend du scheduling
//   des threads → les centroïdes peuvent différer de ~1e-6 en fp32
//   par rapport à la version séquentielle. C'est attendu et documenté
//   (cf. exercice "random order for a sum", X. Dupré).
// ─────────────────────────────────────────────────────────────
void update_parallel_v2(
    const float* X,
    const int*   labels,
    float*       centroids,
    int n, int d, int K
) {
    // Initialiser le buffer global à zéro
    memset(centroids, 0, K * d * sizeof(float));
    int* counts = (int*)calloc(K, sizeof(int));

    #pragma omp parallel
    {
        // ── Allocation du buffer local par thread ──
        float* local_sum   = (float*)calloc(K * d, sizeof(float));
        int*   local_count = (int*)calloc(K,       sizeof(int));

        // ── Accumulation locale (sans verrou) ──
        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++) {
            int k = labels[i];
            local_count[k]++;
            for (int j = 0; j < d; j++) {
                local_sum[k*d + j] += X[i*d + j];
            }
        }

        // ── Fusion dans le buffer global (un thread à la fois) ──
        // La section critical est courte : O(K × d) opérations
        #pragma omp critical
        {
            for (int k = 0; k < K; k++) {
                counts[k] += local_count[k];
                for (int j = 0; j < d; j++) {
                    centroids[k*d + j] += local_sum[k*d + j];
                }
            }
        }

        free(local_sum);
        free(local_count);
    }

    // Diviser par les counts
    for (int k = 0; k < K; k++) {
        if (counts[k] > 0) {
            for (int j = 0; j < d; j++) {
                centroids[k*d + j] /= (float)counts[k];
            }
        }
    }
    free(counts);
}
