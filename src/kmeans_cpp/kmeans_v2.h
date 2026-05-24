#pragma once

// Version 2 : les DEUX phases sont parallèles
// Phase 1 (assignment) : identique à v1
// Phase 2 (update)     : buffers locaux par thread → pas de race condition

void assign_parallel_v2(
    const float* X,
    const float* centroids,
    int*         labels,
    int n, int d, int K
);

void update_parallel_v2(
    const float* X,
    const int*   labels,
    float*       centroids,
    int n, int d, int K
);
