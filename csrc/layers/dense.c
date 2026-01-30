#include "dense.h"

void dense_forward(
    const float* x,
    const float* W,
    const float* b,
    float* y,
    size_t in_features,
    size_t out_features
) {
    for (size_t j = 0; j < out_features; j++) {
        float acc = b[j];
        const float* wrow = W + j * in_features;
        for (size_t i = 0; i < in_features; i++) {
            acc += x[i] * wrow[i];
        }
        y[j] = acc;
    }
}
