// model.c
#include "model.h"
#include "layers/dense.h"
#include "layers/activations.h"

API void dense_relu(
    const float* x,
    const float* W,
    const float* b,
    float* y,
    size_t in_features,
    size_t out_features
) {
    dense_forward(x, W, b, y, in_features, out_features);

    for (size_t i = 0; i < out_features; i++) {
        y[i] = relu(y[i]);
    }
}
