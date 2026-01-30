#ifndef DENSE_H
#define DENSE_H

#include <stddef.h>

void dense_forward(
    const float* x,
    const float* W,
    const float* b,
    float* y,
    size_t in_features,
    size_t out_features
);

#endif
