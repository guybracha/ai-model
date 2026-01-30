#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>

#ifdef _WIN32
  #define API __declspec(dllexport)
#else
  #define API
#endif

#ifdef __cplusplus
extern "C" {
#endif

API void dense_relu(
    const float* x,
    const float* W,
    const float* b,
    float* y,
    size_t in_features,
    size_t out_features
);

#ifdef __cplusplus
}
#endif

#endif
