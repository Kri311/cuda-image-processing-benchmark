#pragma once

#include "image_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

// Struct to hold GPU execution times
typedef struct {
    double h2d_ms;
    double kernel_ms;
    double d2h_ms;
    double total_ms;
} GpuTimings;

// CUDA Implementations
void cuda_grayscale(const Image* input, Image* output, GpuTimings* timings);
void cuda_brightness(const Image* input, Image* output, int brightness, GpuTimings* timings);
void cuda_invert(const Image* input, Image* output, GpuTimings* timings);

// Global and Shared Memory implementations
void cuda_gaussian_blur(const Image* input, Image* output, GpuTimings* timings);
void cuda_gaussian_blur_shared(const Image* input, Image* output, GpuTimings* timings);

void cuda_sobel(const Image* input, Image* output, GpuTimings* timings);
void cuda_sobel_shared(const Image* input, Image* output, GpuTimings* timings);

#ifdef __cplusplus
}
#endif
