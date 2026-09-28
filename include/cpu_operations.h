#pragma once

#include "image_utils.h"

#ifdef __cplusplus
extern "C" {
#endif

void cpu_grayscale(const Image* input, Image* output);
void cpu_brightness(const Image* input, Image* output, int brightness);
void cpu_invert(const Image* input, Image* output);
void cpu_gaussian_blur(const Image* input, Image* output);
void cpu_sobel(const Image* input, Image* output);

#ifdef __cplusplus
}
#endif
