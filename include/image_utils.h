#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    unsigned char* data;
    int width;
    int height;
    int channels;
} Image;

/* Initialize an empty image (allocates memory) */
void init_image(Image* img, int w, int h, int c);

/* Free image data */
void free_image(Image* img);

/* Load an image from file */
bool load_image(const char* filename, Image* img);

/* Save an image to file */
bool save_image(const char* filename, const Image* img);

/* Generate a synthetic test image if none provided */
bool generate_synthetic_image(const char* filename, int width, int height);

/* Compare two images (CPU vs GPU results) */
typedef struct {
    bool passed;
    int max_error;
    double mean_error;
    int mismatched_pixels;
} CompareResult;

CompareResult compare_images(const Image* img1, const Image* img2, int tolerance);

#ifdef __cplusplus
}
#endif
