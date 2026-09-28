#include "image_utils.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

void init_image(Image* img, int w, int h, int c) {
    img->width = w;
    img->height = h;
    img->channels = c;
    img->data = (unsigned char*)malloc(w * h * c);
}

void free_image(Image* img) {
    if (img && img->data) {
        free(img->data);
        img->data = NULL;
    }
}

bool load_image(const char* filename, Image* img) {
    img->data = stbi_load(filename, &img->width, &img->height, &img->channels, 0);
    if (!img->data) {
        fprintf(stderr, "Failed to load image: %s\n", filename);
        return false;
    }
    return true;
}

bool save_image(const char* filename, const Image* img) {
    int success = 0;
    if (strstr(filename, ".jpg") || strstr(filename, ".jpeg")) {
        success = stbi_write_jpg(filename, img->width, img->height, img->channels, img->data, 100);
    } else {
        success = stbi_write_png(filename, img->width, img->height, img->channels, img->data, img->width * img->channels);
    }
    
    if (!success) {
        fprintf(stderr, "Failed to save image: %s\n", filename);
        return false;
    }
    return true;
}

bool generate_synthetic_image(const char* filename, int width, int height) {
    Image img;
    init_image(&img, width, height, 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = (y * width + x) * 3;
            img.data[idx] = (unsigned char)(255.0 * x / width);       // R
            img.data[idx+1] = (unsigned char)(255.0 * y / height);    // G
            img.data[idx+2] = (unsigned char)(128);                   // B
        }
    }
    bool success = save_image(filename, &img);
    free_image(&img);
    return success;
}

CompareResult compare_images(const Image* img1, const Image* img2, int tolerance) {
    CompareResult res = {true, 0, 0.0, 0};
    
    if (img1->width != img2->width || img1->height != img2->height || img1->channels != img2->channels) {
        res.passed = false;
        return res;
    }
    
    long long total_error = 0;
    int size = img1->width * img1->height * img1->channels;
    
    for (int i = 0; i < size; ++i) {
        int err = abs((int)img1->data[i] - (int)img2->data[i]);
        if (err > res.max_error) {
            res.max_error = err;
        }
        total_error += err;
        if (err > tolerance) {
            res.mismatched_pixels++;
            res.passed = false;
        }
    }
    
    res.mean_error = (double)total_error / size;
    return res;
}
