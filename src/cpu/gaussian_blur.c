#include "cpu_operations.h"

static int clamp(int val, int min, int max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

void cpu_gaussian_blur(const Image* input, Image* output) {
    int kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    int kernel_sum = 16;
    
    int w = input->width;
    int h = input->height;
    int c = input->channels;
    
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int ch = 0; ch < c; ++ch) {
                if (c == 4 && ch == 3) {
                    output->data[(y * w + x) * c + ch] = input->data[(y * w + x) * c + ch];
                    continue;
                }
                
                int sum = 0;
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int ny = clamp(y + ky, 0, h - 1);
                        int nx = clamp(x + kx, 0, w - 1);
                        
                        sum += input->data[(ny * w + nx) * c + ch] * kernel[ky + 1][kx + 1];
                    }
                }
                output->data[(y * w + x) * c + ch] = (unsigned char)(sum / kernel_sum);
            }
        }
    }
}
