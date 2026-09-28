#include "cpu_operations.h"
#include <math.h>

static int clamp(int val, int min, int max) {
    if (val < min) return min;
    if (val > max) return max;
    return val;
}

void cpu_sobel(const Image* input, Image* output) {
    int Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };
    
    int Gy[3][3] = {
        {-1, -2, -1},
        { 0,  0,  0},
        { 1,  2,  1}
    };
    
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
                
                int sumX = 0;
                int sumY = 0;
                
                for (int ky = -1; ky <= 1; ++ky) {
                    for (int kx = -1; kx <= 1; ++kx) {
                        int ny = clamp(y + ky, 0, h - 1);
                        int nx = clamp(x + kx, 0, w - 1);
                        
                        int val = input->data[(ny * w + nx) * c + ch];
                        sumX += val * Gx[ky + 1][kx + 1];
                        sumY += val * Gy[ky + 1][kx + 1];
                    }
                }
                
                int magnitude = (int)sqrt(sumX * sumX + sumY * sumY);
                output->data[(y * w + x) * c + ch] = (unsigned char)clamp(magnitude, 0, 255);
            }
        }
    }
}
