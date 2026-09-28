#include "cpu_operations.h"

void cpu_grayscale(const Image* input, Image* output) {
    int size = input->width * input->height;
    for (int i = 0; i < size; ++i) {
        int idx = i * input->channels;
        int out_idx = i * output->channels;
        
        if (input->channels >= 3) {
            unsigned char r = input->data[idx];
            unsigned char g = input->data[idx + 1];
            unsigned char b = input->data[idx + 2];
            
            unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
            
            output->data[out_idx] = gray;
            if (output->channels > 1) {
                output->data[out_idx + 1] = gray;
                output->data[out_idx + 2] = gray;
            }
            if (input->channels == 4 && output->channels == 4) {
                output->data[out_idx + 3] = input->data[idx + 3];
            }
        }
    }
}
