#include "cpu_operations.h"

void cpu_brightness(const Image* input, Image* output, int brightness) {
    int size = input->width * input->height * input->channels;
    for (int i = 0; i < size; ++i) {
        // Skip alpha channel
        if (input->channels == 4 && (i % 4 == 3)) {
            output->data[i] = input->data[i];
            continue;
        }
        
        int val = input->data[i] + brightness;
        if (val > 255) val = 255;
        if (val < 0) val = 0;
        output->data[i] = (unsigned char)val;
    }
}
