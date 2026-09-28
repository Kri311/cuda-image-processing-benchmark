#include "cpu_operations.h"

void cpu_invert(const Image* input, Image* output) {
    int size = input->width * input->height * input->channels;
    for (int i = 0; i < size; ++i) {
        if (input->channels == 4 && (i % 4 == 3)) {
            output->data[i] = input->data[i];
            continue;
        }
        
        output->data[i] = 255 - input->data[i];
    }
}
