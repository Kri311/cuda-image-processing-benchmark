#include "cuda_operations.cuh"
#include <stdio.h>

#define CUDA_CHECK(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

__global__
void invert_kernel(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            if (channels == 4 && c == 3) {
                output[(y * width + x) * channels + c] = input[(y * width + x) * channels + c];
                continue;
            }
            int idx = (y * width + x) * channels + c;
            output[idx] = 255 - input[idx];
        }
    }
}

void cuda_invert(const Image* input, Image* output, GpuTimings* timings) {
    int size = input->width * input->height * input->channels;
    unsigned char *d_input, *d_output;
    cudaEvent_t start_event, stop_event, h2d_start, h2d_stop, d2h_start, d2h_stop, k_start, k_stop;
    
    cudaEventCreate(&start_event); cudaEventCreate(&stop_event);
    cudaEventCreate(&h2d_start); cudaEventCreate(&h2d_stop);
    cudaEventCreate(&d2h_start); cudaEventCreate(&d2h_stop);
    cudaEventCreate(&k_start); cudaEventCreate(&k_stop);
    
    cudaEventRecord(start_event, 0);
    CUDA_CHECK(cudaMalloc((void**)&d_input, size));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size));
    
    cudaEventRecord(h2d_start, 0);
    CUDA_CHECK(cudaMemcpy(d_input, input->data, size, cudaMemcpyHostToDevice));
    cudaEventRecord(h2d_stop, 0);
    
    dim3 block(16, 16);
    dim3 grid((input->width + block.x - 1) / block.x, (input->height + block.y - 1) / block.y);
    
    cudaEventRecord(k_start, 0);
    invert_kernel<<<grid, block>>>(d_input, d_output, input->width, input->height, input->channels);
    cudaEventRecord(k_stop, 0);
    
    CUDA_CHECK(cudaGetLastError());
    
    cudaEventRecord(d2h_start, 0);
    CUDA_CHECK(cudaMemcpy(output->data, d_output, size, cudaMemcpyDeviceToHost));
    cudaEventRecord(d2h_stop, 0);
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    
    float t_total, t_h2d, t_d2h, t_k;
    cudaEventElapsedTime(&t_total, start_event, stop_event);
    cudaEventElapsedTime(&t_h2d, h2d_start, h2d_stop);
    cudaEventElapsedTime(&t_k, k_start, k_stop);
    cudaEventElapsedTime(&t_d2h, d2h_start, d2h_stop);
    
    timings->total_ms = (double)t_total;
    timings->h2d_ms = (double)t_h2d;
    timings->kernel_ms = (double)t_k;
    timings->d2h_ms = (double)t_d2h;
    
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
    cudaEventDestroy(h2d_start); cudaEventDestroy(h2d_stop);
    cudaEventDestroy(d2h_start); cudaEventDestroy(d2h_stop);
    cudaEventDestroy(k_start); cudaEventDestroy(k_stop);
}
