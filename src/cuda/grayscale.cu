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
void grayscale_kernel(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * channels;
        if (channels >= 3) {
            unsigned char r = input[idx];
            unsigned char g = input[idx + 1];
            unsigned char b = input[idx + 2];
            
            unsigned char gray = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
            
            output[idx] = gray;
            output[idx + 1] = gray;
            output[idx + 2] = gray;
            if (channels == 4) {
                output[idx + 3] = input[idx + 3];
            }
        }
    }
}

void cuda_grayscale(const Image* input, Image* output, GpuTimings* timings) {
    int size = input->width * input->height * input->channels;
    unsigned char *d_input, *d_output;
    cudaEvent_t start_event, stop_event, h2d_start, h2d_stop, d2h_start, d2h_stop;
    
    CUDA_CHECK(cudaEventCreate(&start_event));
    CUDA_CHECK(cudaEventCreate(&stop_event));
    CUDA_CHECK(cudaEventCreate(&h2d_start));
    CUDA_CHECK(cudaEventCreate(&h2d_stop));
    CUDA_CHECK(cudaEventCreate(&d2h_start));
    CUDA_CHECK(cudaEventCreate(&d2h_stop));
    
    cudaEventRecord(start_event, 0);
    
    CUDA_CHECK(cudaMalloc((void**)&d_input, size));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size));
    
    cudaEventRecord(h2d_start, 0);
    CUDA_CHECK(cudaMemcpy(d_input, input->data, size, cudaMemcpyHostToDevice));
    cudaEventRecord(h2d_stop, 0);
    
    dim3 block(16, 16);
    dim3 grid((input->width + block.x - 1) / block.x, (input->height + block.y - 1) / block.y);
    
    float kernel_time = 0;
    cudaEvent_t k_start, k_stop;
    cudaEventCreate(&k_start);
    cudaEventCreate(&k_stop);
    
    cudaEventRecord(k_start, 0);
    grayscale_kernel<<<grid, block>>>(d_input, d_output, input->width, input->height, input->channels);
    cudaEventRecord(k_stop, 0);
    
    CUDA_CHECK(cudaGetLastError());
    
    cudaEventRecord(d2h_start, 0);
    CUDA_CHECK(cudaMemcpy(output->data, d_output, size, cudaMemcpyDeviceToHost));
    cudaEventRecord(d2h_stop, 0);
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    
    cudaEventRecord(stop_event, 0);
    cudaEventSynchronize(stop_event);
    cudaEventSynchronize(h2d_stop);
    cudaEventSynchronize(d2h_stop);
    cudaEventSynchronize(k_stop);
    
    float t_total, t_h2d, t_d2h;
    cudaEventElapsedTime(&t_total, start_event, stop_event);
    cudaEventElapsedTime(&t_h2d, h2d_start, h2d_stop);
    cudaEventElapsedTime(&kernel_time, k_start, k_stop);
    cudaEventElapsedTime(&t_d2h, d2h_start, d2h_stop);
    
    timings->total_ms = (double)t_total;
    timings->h2d_ms = (double)t_h2d;
    timings->kernel_ms = (double)kernel_time;
    timings->d2h_ms = (double)t_d2h;
    
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    cudaEventDestroy(h2d_start);
    cudaEventDestroy(h2d_stop);
    cudaEventDestroy(d2h_start);
    cudaEventDestroy(d2h_stop);
    cudaEventDestroy(k_start);
    cudaEventDestroy(k_stop);
}
