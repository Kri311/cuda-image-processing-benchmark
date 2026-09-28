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

__constant__ int d_kernel[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};

__global__
void gaussian_blur_kernel(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        for (int c = 0; c < channels; ++c) {
            if (channels == 4 && c == 3) {
                output[(y * width + x) * channels + c] = input[(y * width + x) * channels + c];
                continue;
            }
            
            int sum = 0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    int ny = min(max(y + ky, 0), height - 1);
                    int nx = min(max(x + kx, 0), width - 1);
                    sum += input[(ny * width + nx) * channels + c] * d_kernel[ky + 1][kx + 1];
                }
            }
            output[(y * width + x) * channels + c] = (unsigned char)(sum / 16);
        }
    }
}

#define TILE_WIDTH 16
#define KERNEL_RADIUS 1
#define TILE_SIZE (TILE_WIDTH + 2 * KERNEL_RADIUS)

__global__
void gaussian_blur_shared_kernel(const unsigned char* input, unsigned char* output, int width, int height, int channels) {
    extern __shared__ unsigned char shared_mem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row_o = blockIdx.y * TILE_WIDTH + ty - KERNEL_RADIUS;
    int col_o = blockIdx.x * TILE_WIDTH + tx - KERNEL_RADIUS;
    
    int row_i = row_o;
    int col_i = col_o;
    
    for (int c = 0; c < channels; ++c) {
        if (channels == 4 && c == 3) {
            if (ty >= KERNEL_RADIUS && ty < TILE_WIDTH + KERNEL_RADIUS && 
                tx >= KERNEL_RADIUS && tx < TILE_WIDTH + KERNEL_RADIUS) {
                if (row_o < height && col_o < width) {
                    output[(row_o * width + col_o) * channels + c] = input[(row_o * width + col_o) * channels + c];
                }
            }
        } else {
            int shared_idx = (ty * TILE_SIZE + tx) * channels + c;
            if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
                shared_mem[shared_idx] = input[(row_i * width + col_i) * channels + c];
            } else {
                int clamp_row = min(max(row_i, 0), height - 1);
                int clamp_col = min(max(col_i, 0), width - 1);
                shared_mem[shared_idx] = input[(clamp_row * width + clamp_col) * channels + c];
            }
            
            __syncthreads();
            
            if (ty >= KERNEL_RADIUS && ty < TILE_WIDTH + KERNEL_RADIUS && 
                tx >= KERNEL_RADIUS && tx < TILE_WIDTH + KERNEL_RADIUS) {
                
                if (row_o < height && col_o < width) {
                    int sum = 0;
                    for (int ky = -1; ky <= 1; ++ky) {
                        for (int kx = -1; kx <= 1; ++kx) {
                            sum += shared_mem[((ty + ky) * TILE_SIZE + (tx + kx)) * channels + c] * d_kernel[ky + 1][kx + 1];
                        }
                    }
                    output[(row_o * width + col_o) * channels + c] = (unsigned char)(sum / 16);
                }
            }
            __syncthreads();
        }
    }
}

static void run_blur_kernel(const Image* input, Image* output, GpuTimings* timings, bool use_shared) {
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
    
    cudaEventRecord(k_start, 0);
    if (use_shared) {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((input->width + TILE_WIDTH - 1) / TILE_WIDTH, (input->height + TILE_WIDTH - 1) / TILE_WIDTH);
        size_t shared_mem_size = TILE_SIZE * TILE_SIZE * input->channels * sizeof(unsigned char);
        gaussian_blur_shared_kernel<<<grid, block, shared_mem_size>>>(d_input, d_output, input->width, input->height, input->channels);
    } else {
        dim3 block(16, 16);
        dim3 grid((input->width + block.x - 1) / block.x, (input->height + block.y - 1) / block.y);
        gaussian_blur_kernel<<<grid, block>>>(d_input, d_output, input->width, input->height, input->channels);
    }
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

void cuda_gaussian_blur(const Image* input, Image* output, GpuTimings* timings) {
    run_blur_kernel(input, output, timings, false);
}

void cuda_gaussian_blur_shared(const Image* input, Image* output, GpuTimings* timings) {
    run_blur_kernel(input, output, timings, true);
}
