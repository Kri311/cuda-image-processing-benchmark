#include "benchmark.h"
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

static char g_log_filename[256] = "";
static bool g_initialized = false;

double get_current_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

void benchmark_init(const char* filename) {
    strncpy(g_log_filename, filename, sizeof(g_log_filename) - 1);
    FILE* f = fopen(g_log_filename, "w");
    if (f) {
        fprintf(f, "operation,width,height,cpu_ms,h2d_ms,kernel_ms,d2h_ms,gpu_total_ms,kernel_speedup,end_to_end_speedup,max_error,mean_error,correct\n");
        fclose(f);
        g_initialized = true;
    }
}

void benchmark_log_result(const BenchmarkResult* result) {
    if (!g_initialized) return;
    
    FILE* f = fopen(g_log_filename, "a");
    if (f) {
        fprintf(f, "%s,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%d,%.3f,%s\n",
                result->operation, result->width, result->height,
                result->cpu_ms, result->h2d_ms, result->kernel_ms, result->d2h_ms, result->gpu_total_ms,
                result->kernel_speedup, result->end_to_end_speedup,
                result->max_error, result->mean_error,
                result->correct ? "true" : "false");
        fclose(f);
    }
}

void benchmark_print_system_info(void) {
    printf("===========================================================\n");
    printf("CUDA IMAGE PROCESSING BENCHMARK\n");
    printf("===========================================================\n\n");

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess || deviceCount == 0) {
        printf("GPU: No CUDA capable devices found.\n\n");
        return;
    }

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("GPU:\n");
    printf("  Device             : %s\n", prop.name);
    printf("  Compute Capability : %d.%d\n", prop.major, prop.minor);
    printf("  VRAM               : %lu MB\n", prop.totalGlobalMem / (1024 * 1024));
    printf("  Shared Mem / Block : %lu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Multiprocessors    : %d\n", prop.multiProcessorCount);
    printf("  Max Threads / Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Clock Rate         : %d MHz\n", prop.clockRate / 1000);
    printf("-----------------------------------------------------------\n");
}

void benchmark_print_result(const BenchmarkResult* result) {
    printf("Operation: %s\n", result->operation);
    printf("-----------------------------------------------------------\n");
    
    printf("CPU:\n");
    printf("  Execution Time       : %.3f ms\n\n", result->cpu_ms);
    
    printf("CUDA:\n");
    printf("  H2D Transfer         : %.3f ms\n", result->h2d_ms);
    printf("  Kernel               : %.3f ms\n", result->kernel_ms);
    printf("  D2H Transfer         : %.3f ms\n", result->d2h_ms);
    printf("  Total GPU            : %.3f ms\n\n", result->gpu_total_ms);
    
    printf("Speedup:\n");
    printf("  Kernel-only          : %.2fx\n", result->kernel_speedup);
    printf("  End-to-end           : %.2fx\n\n", result->end_to_end_speedup);
    
    printf("Correctness:\n");
    if (result->correct) {
        printf("  PASS\n");
    } else {
        printf("  FAIL (Max error: %d, Mean error: %.3f)\n", result->max_error, result->mean_error);
    }
    printf("-----------------------------------------------------------\n");
}
