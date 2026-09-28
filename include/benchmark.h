#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    char operation[64];
    int width;
    int height;
    
    // CPU timings
    double cpu_ms;
    
    // GPU timings
    double h2d_ms;
    double kernel_ms;
    double d2h_ms;
    double gpu_total_ms;
    
    // Speedup
    double kernel_speedup;
    double end_to_end_speedup;
    
    // Correctness
    int max_error;
    double mean_error;
    bool correct;
} BenchmarkResult;

void benchmark_init(const char* filename);
void benchmark_log_result(const BenchmarkResult* result);
void benchmark_print_system_info(void);
void benchmark_print_result(const BenchmarkResult* result);

/* C Timer utility */
double get_current_time_ms(void);

#ifdef __cplusplus
}
#endif
