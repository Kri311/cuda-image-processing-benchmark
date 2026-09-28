#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "image_utils.h"
#include "benchmark.h"
#include "cpu_operations.h"
#include "cuda_operations.cuh"

void print_usage(const char* progName) {
    printf("Usage: %s --input <image_path> --operation <op> [options]\n", progName);
    printf("Operations: grayscale, brightness, invert, blur, sobel, all\n");
    printf("Options:\n");
    printf("  --value <int>      : Value for operations like brightness (default 50)\n");
    printf("  --iterations <int> : Number of iterations for benchmarking (default 10)\n");
}

void run_benchmark(const char* op_name, const Image* input, Image* cpu_out, Image* gpu_out, Image* gpu_shared_out, int value, int iterations) {
    BenchmarkResult result;
    memset(&result, 0, sizeof(BenchmarkResult));
    strncpy(result.operation, op_name, sizeof(result.operation) - 1);
    result.width = input->width;
    result.height = input->height;
    
    double total_cpu = 0;
    double total_h2d = 0;
    double total_kernel = 0;
    double total_d2h = 0;
    double total_gpu = 0;
    
    // Warm-up
    GpuTimings warmup_timing = {0};
    if (strcmp(op_name, "grayscale") == 0) cuda_grayscale(input, gpu_out, &warmup_timing);
    else if (strcmp(op_name, "brightness") == 0) cuda_brightness(input, gpu_out, value, &warmup_timing);
    else if (strcmp(op_name, "invert") == 0) cuda_invert(input, gpu_out, &warmup_timing);
    else if (strcmp(op_name, "blur") == 0) cuda_gaussian_blur(input, gpu_out, &warmup_timing);
    else if (strcmp(op_name, "sobel") == 0) cuda_sobel(input, gpu_out, &warmup_timing);
    
    for (int i = 0; i < iterations; ++i) {
        // CPU
        double cpu_start = get_current_time_ms();
        if (strcmp(op_name, "grayscale") == 0) cpu_grayscale(input, cpu_out);
        else if (strcmp(op_name, "brightness") == 0) cpu_brightness(input, cpu_out, value);
        else if (strcmp(op_name, "invert") == 0) cpu_invert(input, cpu_out);
        else if (strcmp(op_name, "blur") == 0) cpu_gaussian_blur(input, cpu_out);
        else if (strcmp(op_name, "sobel") == 0) cpu_sobel(input, cpu_out);
        double cpu_end = get_current_time_ms();
        total_cpu += (cpu_end - cpu_start);
        
        // GPU
        GpuTimings timing = {0};
        if (strcmp(op_name, "grayscale") == 0) cuda_grayscale(input, gpu_out, &timing);
        else if (strcmp(op_name, "brightness") == 0) cuda_brightness(input, gpu_out, value, &timing);
        else if (strcmp(op_name, "invert") == 0) cuda_invert(input, gpu_out, &timing);
        else if (strcmp(op_name, "blur") == 0) cuda_gaussian_blur(input, gpu_out, &timing);
        else if (strcmp(op_name, "sobel") == 0) cuda_sobel(input, gpu_out, &timing);
        
        total_h2d += timing.h2d_ms;
        total_kernel += timing.kernel_ms;
        total_d2h += timing.d2h_ms;
        total_gpu += timing.total_ms;
    }
    
    result.cpu_ms = total_cpu / iterations;
    result.h2d_ms = total_h2d / iterations;
    result.kernel_ms = total_kernel / iterations;
    result.d2h_ms = total_d2h / iterations;
    result.gpu_total_ms = total_gpu / iterations;
    
    result.kernel_speedup = result.cpu_ms / result.kernel_ms;
    result.end_to_end_speedup = result.cpu_ms / result.gpu_total_ms;
    
    CompareResult comp = compare_images(cpu_out, gpu_out, 2);
    result.correct = comp.passed;
    result.max_error = comp.max_error;
    result.mean_error = comp.mean_error;
    
    benchmark_print_result(&result);
    benchmark_log_result(&result);
    
    char out_filename[256];
    snprintf(out_filename, sizeof(out_filename), "../images/output/%s_cpu.png", op_name);
    save_image(out_filename, cpu_out);
    snprintf(out_filename, sizeof(out_filename), "../images/output/%s_cuda.png", op_name);
    save_image(out_filename, gpu_out);

    // Shared Memory Benchmark for Blur and Sobel
    if (strcmp(op_name, "blur") == 0 || strcmp(op_name, "sobel") == 0) {
        BenchmarkResult shared_res = result;
        snprintf(shared_res.operation, sizeof(shared_res.operation), "%s_shared", op_name);
        
        // Warm-up
        if (strcmp(op_name, "blur") == 0) cuda_gaussian_blur_shared(input, gpu_shared_out, &warmup_timing);
        else if (strcmp(op_name, "sobel") == 0) cuda_sobel_shared(input, gpu_shared_out, &warmup_timing);
        
        total_h2d = 0; total_kernel = 0; total_d2h = 0; total_gpu = 0;
        
        for (int i = 0; i < iterations; ++i) {
            GpuTimings timing = {0};
            if (strcmp(op_name, "blur") == 0) cuda_gaussian_blur_shared(input, gpu_shared_out, &timing);
            else if (strcmp(op_name, "sobel") == 0) cuda_sobel_shared(input, gpu_shared_out, &timing);
            
            total_h2d += timing.h2d_ms;
            total_kernel += timing.kernel_ms;
            total_d2h += timing.d2h_ms;
            total_gpu += timing.total_ms;
        }
        
        shared_res.h2d_ms = total_h2d / iterations;
        shared_res.kernel_ms = total_kernel / iterations;
        shared_res.d2h_ms = total_d2h / iterations;
        shared_res.gpu_total_ms = total_gpu / iterations;
        
        shared_res.kernel_speedup = result.cpu_ms / shared_res.kernel_ms;
        shared_res.end_to_end_speedup = result.cpu_ms / shared_res.gpu_total_ms;
        
        CompareResult shared_comp = compare_images(cpu_out, gpu_shared_out, 2);
        shared_res.correct = shared_comp.passed;
        shared_res.max_error = shared_comp.max_error;
        shared_res.mean_error = shared_comp.mean_error;
        
        benchmark_print_result(&shared_res);
        benchmark_log_result(&shared_res);
        
        snprintf(out_filename, sizeof(out_filename), "../images/output/%s_cuda_shared.png", op_name);
        save_image(out_filename, gpu_shared_out);
    }
}

int main(int argc, char** argv) {
    char inputPath[256] = "";
    char operation[64] = "all";
    int value = 50;
    int iterations = 10;
    
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            strncpy(inputPath, argv[++i], sizeof(inputPath) - 1);
        } else if (strcmp(argv[i], "--operation") == 0 && i + 1 < argc) {
            strncpy(operation, argv[++i], sizeof(operation) - 1);
        } else if (strcmp(argv[i], "--value") == 0 && i + 1 < argc) {
            value = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (strlen(inputPath) == 0) {
        printf("No input image provided, generating synthetic test image (1920x1080)...\n");
        strncpy(inputPath, "../images/input/synthetic.png", sizeof(inputPath) - 1);
        generate_synthetic_image(inputPath, 1920, 1080);
    }
    
    Image input;
    if (!load_image(inputPath, &input)) {
        return 1;
    }
    
    printf("Image:\n");
    printf("  File       : %s\n", inputPath);
    printf("  Resolution : %d x %d\n", input.width, input.height);
    printf("  Channels   : %d\n", input.channels);
    printf("  Pixels     : %d\n\n", input.width * input.height);
    
    benchmark_print_system_info();
    benchmark_init("../results/benchmark.csv");
    
    Image cpu_out, gpu_out, gpu_shared_out;
    init_image(&cpu_out, input.width, input.height, input.channels);
    init_image(&gpu_out, input.width, input.height, input.channels);
    init_image(&gpu_shared_out, input.width, input.height, input.channels);
    
    if (strcmp(operation, "all") == 0) {
        const char* ops[] = {"grayscale", "brightness", "invert", "blur", "sobel"};
        for (int i = 0; i < 5; ++i) {
            run_benchmark(ops[i], &input, &cpu_out, &gpu_out, &gpu_shared_out, value, iterations);
        }
    } else {
        run_benchmark(operation, &input, &cpu_out, &gpu_out, &gpu_shared_out, value, iterations);
    }
    
    free_image(&input);
    free_image(&cpu_out);
    free_image(&gpu_out);
    free_image(&gpu_shared_out);
    
    return 0;
}
