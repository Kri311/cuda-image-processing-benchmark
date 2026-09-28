# CUDA Image Processing Benchmark: CPU vs GPU Acceleration

A pure C and CUDA-based image processing benchmark that implements common image-processing operations on both a CPU and an NVIDIA GPU. It measures execution and memory-transfer times, verifies GPU correctness against the CPU implementation, and reports exact GPU speedup metrics.

## Project Overview
This project serves as a comprehensive benchmark to compare C (Host) vs CUDA (Device) execution for fundamental image processing algorithms (Grayscale, Brightness adjustment, Invert, Gaussian Blur, and Sobel edge detection). It highlights the performance benefits of NVIDIA CUDA and demonstrates optimization techniques like shared memory tiling.

## Why CUDA?
Image processing operations are highly parallelizable because the computation for each pixel is mostly independent of others. A CPU processes pixels sequentially or with limited thread parallelism. A GPU, equipped with thousands of cores, can execute an operation on multiple pixels simultaneously, offering massive speedups for large images.

## CUDA Concepts Demonstrated
- **Threads and Blocks**: Launching thousands of threads organized in a 2D grid of thread blocks.
- **2D Indexing**: Mapping `threadIdx` and `blockIdx` to 2D image coordinates `(x, y)`.
- **Global Memory**: Allocating device memory (`cudaMalloc`) and accessing it directly from threads.
- **Shared Memory**: Using fast on-chip `__shared__` memory for neighborhood operations (convolution) to minimize slow global memory accesses.
- **Host and Device Memory**: Managing CPU (`Host`) and GPU (`Device`) memory, and transferring data via `cudaMemcpy`.
- **Synchronization**: Using `__syncthreads()` to synchronize threads within a block before accessing shared memory.
- **CUDA Events**: Utilizing `cudaEventRecord()` and `cudaEventElapsedTime()` for precise GPU timing without blocking asynchronous launches.
- **C Interoperability (`extern "C"`)**: Interfacing standard C host code with C++-based `nvcc` device code safely.

## Architecture
```
Host (C / CPU)                      Device (CUDA / GPU)
-----------------------------------------------------------------
Load Image (stb_image)         |    
allocate memory (Host)         |    allocate memory (Device)
Host -> Device Transfer  ----->|    
                               |    Launch Kernel <<<Grid, Block>>>
                               |       - Threads compute pixels
                               |       - Uses Global/Shared Mem
Device -> Host Transfer  <-----|
verify correctness & timing    |    free memory (Device)
Save Image (stb_image_write)   |
```

## Installation

### Requirements
- NVIDIA GPU (Compute Capability 3.0+)
- NVIDIA display driver
- CUDA Toolkit (nvcc)
- CMake (3.14+)
- C99 compatible compiler (gcc, clang)

### Check System
```bash
nvidia-smi
nvcc --version
cmake --version
gcc --version
```

## Build
The project relies on CMake to build correctly. 

```bash
mkdir build
cd build
cmake ..
make -j$(nproc)
```
*(Alternatively, a `Makefile` is provided in the root directory. You can just run `make`)*

## Usage
The CLI allows specifying an input image and operations. 
**Note: You should execute the binary from within the `build/` directory.**

**Run all benchmarks on a test image:**
```bash
./cuda_image_benchmark --input /absolute/path/to/image.jpg --operation all
```

**Run specific operation:**
```bash
./cuda_image_benchmark --input /absolute/path/to/image.jpg --operation blur
```

**Change brightness value and iterations:**
```bash
./cuda_image_benchmark --input /absolute/path/to/image.jpg --operation brightness --value 75 --iterations 20
```
*Note: If no input image is provided, a synthetic 1920x1080 test image will be automatically generated inside `images/input/`.*

## Benchmarking
The project runs CPU and GPU versions multiple times (default 10) and calculates the mean time, excluding the first warmup run for the GPU.
The GPU timing is split into:
- **H2D Transfer**: Host to Device memory copy (PCIe)
- **Kernel**: Execution time strictly on the GPU
- **D2H Transfer**: Device to Host memory copy (PCIe)
- **Total GPU**: End-to-end time

## Results
Benchmark results are printed to the console and saved to `results/benchmark.csv`. Output processed images are saved in `images/output/`.
*Benchmark results are hardware-dependent. Run the benchmark on your own NVIDIA GPU to generate results/benchmark.csv.*

## Correctness
Every CUDA kernel's output is compared pixel-by-pixel against the CPU reference implementation. 
The program calculates the Max Error and Mean Error, and checks if it falls within an acceptable tolerance, reporting `PASS` or `FAIL`.

## Optimization
For **Gaussian Blur** and **Sobel**, two CUDA versions are provided:
1. **Global Memory Version**: Each thread reads from global memory. In convolutions, adjacent pixels are read multiple times by different threads.
2. **Shared Memory Version**: Threads cooperatively load a tile (block + halo/radius) of the image into shared memory. Threads then perform the convolution reading from the fast shared memory. This significantly reduces global memory bandwidth pressure.

## Limitations
- PCIe transfer overhead can outweigh kernel execution time for small images (e.g. 512x512). Massive speedups are typically seen on >1080p images.
- Performance depends heavily on the specific GPU model.
- Boundary handling in shared memory can introduce thread divergence.

## Future Improvements
- **CUDA Streams**: Overlap memory transfers and computation.
- **Pinned Memory**: Use `cudaMallocHost` for faster PCIe transfers.
- **Constant Memory**: Store the convolution kernels (e.g., Gaussian, Sobel) in constant memory.
- **Separable Convolution**: Apply 1D blur horizontally then vertically for Gaussian blur to reduce operations from $O(K^2)$ to $O(2K)$.
