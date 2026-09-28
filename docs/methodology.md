# Methodology

## 1. CPU Implementations
CPU functions (written in native C) iterate sequentially over the 1D image array using nested `for` loops for `y` (height) and `x` (width).
In neighborhood operations (blur, sobel), they loop over a 3x3 kernel. Boundaries are handled using manual `clamp` functions to repeat edge pixels. 

## 2. CUDA Kernels
CUDA kernels replace the outer `for` loops of the CPU code with a massively parallel grid of threads.
Each thread calculates its absolute `x` and `y` position:
```c
int x = blockIdx.x * blockDim.x + threadIdx.x;
int y = blockIdx.y * blockDim.y + threadIdx.y;
```
If `x < width` and `y < height`, the thread processes that specific pixel. 

## 3. Thread Indexing
Threads are grouped into blocks (e.g., 16x16 threads). Blocks are grouped into a grid covering the whole image:
```c
dim3 block(16, 16);
dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
```
This ensures there are enough threads to cover images of any arbitrary dimension (handling padding cleanly).

## 4. Shared Memory Tiling
For convolutions (3x3), a pixel needs its 8 neighbors. If using global memory, threads fetch overlapping pixels, causing redundant memory accesses.
Shared memory solves this:
1. A block of threads loads a tile of the image plus a "halo" (1 pixel border) into `extern __shared__` memory.
2. `__syncthreads()` ensures all threads finish loading before any start computing. (Alpha channels safely bypass this to prevent deadlock).
3. Threads perform the convolution reading from the fast shared memory.

## 5. Benchmarking
CPU timing is done using the POSIX `clock_gettime(CLOCK_MONOTONIC)` function for high-resolution microsecond accuracy.
GPU timing is done using `cudaEvent_t`. This is strictly necessary because CUDA kernel launches (`<<<...>>>`) are asynchronous. `cudaEventSynchronize` forces the CPU to wait until the GPU finishes recording the stop event, allowing accurate timing.

## 6. Verification
The `compare_images` C function loops over all pixels of the CPU output and GPU output arrays, computing the absolute difference. `max_error` and `mean_error` are reported. A small tolerance (+/- 2) is used because floating-point precision/rounding logic differs slightly between host C computations and GPU architecture.

## 7. Speedups
- **Kernel-only Speedup**: `CPU Time / GPU Kernel Time`. This shows raw compute acceleration and highlights the raw processing power of the GPU cores.
- **End-to-End Speedup**: `CPU Time / (H2D + Kernel + D2H)`. This represents the realistic speedup for a user application where data originates and ends on the host RAM. PCIe transfer speed heavily dictates this bottleneck.
