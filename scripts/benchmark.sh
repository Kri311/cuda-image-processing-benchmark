#!/bin/bash

# Simple script to run all benchmarks

echo "Building project..."
make clean
make -j$(nproc)

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Running benchmarks on synthetic image..."
./cuda_image_benchmark --operation all --iterations 20

echo "Benchmark complete. Results saved to results/benchmark.csv"
