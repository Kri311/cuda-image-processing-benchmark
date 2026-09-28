CC = gcc
NVCC = nvcc

CFLAGS = -std=c99 -O3 -Wall -Wextra -Iinclude -Ithird_party
NVCCFLAGS = -std=c++17 -O3 -use_fast_math -Iinclude -Ithird_party

TARGET = cuda_image_benchmark

# Source files
CPU_SRCS = $(wildcard src/cpu/*.c)
CUDA_SRCS = $(wildcard src/cuda/*.cu)
UTILS_SRCS = $(wildcard src/utils/*.c)
MAIN_SRC = src/main.c

# Object files
CPU_OBJS = $(CPU_SRCS:.c=.o)
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
UTILS_OBJS = $(UTILS_SRCS:.c=.o)
MAIN_OBJ = $(MAIN_SRC:.c=.o)

OBJS = $(MAIN_OBJ) $(CPU_OBJS) $(CUDA_OBJS) $(UTILS_OBJS)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -lm

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
	rm -rf build

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
