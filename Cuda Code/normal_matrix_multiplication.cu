#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for basic matrix multiplication
__global__ void matrixMulKernel(float* A, float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float value = 0;
        for (int i = 0; i < k; ++i) {
            value += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = value;
    }
}

// Host function to perform matrix multiplication on GPU
extern "C" void normal_matrix_multiply(float* h_A, float* h_B, float* h_C, int m, int n, int k) {
    // Allocate memory on GPU
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                       (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch matrix multiplication kernel
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, m, n, k);

    // Copy result matrix C back to host
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
