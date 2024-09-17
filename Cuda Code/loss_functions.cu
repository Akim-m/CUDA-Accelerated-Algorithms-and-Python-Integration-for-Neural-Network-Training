#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <iostream>

// CUDA kernel for computing absolute differences (used for MAE)
__global__ void abs_diff(float* y_true, float* y_pred, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = fabsf(y_true[idx] - y_pred[idx]);
    }
}

// CUDA kernel for computing squared differences (used for MSE)
__global__ void squared_diff(float* y_true, float* y_pred, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float diff = y_true[idx] - y_pred[idx];
        output[idx] = diff * diff;
    }
}

// CUDA kernel for computing Huber Loss
__global__ void huber_loss(float* y_true, float* y_pred, float* output, int n, float delta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float diff = y_true[idx] - y_pred[idx];
        if (fabsf(diff) <= delta) {
            output[idx] = 0.5 * diff * diff;
        } else {
            output[idx] = delta * (fabsf(diff) - 0.5 * delta);
        }
    }
}

// CUDA kernel for computing Cross-Entropy Loss
__global__ void cross_entropy_loss(float* y_true, float* y_pred, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = -y_true[idx] * logf(y_pred[idx]) - (1.0 - y_true[idx]) * logf(1.0 - y_pred[idx]);
    }
}

// Reduction using cuBLAS to sum up the result array for final loss value
void cublas_sum(float* d_output, int n, float* result) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float sum_result;
    cublasSasum(handle, n, d_output, 1, &sum_result);  // Summing the array

    // Copy result back to host
    *result = sum_result;

    cublasDestroy(handle);
}

// Host function for MAE (Mean Absolute Error)
extern "C" void mae_loss(float* y_true, float* y_pred, int n, float* result) {
    float *d_y_true, *d_y_pred, *d_output;

    // Allocate device memory
    cudaMalloc((void**)&d_y_true, n * sizeof(float));
    cudaMalloc((void**)&d_y_pred, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    // Copy inputs to device
    cudaMemcpy(d_y_true, y_true, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_pred, y_pred, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to compute absolute differences
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    abs_diff<<<numBlocks, blockSize>>>(d_y_true, d_y_pred, d_output, n);

    // Sum up the result using cuBLAS
    cublas_sum(d_output, n, result);

    // Free device memory
    cudaFree(d_y_true);
    cudaFree(d_y_pred);
    cudaFree(d_output);

    // Normalize by number of elements (for Mean)
    *result /= n;
}

// Host function for MSE (Mean Squared Error)
extern "C" void mse_loss(float* y_true, float* y_pred, int n, float* result) {
    float *d_y_true, *d_y_pred, *d_output;

    // Allocate device memory
    cudaMalloc((void**)&d_y_true, n * sizeof(float));
    cudaMalloc((void**)&d_y_pred, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    // Copy inputs to device
    cudaMemcpy(d_y_true, y_true, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_pred, y_pred, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to compute squared differences
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    squared_diff<<<numBlocks, blockSize>>>(d_y_true, d_y_pred, d_output, n);

    // Sum up the result using cuBLAS
    cublas_sum(d_output, n, result);

    // Free device memory
    cudaFree(d_y_true);
    cudaFree(d_y_pred);
    cudaFree(d_output);

    // Normalize by number of elements (for Mean)
    *result /= n;
}

// Host function for Cross-Entropy Loss
extern "C" void cross_entropy_loss_host(float* y_true, float* y_pred, int n, float* result) {
    float *d_y_true, *d_y_pred, *d_output;

    // Allocate device memory
    cudaMalloc((void**)&d_y_true, n * sizeof(float));
    cudaMalloc((void**)&d_y_pred, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    // Copy inputs to device
    cudaMemcpy(d_y_true, y_true, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_pred, y_pred, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to compute cross-entropy loss
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    cross_entropy_loss<<<numBlocks, blockSize>>>(d_y_true, d_y_pred, d_output, n);

    // Sum up the result using cuBLAS
    cublas_sum(d_output, n, result);

    // Free device memory
    cudaFree(d_y_true);
    cudaFree(d_y_pred);
    cudaFree(d_output);

    // Normalize by number of elements (for Mean)
    *result /= n;
}

// Host function for Huber Loss
extern "C" void huber_loss_host(float* y_true, float* y_pred, int n, float delta, float* result) {
    float *d_y_true, *d_y_pred, *d_output;

    // Allocate device memory
    cudaMalloc((void**)&d_y_true, n * sizeof(float));
    cudaMalloc((void**)&d_y_pred, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    // Copy inputs to device
    cudaMemcpy(d_y_true, y_true, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y_pred, y_pred, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel to compute huber loss
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    huber_loss<<<numBlocks, blockSize>>>(d_y_true, d_y_pred, d_output, n, delta);

    // Sum up the result using cuBLAS
    cublas_sum(d_output, n, result);

    // Free device memory
    cudaFree(d_y_true);
    cudaFree(d_y_pred);
    cudaFree(d_output);

    // Normalize by number of elements (for Mean)
    *result /= n;
}
