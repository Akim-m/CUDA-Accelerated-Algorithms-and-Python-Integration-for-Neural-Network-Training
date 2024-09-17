#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>
#include <iostream>

// Activation functions on the GPU
__device__ float linear(float x) {
    return x;
}

__device__ float sigmoid(float x) {
    return 1.0 / (1.0 + expf(-x));
}

__device__ float tanh_act(float x) {
    return tanhf(x);
}

__device__ float relu(float x) {
    return fmaxf(0.0, x);
}

// Softmax is slightly different; it's computed over a vector
__global__ void softmax(float* input, float* output, int n) {
    float max_val = input[0];
    for (int i = 1; i < n; ++i)
        max_val = fmaxf(max_val, input[i]);

    float sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += expf(input[i] - max_val);

    for (int i = 0; i < n; ++i)
        output[i] = expf(input[i] - max_val) / sum;
}

__global__ void apply_activation(float* input, float* output, int n, int activation) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        switch (activation) {
            case 0: output[idx] = linear(x); break;    // Linear
            case 1: output[idx] = sigmoid(x); break;   // Sigmoid
            case 2: output[idx] = tanh_act(x); break;  // Tanh
            case 3: output[idx] = relu(x); break;      // ReLU
            default: output[idx] = x; break;
        }
    }
}

// CUDA interface functions for Python
extern "C" void activate(float* input, float* output, int n, int activation) {
    float* d_input;
    float* d_output;

    // Allocate device memory
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch activation kernel (excluding Softmax for now)
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    apply_activation<<<numBlocks, blockSize>>>(d_input, d_output, n, activation);

    // Copy result back to host
    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
}

extern "C" void softmax_activate(float* input, float* output, int n) {
    float* d_input;
    float* d_output;

    // Allocate device memory
    cudaMalloc((void**)&d_input, n * sizeof(float));
    cudaMalloc((void**)&d_output, n * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch softmax kernel
    softmax<<<1, 1>>>(d_input, d_output, n);

    // Copy result back to host
    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
}
