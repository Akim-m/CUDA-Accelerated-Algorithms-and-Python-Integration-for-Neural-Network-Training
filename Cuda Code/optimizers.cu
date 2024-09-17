#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

// Kernel for SGD optimizer
__global__ void sgd_update(float* weights, float* gradients, float lr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        weights[idx] -= lr * gradients[idx];
    }
}



__global__ void adam_update(float* weights, float* gradients, float* m, float* v, 
                            float beta1, float beta2, float alpha, float epsilon, 
                            int t, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float grad = gradients[idx];
        
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;

        // Update biased second moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;

        // Compute bias-corrected first moment estimate
        float m_hat = m[idx] / (1.0f - powf(beta1, t));

        // Compute bias-corrected second moment estimate
        float v_hat = v[idx] / (1.0f - powf(beta2, t));

        // Update the weights
        weights[idx] -= alpha * m_hat / (sqrtf(v_hat) + epsilon);
    }
}



__global__ void rmsprop_update(float* weights, float* gradients, float* s, 
                               float alpha, float beta, float epsilon, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Update running average of squared gradients
        s[idx] = beta * s[idx] + (1.0f - beta) * gradients[idx] * gradients[idx];

        // Update the weights
        weights[idx] -= alpha * gradients[idx] / (sqrtf(s[idx]) + epsilon);
    }
}


// Host function for SGD
extern "C" void sgd_optimizer(float* d_weights, float* d_gradients, float lr, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    sgd_update<<<numBlocks, blockSize>>>(d_weights, d_gradients, lr, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "SGD CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
    cudaDeviceSynchronize();
}


extern "C" void adam_optimizer(float* d_weights, float* d_gradients, float* d_m, float* d_v, 
                               float beta1, float beta2, float alpha, float epsilon, 
                               int t, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    adam_update<<<numBlocks, blockSize>>>(d_weights, d_gradients, d_m, d_v, beta1, beta2, alpha, epsilon, t, n);
}

extern "C" void rmsprop_optimizer(float* d_weights, float* d_gradients, float* d_s, 
                                  float alpha, float beta, float epsilon, int n) {
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    rmsprop_update<<<numBlocks, blockSize>>>(d_weights, d_gradients, d_s, alpha, beta, epsilon, n);
}
