#include <stdio.h>
#include "matrix_add_op.cu.hh"

template<typename T>
__global__ void MatrixAddOpForwardCudaKernel(T* top, const int N, 
                                             const T* matrixA, const T* matrixB, 
                                             const float bias) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;i += blockDim.x * gridDim.x) {
    top[i] = matrixA[i] + matrixB[i] + (T) bias;
  }
}

template<typename T>
void MatrixAddOpForwardCudaKernelLauncher(T* top, const int N, 
                                          const T* matrixA, const T* matrixB, 
                                          const float bias) {
  MatrixAddOpForwardCudaKernel<T> <<<32, 256>>>(top, N, 
                                                matrixA, matrixB, 
                                                bias);
  cudaDeviceSynchronize();
}

#define REGISTER_MATRIX_ADD_FORWARD(T) \
  template void MatrixAddOpForwardCudaKernelLauncher<T>(T* top, const int N, \
                                                        const T* matrixA, const T* matrixB, \
                                                        const float bias);

REGISTER_MATRIX_ADD_FORWARD(int);
REGISTER_MATRIX_ADD_FORWARD(float);
REGISTER_MATRIX_ADD_FORWARD(double);

template<typename T>
__global__ void MatrixAddOpBackwardCudaKernel(const T* top_diff, const int N, 
                                              const T* matrixA, const T* matrixB,
                                              T* grad_matrixA, T* grad_matrixB) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;i += blockDim.x * gridDim.x) {
    grad_matrixA[i] = top_diff[i];
    grad_matrixB[i] = top_diff[i];
  }
}

template<typename T>
void MatrixAddOpBackwardCudaKernelLauncher(const T* top_diff, const int N, 
                                           const T* matrixA, const T* matrixB,
                                           T* grad_matrixA, T* grad_matrixB) {
  MatrixAddOpBackwardCudaKernel<T> <<<32, 256>>>(top_diff, N, 
                                                 matrixA, matrixB,
                                                 grad_matrixA, grad_matrixB);
  cudaDeviceSynchronize();
}

#define REGISTER_MATRIX_ADD_GRADIENT(T) \
  template void MatrixAddOpBackwardCudaKernelLauncher<T>(const T* top_diff, const int N, \
                                                         const T* matrixA, const T* matrixB, \
                                                         T* grad_matrixA, T* grad_matrixB);

REGISTER_MATRIX_ADD_GRADIENT(int);
REGISTER_MATRIX_ADD_GRADIENT(float);
REGISTER_MATRIX_ADD_GRADIENT(double);
