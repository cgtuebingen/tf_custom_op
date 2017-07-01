#ifndef MATRIX_ADD_OP_HH
#define MATRIX_ADD_OP_HH

template<typename T>
void MatrixAddOpForwardCudaKernelLauncher(T* top, const int N, 
                                          const T* matrixA, const T* matrixB, 
                                          const float bias);

template<typename T>
void MatrixAddOpBackwardCudaKernelLauncher(const T* top_diff, const int N, 
                                           const T* matrixA, const T* matrixB,
                                           T* grad_matrixA, T* grad_matrixB);
#endif