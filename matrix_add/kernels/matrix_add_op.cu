// ComputerGraphics Tuebingen, 2017

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <stdio.h>
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "matrix_add_op.h"


namespace MatrixAddCuda {

template<typename T>
__global__ void forward(T* top,
                        const int N,
                        const T* matrixA,
                        const T* matrixB,
                        const T bias) {

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    top[i] = matrixA[i] + matrixB[i] + (T) bias;
  }

}


template<typename T>
__global__ void backward(const T* top_diff,
                         const int N,
                         T* grad_matrixA,
                         T* grad_matrixB) {

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    grad_matrixA[i] = top_diff[i];
    grad_matrixB[i] = top_diff[i];
  }

}

} // namespace MatrixAddCuda



namespace tensorflow {
namespace functor {

template <typename Dtype>
struct MatrixAddFunctor<GPUDevice, Dtype> {
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const Tensor& matrix_a,
                   const Tensor& matrix_b,
                   Tensor *output,
                   Dtype bias) {

    const int N = matrix_a.NumElements();

    ::tensorflow::CudaLaunchConfig config =
      ::tensorflow::GetCudaLaunchConfig(N, ctx->eigen_device<GPUDevice>());

    MatrixAddCuda::forward<Dtype>
      <<<config.block_count, config.thread_per_block>>>(
        output->flat<Dtype>().data(),
        matrix_a.NumElements(),
        matrix_a.flat<Dtype>().data(),
        matrix_b.flat<Dtype>().data(),
        bias);

  }
};

template struct MatrixAddFunctor<GPUDevice, int>;
template struct MatrixAddFunctor<GPUDevice, float>;
template struct MatrixAddFunctor<GPUDevice, double>;



template <typename Dtype>
struct MatrixAddGrad<GPUDevice, Dtype> {
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const Tensor& top_diff,
                   Tensor *grad_matrix_a,
                   Tensor *grad_matrix_b) {

    const int N = top_diff.NumElements();

    ::tensorflow::CudaLaunchConfig config =
      ::tensorflow::GetCudaLaunchConfig(N, ctx->eigen_device<GPUDevice>());

    MatrixAddCuda::backward<Dtype>
      <<<config.block_count, config.thread_per_block>>>(
        top_diff.flat<Dtype>().data(),
        top_diff.NumElements(),
        grad_matrix_a->flat<Dtype>().data(),
        grad_matrix_b->flat<Dtype>().data());

  }
};

template struct MatrixAddGrad<GPUDevice, int>;
template struct MatrixAddGrad<GPUDevice, float>;
template struct MatrixAddGrad<GPUDevice, double>;


} // namespace functor
} // namespace tensorflow

#endif  // GOOGLE_CUDA
