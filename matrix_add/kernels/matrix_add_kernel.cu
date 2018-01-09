// ComputerGraphics Tuebingen, 2017

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "matrix_add_op.h"


namespace {

using CudaLaunchConfig = ::tensorflow::CudaLaunchConfig;

template<typename T>
__global__ void forward(CudaLaunchConfig cfg,
                        T* top,
                        const int N,
                        const T* matrixA,
                        const T* matrixB,
                        const T bias) {
  // for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
  CUDA_1D_KERNEL_LOOP(i, cfg.virtual_thread_count){
    top[i] = matrixA[i] + matrixB[i] + (T) bias;
  }
}


template<typename T>
__global__ void backward(CudaLaunchConfig cfg,
                         const T* top_diff,
                         const int N,
                         T* grad_matrixA,
                         T* grad_matrixB) {

  // for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
  CUDA_1D_KERNEL_LOOP(i, cfg.virtual_thread_count){
    grad_matrixA[i] = top_diff[i];
    grad_matrixB[i] = top_diff[i];
  }
}

} // anonymous namespace


namespace tensorflow {
namespace functor {

template <typename Dtype>
struct MatrixAddFunctor<GPUDevice, Dtype> {
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const Tensor& mA_,
                   const Tensor& mB_,
                   Tensor *mC_,
                   Dtype bias) {

    const int N = mA_.NumElements();

    ::tensorflow::CudaLaunchConfig cfg =
      ::tensorflow::GetCudaLaunchConfig(N, ctx->eigen_device<GPUDevice>());

    forward<Dtype>
      <<<cfg.block_count, cfg.thread_per_block>>>(
        cfg,
        mC_->flat<Dtype>().data(),
        mA_.NumElements(),
        mA_.flat<Dtype>().data(),
        mB_.flat<Dtype>().data(),
        bias);

  }
};

template struct MatrixAddFunctor<GPUDevice, int>;
template struct MatrixAddFunctor<GPUDevice, float>;
template struct MatrixAddFunctor<GPUDevice, double>;


template <typename Dtype>
struct MatrixAddGrad<GPUDevice, Dtype> {
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const Tensor& topdiff_,
                   Tensor *grad_mA_,
                   Tensor *grad_mB_) {

    const int N = topdiff_.NumElements();

    ::tensorflow::CudaLaunchConfig cfg =
      ::tensorflow::GetCudaLaunchConfig(N, ctx->eigen_device<GPUDevice>());

    backward<Dtype>
      <<<cfg.block_count, cfg.thread_per_block>>>(
        cfg,
        topdiff_.flat<Dtype>().data(),
        topdiff_.NumElements(),
        grad_mA_->flat<Dtype>().data(),
        grad_mB_->flat<Dtype>().data());

  }
};

template struct MatrixAddGrad<GPUDevice, int>;
template struct MatrixAddGrad<GPUDevice, float>;
template struct MatrixAddGrad<GPUDevice, double>;


} // namespace functor
} // namespace tensorflow

#endif  // GOOGLE_CUDA
