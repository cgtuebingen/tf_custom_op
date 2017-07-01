// ComputerGraphics Tuebingen, 2017

#ifndef MATRIX_ADD_OP_HH
#define MATRIX_ADD_OP_HH

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
class OpKernelContext;
class Tensor;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;
}


namespace tensorflow {
namespace functor {

    template <typename Device, typename Dtype>
    struct MatrixAddFunctor{
      void operator ()(::tensorflow::OpKernelContext* ctx,
                       const Tensor& matrix_a,
                       const Tensor& matrix_b,
                       Tensor *output, 
                       Dtype bias);
    };

    template <typename Device, typename Dtype>
    struct MatrixAddGrad{
      void operator ()(::tensorflow::OpKernelContext* ctx,
                       const Tensor& top_diff,
                       Tensor *grad_matrix_a,
                       Tensor *grad_matrix_b);
    };


} // namespace functor
} // namespace tensorflow

#endif
