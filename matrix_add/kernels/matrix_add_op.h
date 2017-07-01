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
                       const Tensor& mA_,
                       const Tensor& mB_,
                       Tensor *mC_, 
                       Dtype bias);
    };

    template <typename Device, typename Dtype>
    struct MatrixAddGrad{
      void operator ()(::tensorflow::OpKernelContext* ctx,
                       const Tensor& topdiff_,
                       Tensor *grad_mA_,
                       Tensor *grad_mB_);
    };


} // namespace functor
} // namespace tensorflow

#endif
