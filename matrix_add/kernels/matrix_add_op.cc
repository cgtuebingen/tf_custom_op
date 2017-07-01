#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <stdio.h>

#include "matrix_add_op.h"

namespace tensorflow {

namespace functor {

template <typename Dtype>
struct MatrixAddFunctor<CPUDevice, Dtype> {
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const Tensor& mA_,
                   const Tensor& mB_,
                   Tensor *mC_,
                   Dtype bias) {

    auto mC = mC_->tensor<Dtype, 4>();
    auto mA = mA_.tensor<Dtype, 4>();
    auto mB = mB_.tensor<Dtype, 4>();

    mC.setZero();

    // get dimensions
    const int B = mA_.shape().dim_size(0);
    const int M = mA_.shape().dim_size(1);
    const int N = mA_.shape().dim_size(2);
    const int D = mA_.shape().dim_size(3);

    // the computation
    for (int b = 0; b < B; ++b)
      for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
          for (int d = 0; d < D; ++d)
            mC(b, r, c, d) = mA(b, r, c, d) + mB(b, r, c, d) + bias;
  }
};

template struct MatrixAddFunctor<CPUDevice, int>;
template struct MatrixAddFunctor<CPUDevice, float>;
template struct MatrixAddFunctor<CPUDevice, double>;


template <typename Dtype>
struct MatrixAddGrad<CPUDevice, Dtype> {
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const Tensor& topdiff_,
                   Tensor *grad_mA_,
                   Tensor *grad_mB_) {

    const int N = topdiff_.NumElements();

    grad_mA_->flat<Dtype>().setZero();
    grad_mB_->flat<Dtype>().setZero();

    const Dtype* topdiff = topdiff_.flat<Dtype>().data();
    Dtype* grad_mA = grad_mA_->flat<Dtype>().data();
    Dtype* grad_mB = grad_mB_->flat<Dtype>().data();

    for (int i = 0; i < N; ++i) {
      grad_mA[i] = topdiff[i];
      grad_mB[i] = topdiff[i];
    }

  }
};

template struct MatrixAddGrad<CPUDevice, int>;
template struct MatrixAddGrad<CPUDevice, float>;
template struct MatrixAddGrad<CPUDevice, double>;


} // namespace functor


// Forward-Pass (CPU, GPU)
// --------------------------------------------------
template<typename Device, typename Dtype>
class MatrixAddOp: public OpKernel {
public:
  explicit MatrixAddOp(OpKernelConstruction* ctx) :
    OpKernel(ctx) {
    OP_REQUIRES_OK(ctx,
                   ctx->GetAttr("bias", &bias_));
  }

  void Compute(OpKernelContext* ctx) override {
    // printf("--> Compute CPU Version <--\n");
    const Tensor& mA = ctx->input(0);
    const Tensor& mB = ctx->input(1);

    const int B = mA.shape().dim_size(0);
    const int M = mA.shape().dim_size(1);
    const int N = mA.shape().dim_size(2);
    const int D = mA.shape().dim_size(3);

    TensorShape output_shape({B, M, N, D});
    // same as 
    // output_shape.AddDim(B); ....

    Tensor* mC = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &mC));
    // same as "OP_REQUIRES_OK(ctx,ctx->allocate_output(0, mA.tensor<Dtype, 4>().shape(), &mC));"

    ::tensorflow::functor::MatrixAddFunctor<Device, Dtype>()(ctx,
        mA, mB, mC, bias_);

  }

private:
  TF_DISALLOW_COPY_AND_ASSIGN(MatrixAddOp);
  float bias_;
};

// Backward-Pass (CPU, GPU)
// --------------------------------------------------
template<typename Device, typename Dtype>
class MatrixAddGradOp: public OpKernel {
public:
  explicit MatrixAddGradOp(OpKernelConstruction* ctx) :
    OpKernel(ctx) {
  }

  void Compute(OpKernelContext* ctx) override {
    // printf("--> Compute CPU Version <--\n");
    const Tensor& mA = ctx->input(0);
    const Tensor& mB = ctx->input(1);
    const Tensor& topdiff = ctx->input(2);

    Tensor* grad_mA = nullptr;
    Tensor* grad_mB = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, mA.shape(), &grad_mA));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, mB.shape(), &grad_mB));

    ::tensorflow::functor::MatrixAddGrad<Device, Dtype>()(ctx,
        topdiff, grad_mA, grad_mB);

  }

};


#define OPNAME(NAME) NAME ## Op
#define REGISTER(NAME, Dtype)                                          \
  REGISTER_KERNEL_BUILDER(                                             \
      Name(#NAME).Device(DEVICE_CPU).TypeConstraint<Dtype>("T"),       \
      OPNAME(NAME)<CPUDevice, Dtype>);                                 \
  REGISTER_KERNEL_BUILDER(                                             \
      Name(#NAME).Device(DEVICE_GPU).TypeConstraint<Dtype>("T"),       \
      OPNAME(NAME)<GPUDevice, Dtype>);                                           


REGISTER(MatrixAdd, int);
REGISTER(MatrixAdd, float);
REGISTER(MatrixAdd, double);
REGISTER(MatrixAddGrad, int);
REGISTER(MatrixAddGrad, float);
REGISTER(MatrixAddGrad, double);



} // namespace tensorflow