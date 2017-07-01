#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <stdio.h>

#include "matrix_add_op.h"

namespace tensorflow {

namespace functor {

template <typename Dtype>
struct MatrixAddFunctor<CPUDevice, Dtype> {
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const Tensor& matrix_a,
                   const Tensor& matrix_b,
                   Tensor *output,
                   Dtype bias) {

    auto mC = output->tensor<Dtype, 4>();
    auto mA = matrix_a.tensor<Dtype, 4>();
    auto mB = matrix_b.tensor<Dtype, 4>();

    mC.setZero();

    // get dimensions
    const int B = matrix_a.shape().dim_size(0);
    const int M = matrix_a.shape().dim_size(1);
    const int N = matrix_a.shape().dim_size(2);
    const int D = matrix_a.shape().dim_size(3);

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
                   const Tensor& top_diff,
                   Tensor *grad_matrix_a,
                   Tensor *grad_matrix_b) {

    const int N = top_diff.NumElements();

    grad_matrix_a->flat<Dtype>().setZero();
    grad_matrix_b->flat<Dtype>().setZero();

    const Dtype* topdiff_ptr = top_diff.flat<Dtype>().data();
    Dtype* grad_matrix_a_ptr = grad_matrix_a->flat<Dtype>().data();
    Dtype* grad_matrix_b_ptr = grad_matrix_b->flat<Dtype>().data();

    for (int i = 0; i < N; ++i) {
      grad_matrix_a_ptr[i] = topdiff_ptr[i];
      grad_matrix_b_ptr[i] = topdiff_ptr[i];
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
    // access incoming tensors (const)
    const Tensor& matrix_a = ctx->input(0);
    const Tensor& matrix_b = ctx->input(1);


    // get dimensions
    const int B = matrix_a.shape().dim_size(0);
    const int M = matrix_a.shape().dim_size(1);
    const int N = matrix_a.shape().dim_size(2);
    const int D = matrix_a.shape().dim_size(3);

    // specify output shape
    TensorShape output_shape;
    output_shape.AddDim(B);
    output_shape.AddDim(M);
    output_shape.AddDim(N);
    output_shape.AddDim(D);
    // same as "OP_REQUIRES_OK(ctx,ctx->allocate_output(0, matrix_a.tensor<Dtype, 4>().shape(), &output));"

    // construct output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output));
    auto out_tensor = output->tensor<Dtype, 4>();

    ::tensorflow::functor::MatrixAddFunctor<Device, Dtype>()(ctx,
        matrix_a, matrix_b, output, bias_);

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

    const Tensor& top_diff = ctx->input(0);
    const Tensor& matrix_a = ctx->input(1);
    const Tensor& matrix_b = ctx->input(2);

    const int N = top_diff.shape().num_elements();

    Tensor* grad_matrix_a = nullptr;
    Tensor* grad_matrix_b = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, matrix_a.shape(), &grad_matrix_a));
    OP_REQUIRES_OK(ctx, ctx->allocate_output(1, matrix_b.shape(), &grad_matrix_b));

    ::tensorflow::functor::MatrixAddGrad<Device, Dtype>()(ctx,
        top_diff, grad_matrix_a, grad_matrix_b);

  }

};


#define REGISTER(type)                                                         \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MatrixAdd").Device(DEVICE_CPU).TypeConstraint<type>("T"),          \
      MatrixAddOp<CPUDevice, type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MatrixAdd").Device(DEVICE_GPU).TypeConstraint<type>("T"),          \
      MatrixAddOp<GPUDevice, type>);                                           \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MatrixAddGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"),      \
      MatrixAddGradOp<CPUDevice, type>);                                       \
  REGISTER_KERNEL_BUILDER(                                                     \
      Name("MatrixAddGrad").Device(DEVICE_GPU).TypeConstraint<type>("T"),      \
      MatrixAddGradOp<GPUDevice, type>);


REGISTER(int);
REGISTER(float);
REGISTER(double);

} // namespace tensorflow