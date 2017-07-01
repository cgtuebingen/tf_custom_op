#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <stdio.h>
#include "tensorflow/core/framework/shape_inference.h"
#define EIGEN_USE_GPU

#include "matrix_add_op.cuh"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Forward-Pass (CPU)
// --------------------------------------------------
template<typename Device, typename Dtype>
class MatrixAddOp: public OpKernel {
public:
  explicit MatrixAddOp(OpKernelConstruction* context) :
    OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("bias", &bias_));
  }

  void Compute(OpKernelContext* context) override {
    // printf("--> Compute CPU Version <--\n");
    // access incoming tensors (const)
    const Tensor& matrix_a = context->input(0);
    const auto matrix_a_tensor = matrix_a.tensor<Dtype, 4>();
    const Tensor& matrix_b = context->input(1);
    const auto matrix_b_tensor = matrix_b.tensor<Dtype, 4>();

    // get dimensions
    const int B = matrix_a.shape().dim_size(0);
    const int M = matrix_a.shape().dim_size(1);
    const int N = matrix_a.shape().dim_size(2);
    const int D = matrix_a.shape().dim_size(3);

    // specify output shape
    // just do
    // "OP_REQUIRES_OK(context,context->allocate_output(0, matrix_a.shape(), &output));"
    // or the longer way
    TensorShape output_shape;
    output_shape.AddDim(B);
    output_shape.AddDim(M);
    output_shape.AddDim(N);
    output_shape.AddDim(D);
    // construct output
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto out_tensor = output->tensor<Dtype, 4>();

    for (int b = 0; b < B; ++b)
      for (int r = 0; r < M; ++r)
        for (int c = 0; c < N; ++c)
          for (int d = 0; d < D; ++d)
            out_tensor(b, r, c, d) = matrix_a_tensor(b, r, c, d) + matrix_b_tensor(b, r, c, d) + bias_;
  }

private:
//  TF_DISALLOW_COPY_AND_ASSIGN(MatrixAddOp);
  float bias_;
};

// Forward-Pass (GPU)
// --------------------------------------------------
template<typename Dtype>
class MatrixAddOp<GPUDevice, Dtype>: public OpKernel {
public:
  explicit MatrixAddOp(OpKernelConstruction* context) :
    OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("bias", &bias_));
  }

  void Compute(OpKernelContext* context) override {
    // printf("--> Compute GPU Version <--\n");
    const Tensor& matrix_a = context->input(0);
    const Tensor& matrix_b = context->input(1);

    // access the elements
    const int N = matrix_a.NumElements();
    auto matrix_a_flat = matrix_a.flat<Dtype>();
    auto matrix_b_flat = matrix_b.flat<Dtype>();

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, matrix_a.shape(), &output));
    auto out_flat = output->flat<Dtype>();

    Dtype* _top = out_flat.data();
    const Dtype* _inA = matrix_a_flat.data();
    const Dtype* _inB = matrix_b_flat.data();

    MatrixAddOpForwardCudaKernelLauncher<Dtype>(_top, N,
        _inA, _inB,
        bias_);
  }

private:
//  TF_DISALLOW_COPY_AND_ASSIGN(MatrixAddOp);
  float bias_;
};

// Backward-Pass (CPU)
// --------------------------------------------------
template<typename Device, typename Dtype>
class MatrixAddGradOp: public OpKernel {
public:
  explicit MatrixAddGradOp(OpKernelConstruction* context) :
    OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    // printf("--> Compute CPU Version <--\n");
    const Tensor& top_diff = context->input(0);
    const Tensor& features = context->input(1);
    const int N = top_diff.NumElements();

    const Dtype* topdiff_ptr = top_diff.flat<Dtype>().data();

    Tensor* matrix_a_grad = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, features.shape(), &matrix_a_grad));
    matrix_a_grad->flat<Dtype>().setZero();

    Tensor* matrix_b_grad = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, features.shape(), &matrix_b_grad));
    matrix_b_grad->flat<Dtype>().setZero();

    Dtype* matrix_a_grad_ptr = matrix_a_grad->flat<Dtype>().data();
    Dtype* matrix_b_grad_ptr = matrix_b_grad->flat<Dtype>().data();

    for (int i = 0; i < N; ++i) {
      matrix_a_grad_ptr[i] = topdiff_ptr[i];
      matrix_b_grad_ptr[i] = topdiff_ptr[i];
    }

  }

};

// Backward-Pass (GPU)
// --------------------------------------------------
template<typename Dtype>
class MatrixAddGradOp<GPUDevice, Dtype>: public OpKernel {
public:
  explicit MatrixAddGradOp(OpKernelConstruction* context) :
    OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {
    // printf("--> Compute GPU Version <--\n");
    const Tensor& top_diff = context->input(0);
    const Tensor& matrix_a = context->input(1);
    const Tensor& matrix_b = context->input(2);

    const int N = top_diff.shape().num_elements();

    Tensor* grad_matrix_a = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, matrix_a.shape(), &grad_matrix_a));
    Tensor* grad_matrix_b = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, matrix_b.shape(), &grad_matrix_b));

    MatrixAddOpBackwardCudaKernelLauncher(top_diff.flat<Dtype>().data(), N,
                                          matrix_a.flat<Dtype>().data(), matrix_b.flat<Dtype>().data(),
                                          grad_matrix_a->flat<Dtype>().data(), grad_matrix_b->flat<Dtype>().data());
  }

};


#define REGISTER_MYCOPY_KERNELS(type)                                          \
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


REGISTER_MYCOPY_KERNELS(int);
REGISTER_MYCOPY_KERNELS(float);
REGISTER_MYCOPY_KERNELS(double);

}
