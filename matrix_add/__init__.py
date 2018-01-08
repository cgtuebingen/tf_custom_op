# ComputerGraphics Tuebingen, 2017

# manually generated file
import tensorflow as tf
import os
from tensorflow.python.framework import ops

__all__ = ['matrix_add', 'matrix_add_grad']

path = os.path.join(os.path.dirname(__file__), 'matrix_add_op.so')
_matrix_add_module = tf.load_op_library(path)

matrix_add = _matrix_add_module.matrix_add
matrix_add_grad = _matrix_add_module.matrix_add_grad


@ops.RegisterGradient("MatrixAdd")
def _MatrixAddGrad(op, *grads):
    bias = op.get_attr('bias')
    matA = op.inputs[0]
    matB = op.inputs[1]
    # top = op.outputs[0]
    topdiff = grads[0]
    return _matrix_add_module.matrix_add_grad(matA, matB, topdiff, bias=bias)


"""
Example:

    from matrix_add import matrix_add
    matrix_add(...)




nvcc -std=c++11 --expt-relaxed-constexpr --shared --gpu-architecture=sm_52 \
    -c -o matrix_add_op.cu.o kernels/matrix_add_kernel.cu \
    -isystem /home/wieschol/.local/lib/python2.7/site-packages/tensorflow/include \
    -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --shared -D _GLIBCXX_USE_CXX11_ABI=1
g++ -std=c++11 -o matrix_add_op.so --shared \
    matrix_add_op.cu.o kernels/matrix_add_op.cc ops/matrix_add.cc kernels/matrix_add_kernel.cc\
    -isystem /home/wieschol/.local/lib/python2.7/site-packages/tensorflow/include  \
    -lcudart -L/usr/local/cuda/lib64 \
    -fPIC --shared -D _GLIBCXX_USE_CXX11_ABI=1
"""
