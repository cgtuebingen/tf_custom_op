TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
TF_NSYNC=/home/wieschol/.local/lib/python2.7/site-packages/tensorflow/include/external/nsync/public
CUDA_LIB = /graphics/projects/cuda/toolkit_8.0/cuda/lib

FLAGS=-fPIC --shared -D_GLIBCXX_USE_CXX11_ABI=0


all: clean build

build: matrix_add_op.so

matrix_add_op.cu.o:
    nvcc -std=c++11 --expt-relaxed-constexpr --shared --gpu-architecture=sm_52 \
      -c -o matrix_add_op.cu.o kernels/matrix_add_kernel.cu \
      -isystem $(TF_INC) \
      -isystem /graphics/opt/opt_Ubuntu16.04/cuda/toolkit_8.0/ \
      -isystem $(TF_NSYNC) \
      -isystem /graphics/opt/opt_Ubuntu16.04/tensorflow/third_party/toolchains/gpus/cuda \
      -L$(TF_LIB) -ltensorflow_framework\
      -D GOOGLE_CUDA=1 -x cu -Xcompiler $(FLAGS)
  
matrix_add_op.so: kernels/matrix_add_op.cc matrix_add_op.cu.o
    g++ -std=c++11 -o matrix_add_op.so --shared \
      matrix_add_op.cu.o kernels/matrix_add_op.cc ops/matrix_add.cc kernels/matrix_add_kernel.cc\
      -isystem $(TF_INC)  \
      -isystem $(TF_NSYNC) \
      -lcudart -L$(CUDA_LIB) \
      -L$(TF_LIB) -ltensorflow_framework\
      $(FLAGS)

clean:
    rm -f *.o *.so *.pyc *.npy




