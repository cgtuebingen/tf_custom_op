TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
CUDA_LIB = /graphics/projects/cuda/toolkit_8.0/cuda/lib
CUDA_LIB = /usr/local/cuda/lib64
# some python versions need ABI=0
ABI=1 

USER_OP=matrix_add
FLAGS=-fPIC --shared -D _GLIBCXX_USE_CXX11_ABI=$(ABI)


all: clean build

build: $(USER_OP)_op.so

$(USER_OP)_op.cu.o:
	nvcc -std=c++11 --expt-relaxed-constexpr --shared --gpu-architecture=sm_52 \
		-c -o $(USER_OP)_op.cu.o kernels/$(USER_OP)_kernel.cu \
		-isystem $(TF_INC) \
		-D GOOGLE_CUDA=1 -x cu -Xcompiler $(FLAGS)
	
$(USER_OP)_op.so: kernels/$(USER_OP)_op.cc $(USER_OP)_op.cu.o
	g++ -std=c++11 -o $(USER_OP)_op.so --shared \
		$(USER_OP)_op.cu.o kernels/$(USER_OP)_op.cc ops/$(USER_OP).cc kernels/$(USER_OP)_kernel.cc\
		-isystem $(TF_INC)  \
		-lcudart -L$(CUDA_LIB) \
		$(FLAGS)

clean:
	rm -f *.o *.so *.pyc *.npy
