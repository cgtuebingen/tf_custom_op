TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
CUDA_LIB = /graphics/projects/cuda/toolkit_8.0/cuda/lib

ABI=1

all: clean build

build: matrix_add_op.so

matrix_add_op.cu.o: matrix_add_op.cu.cc
	nvcc -std=c++11 -c -o $@ $? -I $(TF_INC) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --shared  -D _GLIBCXX_USE_CXX11_ABI=$(ABI)

matrix_add_op.so: matrix_add_op.cc matrix_add_op.cu.o
	g++ -std=c++11 -shared -o $@ $? -I $(TF_INC) -fPIC -lcudart -L$(CUDA_LIB) -D _GLIBCXX_USE_CXX11_ABI=$(ABI)

clean:
	rm -f *.o *.so *.pyc *.npy