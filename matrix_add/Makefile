TF_INC = $(shell python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
CUDA_LIB = /graphics/projects/cuda/toolkit_8.0/cuda/lib
CUDA_LIB = /usr/local/cuda/lib64

USER_OP=matrix_add_op
FLAGS=-fPIC --shared -D _GLIBCXX_USE_CXX11_ABI=$(ABI)

ABI=1

all: clean build

build: $(USER_OP).so

$(USER_OP).so: $(USER_OP).cc
	nvcc -std=c++11 -c -o $(USER_OP).cu.o $(USER_OP).cu -I $(TF_INC) -D GOOGLE_CUDA=1 -x cu -Xcompiler $(FLAGS)
	g++ -std=c++11 -o $(USER_OP).so $(USER_OP).cc $(USER_OP).cu.o -I $(TF_INC)  -lcudart -L$(CUDA_LIB)    $(FLAGS)

clean:
	rm -f *.o *.so *.pyc *.npy