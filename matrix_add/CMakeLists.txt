cmake_minimum_required( VERSION 2.8 )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CMAKE_POLICY( SET CMP0046 OLD)


# set necessary flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -march=native -fopenmp -D_GLIBCXX_USE_CXX11_ABI=0")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fPIC --shared -D_GLIBCXX_USE_CXX11_ABI=0" )
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11  --expt-relaxed-constexpr -D GOOGLE_CUDA=1 --gpu-architecture=sm_52 -D_GLIBCXX_USE_CXX11_ABI=0" )

project( TFCustomOp )

# use cuda
find_package(CUDA 8.0 EXACT REQUIRED)
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
set(CUDA_SAMPLE_INC "${CUDA_INCLUDE_DIRS}/../samples/common/inc")
message(STATUS "CUDA_SAMPLE_INC: ${CUDA_SAMPLE_INC}")



# add modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/Modules")

# find packages
find_package(TensorFlow REQUIRED)


include_directories("${CUDA_INCLUDE_DIRS}/../../")
message(STATUS "include ${CUDA_INCLUDE_DIRS}/../../")
include_directories("/home/wieschol/.local/lib/python2.7/site-packages/tensorflow/include/external/nsync/public")
message(STATUS "include /home/wieschol/.local/lib/python2.7/site-packages/tensorflow/include/external/nsync/public")
include_directories("/graphics/opt/opt_Ubuntu16.04/tensorflow/third_party/toolchains/gpus/cuda")
message(STATUS "include /graphics/opt/opt_Ubuntu16.04/tensorflow/third_party/toolchains/gpus/cuda")

include_directories(SYSTEM "${CMAKE_SOURCE_DIR}/extra")
include_directories(SYSTEM ${TensorFlow_INCLUDE_DIR})
include_directories(SYSTEM "kernels")

# link_directories( /home/wieschol/.local/lib/python2.7/site-packages/tensorflow )

# find_library(TF_LIBABRY libtensorflow_framework HINTS /home/wieschol/.local/lib/python2.7/site-packages/tensorflow)

macro(add_tfop arg)
  cuda_add_library(${arg}_op_cu SHARED kernels/${arg}_kernel.cu)
  set_target_properties(${arg}_op_cu PROPERTIES PREFIX "")

  add_library(${arg}_op SHARED kernels/${arg}_op.cc kernels/${arg}_kernel.cc ops/${arg}.cc )

  set_target_properties(${arg}_op PROPERTIES PREFIX "")
  target_link_libraries(${arg}_op LINK_PUBLIC ${arg}_op_cu ${TensorFlow_LIBRARY})


  message("arg = ${arg}")
endmacro()


add_tfop("matrix_add")

# (TENSORFLOW)/tensorflow/contrib/makefile/downloads/nsync/public