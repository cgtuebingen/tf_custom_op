cmake_minimum_required( VERSION 2.8 )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set necessary flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -march=native -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -fPIC --shared -D _GLIBCXX_USE_CXX11_ABI=1" )
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11  --expt-relaxed-constexpr -D GOOGLE_CUDA=1 --gpu-architecture=sm_52" )

project( TFCustomOp )

# use cuda
find_package(CUDA 8.0 EXACT REQUIRED)
message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
set(CUDA_SAMPLE_INC "${CUDA_INCLUDE_DIRS}/../samples/common/inc")
message(STATUS "CUDA_SAMPLE_INC: ${CUDA_SAMPLE_INC}")

# get TF include path
execute_process(
    COMMAND "python" -c
            "try: import tensorflow as tf; print tf.sysconfig.get_include() \nexcept:pass\n"
    OUTPUT_VARIABLE TF_INC)
message(STATUS "TF_INC: ${TF_INC}")



include_directories(${TF_INC})
include_directories("kernels")

macro(add_tfop arg)
  cuda_add_library(${arg}_op_cu SHARED kernels/${arg}_kernel.cu)
  set_target_properties(${arg}_op_cu PROPERTIES PREFIX "")

  add_library(${arg}_op SHARED kernels/${arg}_op.cc kernels/${arg}_kernel.cc ops/${arg}.cc )
  set_target_properties(${arg}_op PROPERTIES PREFIX "")
  target_link_libraries(${arg}_op LINK_PUBLIC ${arg}_op_cu)

  message("arg = ${arg}")
endmacro()


add_tfop("matrix_add")

