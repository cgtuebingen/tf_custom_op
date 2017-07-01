cmake_minimum_required( VERSION 2.8 )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SSE_FLAGS} -march=native -fopenmp")

project( MatrixAddOp )


# set CUDA paths
# message(STATUS "Module path: ${CMAKE_MODULE_PATH}")
# LIST(APPEND CMAKE_MODULE_PATH "/usr/share/cmake-3.5/Modules/FindCUDA")
# message(STATUS "Module path: ${CMAKE_MODULE_PATH}")

find_package(CUDA 8.0 EXACT REQUIRED)
# message(STATUS "CUDA_INCLUDE_DIRS: ${CUDA_INCLUDE_DIRS}")
# include_directories(${CUDA_INCLUDE_DIRS})

execute_process(
    COMMAND "python" -c
            "try: import tensorflow as tf; print tf.sysconfig.get_include() \nexcept:pass\n"
    OUTPUT_VARIABLE TF_INC)


include_directories("kernels")
include_directories(${TF_INC})

cuda_add_library(matrix_add_op_cu SHARED kernels/matrix_add_kernel.cu)
set_target_properties(matrix_add_op_cu PROPERTIES PREFIX "")

add_library(matrix_add_op SHARED kernels/matrix_add_op.cc kernels/matrix_add_kernel.cc ops/matrix_add.cc )
set_target_properties(matrix_add_op PROPERTIES PREFIX "")
target_link_libraries(matrix_add_op LINK_PUBLIC matrix_add_op_cu)

# add_executable(MyOpt opt_6x6_analytics.cpp)

# if(CUDA)
#     target_link_libraries(MyOpt ${LAPACK_LIBRARIES} ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CERES_LIBRARIES} ${Suitesparse_LIBRARIES})
# else()
#     target_link_libraries( MyOpt ${OpenCV_LIBS} ${EIGEN3_INCLUDE_DIR} ${CERES_LIBRARIES})
# endif()