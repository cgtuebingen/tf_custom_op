# University Tuebingen, 2018
include(FindPackageHandleStandardArgs)
unset(TENSORFLOW_FOUND)


execute_process(
    COMMAND "python" -c
            "import tensorflow as tf; print(tf.__cxx11_abi_flag__)\n"
    OUTPUT_VARIABLE GUESSED_TF_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "GUESSED_TF_ABI: ${GUESSED_TF_ABI}")


execute_process(
    COMMAND "python" -c "import tensorflow as tf; print tf.sysconfig.get_include()"
    OUTPUT_VARIABLE GUESSED_TF_INC
    OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "GUESSED_TF_INC: ${GUESSED_TF_INC}")

# find_path(TensorFlow_INCLUDE_DIR
#         NAMES
#         tensorflow/core
#         tensorflow/cc
#         third_party
#         HINTS
#         ${GUESSED_TF_INC}
#         /home/wieschol/.local/lib/python2.7/site-packages/tensorflow/include
#         /usr/local/include/google/tensorflow
#         /usr/include/google/tensorflow)


execute_process(
    COMMAND "python" -c "import tensorflow as tf; print tf.sysconfig.get_lib()"
    OUTPUT_VARIABLE GUESSED_TF_LIB
    OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "GUESSED_TF_LIB: ${GUESSED_TF_LIB}")


# find_library(TensorFlow_LIBRARY NAMES libtensorflow_framework.so
#         HINTS
#         ${GUESSED_TF_LIB}
#         /home/wieschol/.local/lib/python2.7/site-packages/tensorflow
#         /usr/lib
#         /usr/local/lib)

# set TensorFlow_FOUND
# find_package_handle_standard_args(TensorFlow DEFAULT_MSG TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)
find_package_handle_standard_args(TensorFlow DEFAULT_MSG GUESSED_TF_INC GUESSED_TF_LIB)


# set external variables for usage in CMakeLists.txt
if(TENSORFLOW_FOUND)
    set(TensorFlow_LIBRARIES ${GUESSED_TF_LIB}/libtensorflow_framework.so)
    # set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
    set(TensorFlow_INCLUDE_DIRS ${GUESSED_TF_INC}/include)
    # set(TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR})
    set(TensorFlow_ABI ${GUESSED_TF_ABI})
endif()


# hide locals from GUI
mark_as_advanced(TensorFlow_INCLUDE_DIRS TensorFlow_LIBRARIES TensorFlow_ABI)

