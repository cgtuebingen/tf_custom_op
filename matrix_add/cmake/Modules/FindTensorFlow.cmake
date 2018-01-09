# University Tuebingen, 2018
include(FindPackageHandleStandardArgs)
unset(TENSORFLOW_FOUND)


# get TF include path
execute_process(
    COMMAND "python" -c
            "try: import tensorflow as tf; print tf.sysconfig.get_include() \nexcept:pass\n"
    OUTPUT_VARIABLE GUESSED_TF_INC)
message(STATUS "GUESSED_TF_INC: ${GUESSED_TF_INC}")

find_path(TensorFlow_INCLUDE_DIR
        NAMES
        tensorflow/core
        tensorflow/cc
        third_party
        HINTS
        ${GUESSED_TF_INC}
        /usr/local/include/google/tensorflow
        /usr/include/google/tensorflow)


execute_process(
    COMMAND "python" -c
            "import tensorflow as tf; print(tf.sysconfig.get_lib())\n"
    OUTPUT_VARIABLE GUESSED_TF_LIB)
message(STATUS "GUESSED_TF_LIB: ${GUESSED_TF_LIB}")


find_library(TensorFlow_LIBRARY NAMES libtensorflow_framework.so
        HINTS
        ${GUESSED_TF_LIB}
        /usr/lib
        /usr/local/lib)

# set TensorFlow_FOUND
find_package_handle_standard_args(TensorFlow DEFAULT_MSG TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)


# set external variables for usage in CMakeLists.txt
if(TENSORFLOW_FOUND)
    set(TensorFlow_LIBRARIES ${TensorFlow_LIBRARY})
    message(STATUS "TensorFlow_LIBRARIES: ${TensorFlow_LIBRARY}")
    set(TensorFlow_INCLUDE_DIRS ${TensorFlow_INCLUDE_DIR})
    message(STATUS "TensorFlow_INCLUDE_DIRS: ${TensorFlow_INCLUDE_DIR}")
endif()


# hide locals from GUI
mark_as_advanced(TensorFlow_INCLUDE_DIR TensorFlow_LIBRARY)

