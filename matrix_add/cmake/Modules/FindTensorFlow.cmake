include(FindPackageHandleStandardArgs)
unset(TENSORFLOW_FOUND)

find_path(TensorFlow_INCLUDE_DIR
        NAMES
        tensorflow/core
        tensorflow/cc
        third_party
        HINTS
        /home/wieschol/.local/lib/python2.7/site-packages/tensorflow/include
        /usr/local/include/google/tensorflow
        /usr/include/google/tensorflow)

find_library(TensorFlow_LIBRARY NAMES libtensorflow_framework.so
        HINTS
        /usr/lib
        /home/wieschol/.local/lib/python2.7/site-packages/tensorflow
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

# Locates the tensorFlow library and include directories.

# # get TF include path
# execute_process(
#     COMMAND "python" -c
#             "try: import tensorflow as tf; print tf.sysconfig.get_include() \nexcept:pass\n"
#     OUTPUT_VARIABLE TF_INC)
# message(STATUS "TF_INC: ${TF_INC}")

# execute_process(
#     COMMAND "python" -c
#             "import tensorflow as tf; print(tf.sysconfig.get_lib())\n"
#     OUTPUT_VARIABLE TF_LIB)
# message(STATUS "TF_LIB: ${TF_LIB}")
