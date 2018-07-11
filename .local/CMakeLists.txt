# University Tuebingen, 2018
cmake_minimum_required( VERSION 2.8 )
# set(LIBFOO_TAR_HEADERS
#   "${CMAKE_CURRENT_BINARY_DIR}/include/foo/foo.h"
#   "${CMAKE_CURRENT_BINARY_DIR}/include/foo/foo_utils.h"
# )

set(LIBFOO_TAR_HEADERS "ff")
add_custom_command(OUTPUT ${LIBFOO_TAR_HEADERS}
  COMMAND python -c "import tensorflow as tf; print tf.sysconfig.get_include()"
  VERBATIM
)

message(STATUS ${LIBFOO_TAR_HEADERS})
