
if(NOT USE_TFOP STREQUAL "OFF")

  if ("${TVM_HOME}" STREQUAL "")
    message(FATAL_ERROR "TVM_HOME is not defined")
  else()
    message("Use TVM_HOME=\"${TVM_HOME}\"")
  endif()

  include_directories(${TVM_HOME}/include)
  include_directories(${TVM_HOME}/3rdparty/dlpack/include)
  include_directories(${TVM_HOME}/3rdparty/dmlc-core/include)

  execute_process(COMMAND python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
    OUTPUT_VARIABLE TF_COMPILE_FLAGS_STR
    RESULT_VARIABLE TF_STATUS)
  if (NOT ${TF_STATUS} EQUAL 0)
    message(FATAL_ERROR "Fail to get TensorFlow compile flags")
  endif()

  execute_process(COMMAND python -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"
    OUTPUT_VARIABLE TF_LINK_FLAGS_STR
    RESULT_VARIABLE TF_STATUS)
  if (NOT ${TF_STATUS} EQUAL 0)
    message(FATAL_ERROR "Fail to get TensorFlow link flags")
  endif()

  string(REGEX REPLACE "\n" " " TF_FLAGS "${TF_COMPILE_FLAGS} ${TF_LINK_FLAGS}")
  message("Use TensorFlow flags=\"${TF_FLAGS}\"")
  separate_arguments(TF_COMPILE_FLAGS UNIX_COMMAND ${TF_COMPILE_FLAGS_STR})
  separate_arguments(TF_LINK_FLAGS UNIX_COMMAND ${TF_LINK_FLAGS_STR})


  set(OP_LIBRARY_NAME tvm_dso_op)
  file(GLOB_RECURSE TFTVM_SRCS src/*.cc)
  add_library(${OP_LIBRARY_NAME} SHARED ${TFTVM_SRCS})
  set_target_properties(${OP_LIBRARY_NAME} PROPERTIES PREFIX "")

  set(TFTVM_COMPILE_FLAGS  -O2 -ldl -g)
  set(TFTVM_LINK_FLAGS  -ltvm_runtime -L${TVM_HOME}/build)
  target_compile_options(${OP_LIBRARY_NAME} PUBLIC ${TFTVM_COMPILE_FLAGS} ${TF_COMPILE_FLAGS})
  target_link_options(${OP_LIBRARY_NAME} PUBLIC ${TFTVM_LINK_FLAGS} ${TF_LINK_FLAGS})

endif()

