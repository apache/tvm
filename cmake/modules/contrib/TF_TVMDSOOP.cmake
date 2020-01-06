
if(NOT USE_TF_TVMDSOOP STREQUAL "OFF")

  # If want build this directly comment out below lines.
  # if ("${TVM_HOME}" STREQUAL "")
  #   message(FATAL_ERROR "TVM_HOME is not defined")
  # else()
  #  message("Use TVM_HOME=\"${TVM_HOME}\"")
  #endif()
  # include_directories(${TVM_HOME}/include)
  # include_directories(${TVM_HOME}/3rdparty/dlpack/include)
  # include_directories(${TVM_HOME}/3rdparty/dmlc-core/include)
  # set(TFTVM_LINK_FLAGS  -ltvm_runtime -L${TVM_HOME}/build)

  find_package(Python COMPONENTS Interpreter)
  
  execute_process(COMMAND ${Python_EXECUTABLE} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_compile_flags()))"
    OUTPUT_VARIABLE TF_COMPILE_FLAGS_STR
    RESULT_VARIABLE TF_STATUS)
  if (NOT ${TF_STATUS} EQUAL 0)
    message(FATAL_ERROR "Fail to get TensorFlow compile flags")
  endif()

  execute_process(COMMAND ${Python_EXECUTABLE} -c "import tensorflow as tf; print(' '.join(tf.sysconfig.get_link_flags()))"
    OUTPUT_VARIABLE TF_LINK_FLAGS_STR
    RESULT_VARIABLE TF_STATUS)
  if (NOT ${TF_STATUS} EQUAL 0)
    message(FATAL_ERROR "Fail to get TensorFlow link flags")
  endif()

  string(REGEX REPLACE "\n" " " TF_FLAGS "${TF_COMPILE_FLAGS} ${TF_LINK_FLAGS}")
  separate_arguments(TF_COMPILE_FLAGS UNIX_COMMAND ${TF_COMPILE_FLAGS_STR})
  separate_arguments(TF_LINK_FLAGS UNIX_COMMAND ${TF_LINK_FLAGS_STR})


  set(OP_LIBRARY_NAME tvm_dso_op)
  file(GLOB_RECURSE TFTVM_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/contrib/tf_op/*.cc)
  add_library(${OP_LIBRARY_NAME} SHARED ${TFTVM_SRCS})
  set_target_properties(${OP_LIBRARY_NAME} PROPERTIES PREFIX "")
  set(TFTVM_LINK_FLAGS  -ltvm -L${CMAKE_CURRENT_BINARY_DIR})
  add_dependencies(${OP_LIBRARY_NAME} tvm) 

  # set(TFTVM_COMPILE_FLAGS  ${CMAKE_CXX_FLAGS})
  target_compile_options(${OP_LIBRARY_NAME} PUBLIC ${TFTVM_COMPILE_FLAGS} ${TF_COMPILE_FLAGS})
  target_link_options(${OP_LIBRARY_NAME} PUBLIC ${TFTVM_LINK_FLAGS} ${TF_LINK_FLAGS})

endif()

