if(NOT USE_PT_TVMDSOOP STREQUAL "OFF")
  find_package(Python3 COMPONENTS Interpreter Development)
  include_directories(${Python3_INCLUDE_DIRS})

  message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")

  execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.__path__[0].strip())"
    OUTPUT_VARIABLE PT_PATH
    RESULT_VARIABLE PT_STATUS)
  if (NOT ${PT_STATUS} EQUAL 0)
    message(FATAL_ERROR "Fail to get pytorch path")
  endif()

  string(REGEX REPLACE "\n" "" PT_PATH "${PT_PATH}")

  set(PT_COMPILE_FLAGS_STR "-I${PT_PATH}/include -D_GLIBCXX_USE_CXX11_ABI=0")
  set(PT_LINK_FLAGS_STR "-L${PT_PATH}/lib -l:libtorch.so -l:libtorch_python.so")

  if(NOT USE_CUDA STREQUAL "OFF")
    add_definitions(-DPT_TVMDSOOP_ENABLE_GPU)
  endif()


  string(REGEX REPLACE "\n" " " PT_FLAGS "${PT_COMPILE_FLAGS} ${PT_LINK_FLAGS}")
  separate_arguments(PT_COMPILE_FLAGS UNIX_COMMAND ${PT_COMPILE_FLAGS_STR})
  separate_arguments(PT_LINK_FLAGS UNIX_COMMAND ${PT_LINK_FLAGS_STR})


  set(LIBRARY_NAME pt_tvmdsoop)
  file(GLOB_RECURSE PTTVM_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/contrib/torch/**/*.cc)
  add_library(${LIBRARY_NAME} SHARED ${PTTVM_SRCS})
  # add_library(${STATIC_NAME} STATIC ${PTTVM_SRCS})
  # set(PTTVM_LINK_FLAGS -ltvm -ltvm_runtime -L${CMAKE_CURRENT_BINARY_DIR})
  set(PTTVM_LINK_FLAGS -ltvm -L${CMAKE_CURRENT_BINARY_DIR})

  if (NOT BUILD_PT_TVMDSOOP_ONLY STREQUAL "ON")
    add_dependencies(${LIBRARY_NAME} tvm) 
  endif()
  # add_dependencies(${LIBRARY_NAME} tvm)

  target_compile_options(${LIBRARY_NAME} PUBLIC ${PTTVM_COMPILE_FLAGS} ${PT_COMPILE_FLAGS})
  target_link_libraries(${LIBRARY_NAME} PUBLIC ${PTTVM_LINK_FLAGS} ${PT_LINK_FLAGS})
  # target_compile_options(${STATIC_NAME} PUBLIC ${PTTVM_COMPILE_FLAGS} ${PT_COMPILE_FLAGS})
  # target_link_libraries(${STATIC_NAME} PUBLIC ${PTTVM_LINK_FLAGS} ${PT_LINK_FLAGS})

endif()

