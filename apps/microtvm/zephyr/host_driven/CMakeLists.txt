# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.13.1)

set(ENV{QEMU_BIN_PATH} "${CMAKE_SOURCE_DIR}/qemu-hack")

set(QEMU_PIPE "\${QEMU_PIPE}")  # QEMU_PIPE is set by the calling TVM instance.

find_package(Zephyr HINTS $ENV{ZEPHYR_BASE})
project(microtvm_zephyr_runtime)

set(CMAKE_VERBOSE_MAKEFILE ON)

target_sources(app PRIVATE src/main.c)

foreach(tvm_lib ${TVM_LIBS})
  string(LENGTH ${tvm_lib} tvm_lib_length)
  math(EXPR tvm_lib_cut "${tvm_lib_length} - 2")
  string(SUBSTRING ${tvm_lib} 3 ${tvm_lib_cut} tvm_lib_name)
  add_library(${tvm_lib_name} STATIC IMPORTED)
  set_target_properties(${tvm_lib_name} PROPERTIES
      IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/${tvm_lib})
  target_link_libraries(app PRIVATE ${tvm_lib_name})
endforeach(tvm_lib ${TVM_LIBS})

target_include_directories(app PRIVATE ${TVM_INCLUDE_DIRS})
