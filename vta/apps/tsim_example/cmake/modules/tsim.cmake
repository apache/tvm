# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(MSVC)
  message(STATUS "TSIM build is skipped in Windows..")
else()
  find_program(PYTHON NAMES python python3 python3.6)
  find_program(VERILATOR NAMES verilator)

  if (VERILATOR AND PYTHON)

    if (TSIM_TOP_NAME STREQUAL "")
      message(FATAL_ERROR "TSIM_TOP_NAME should be defined")
    endif()

    if (TSIM_BUILD_NAME STREQUAL "")
      message(FATAL_ERROR "TSIM_BUILD_NAME should be defined")
    endif()

    set(TSIM_CONFIG ${PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/python/tsim/config.py)

    execute_process(COMMAND ${TSIM_CONFIG} --get-target OUTPUT_VARIABLE __TSIM_TARGET)
    execute_process(COMMAND ${TSIM_CONFIG} --get-top-name OUTPUT_VARIABLE __TSIM_TOP_NAME)
    execute_process(COMMAND ${TSIM_CONFIG} --get-build-name OUTPUT_VARIABLE __TSIM_BUILD_NAME)
    execute_process(COMMAND ${TSIM_CONFIG} --get-use-trace OUTPUT_VARIABLE __TSIM_USE_TRACE)
    execute_process(COMMAND ${TSIM_CONFIG} --get-trace-name OUTPUT_VARIABLE __TSIM_TRACE_NAME)

    string(STRIP ${__TSIM_TARGET} TSIM_TARGET)
    string(STRIP ${__TSIM_TOP_NAME} TSIM_TOP_NAME)
    string(STRIP ${__TSIM_BUILD_NAME} TSIM_BUILD_NAME)
    string(STRIP ${__TSIM_USE_TRACE} TSIM_USE_TRACE)
    string(STRIP ${__TSIM_TRACE_NAME} TSIM_TRACE_NAME)

    set(TSIM_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${TSIM_BUILD_NAME})

    if (TSIM_TARGET STREQUAL "chisel")

      find_program(SBT NAMES sbt)

      if (SBT)

        # Install Chisel VTA package for DPI modules
        set(VTA_CHISEL_DIR ${VTA_DIR}/hardware/chisel)

        execute_process(WORKING_DIRECTORY ${VTA_CHISEL_DIR}
          COMMAND ${SBT} publishLocal RESULT_VARIABLE RETCODE)

        if (NOT RETCODE STREQUAL "0")
          message(FATAL_ERROR "[TSIM] sbt failed to install VTA scala package")
        endif()

        # Chisel - Scala to Verilog compilation
        set(TSIM_CHISEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/hardware/chisel)
        set(CHISEL_TARGET_DIR ${TSIM_BUILD_DIR}/chisel)
        set(CHISEL_OPT "test:runMain test.Elaborate --target-dir ${CHISEL_TARGET_DIR} --top-name ${TSIM_TOP_NAME}")

        execute_process(WORKING_DIRECTORY ${TSIM_CHISEL_DIR} COMMAND ${SBT} ${CHISEL_OPT} RESULT_VARIABLE RETCODE)

        if (NOT RETCODE STREQUAL "0")
          message(FATAL_ERROR "[TSIM] sbt failed to compile from Chisel to Verilog.")
        endif()

        file(GLOB VERILATOR_RTL_SRC ${CHISEL_TARGET_DIR}/*.v)

      else()
        message(FATAL_ERROR "[TSIM] sbt should be installed for Chisel")
      endif() # sbt

    elseif (TSIM_TARGET STREQUAL "verilog")

      set(VTA_VERILOG_DIR ${VTA_DIR}/hardware/chisel/src/main/resources/verilog)
      set(TSIM_VERILOG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/hardware/verilog)
      file(GLOB VERILATOR_RTL_SRC ${VTA_VERILOG_DIR}/*.v ${TSIM_VERILOG_DIR}/*.v)

    else()
      message(STATUS "[TSIM] target language can be only verilog or chisel...")
    endif() # TSIM_TARGET

    if (TSIM_TARGET STREQUAL "chisel" OR TSIM_TARGET STREQUAL "verilog")

      # Check if tracing can be enabled
      if (NOT TSIM_USE_TRACE STREQUAL "OFF")
        message(STATUS "[TSIM] Verilog enable tracing")
      else()
        message(STATUS "[TSIM] Verilator disable tracing")
      endif()

      # Verilator - Verilog to C++ compilation
      set(VERILATOR_TARGET_DIR ${TSIM_BUILD_DIR}/verilator)
      set(VERILATOR_OPT +define+RANDOMIZE_GARBAGE_ASSIGN +define+RANDOMIZE_REG_INIT)
      list(APPEND VERILATOR_OPT +define+RANDOMIZE_MEM_INIT --x-assign unique)
      list(APPEND VERILATOR_OPT --output-split 20000 --output-split-cfuncs 20000)
      list(APPEND VERILATOR_OPT --top-module ${TSIM_TOP_NAME} -Mdir ${VERILATOR_TARGET_DIR})
      list(APPEND VERILATOR_OPT --cc ${VERILATOR_RTL_SRC})

      if (NOT TSIM_USE_TRACE STREQUAL "OFF")
        list(APPEND VERILATOR_OPT --trace)
      endif()

      execute_process(COMMAND ${VERILATOR} ${VERILATOR_OPT} RESULT_VARIABLE RETCODE)

      if (NOT RETCODE STREQUAL "0")
        message(FATAL_ERROR "[TSIM] Verilator failed to compile Verilog to C++...")
      endif()

      # Build shared library (.so)
      set(VTA_HW_DPI_DIR ${VTA_DIR}/hardware/dpi)
      set(VERILATOR_INC_DIR /usr/local/share/verilator/include)
      set(VERILATOR_LIB_SRC ${VERILATOR_INC_DIR}/verilated.cpp ${VERILATOR_INC_DIR}/verilated_dpi.cpp)

      if (NOT TSIM_USE_TRACE STREQUAL "OFF")
        list(APPEND VERILATOR_LIB_SRC ${VERILATOR_INC_DIR}/verilated_vcd_c.cpp)
      endif()

      file(GLOB VERILATOR_GEN_SRC ${VERILATOR_TARGET_DIR}/*.cpp)
      file(GLOB VERILATOR_SRC ${VTA_HW_DPI_DIR}/tsim_device.cc)
      add_library(tsim SHARED ${VERILATOR_LIB_SRC} ${VERILATOR_GEN_SRC} ${VERILATOR_SRC})

      set(VERILATOR_DEF VL_TSIM_NAME=V${TSIM_TOP_NAME} VL_PRINTF=printf VM_COVERAGE=0 VM_SC=0)
      if (NOT TSIM_USE_TRACE STREQUAL "OFF")
        list(APPEND VERILATOR_DEF VM_TRACE=1 TSIM_TRACE_FILE=${TSIM_BUILD_DIR}/${TSIM_TRACE_NAME}.vcd)
      else()
        list(APPEND VERILATOR_DEF VM_TRACE=0)
      endif()
      target_compile_definitions(tsim PRIVATE ${VERILATOR_DEF})
      target_compile_options(tsim PRIVATE -Wno-sign-compare -include V${TSIM_TOP_NAME}.h)
      target_include_directories(tsim PRIVATE ${VERILATOR_TARGET_DIR} ${VERILATOR_INC_DIR} ${VERILATOR_INC_DIR}/vltstd ${VTA_DIR}/include)

      if(APPLE)
        set_target_properties(tsim PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
      endif(APPLE)

    endif() # TSIM_TARGET STREQUAL "chisel" OR TSIM_TARGET STREQUAL "verilog"

  else()
    message(STATUS "[TSIM] could not find Python or Verilator, build is skipped...")
  endif() # VERILATOR
endif() # MSVC
