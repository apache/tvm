if(MSVC)
  message(STATUS "VSIM build is skipped in Windows..")
else()
  find_program(PYTHON NAMES python python3 python3.6)
  find_program(VERILATOR NAMES verilator)

  if (VERILATOR AND PYTHON)

    if (VSIM_TOP_NAME STREQUAL "")
      message(FATAL_ERROR "VSIM_TOP_NAME should be defined")
    endif()

    if (VSIM_BUILD_NAME STREQUAL "")
      message(FATAL_ERROR "VSIM_BUILD_NAME should be defined")
    endif()

    set(VSIM_CONFIG ${PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/python/vsim/config.py)

    execute_process(COMMAND ${VSIM_CONFIG} --get-target OUTPUT_VARIABLE __VSIM_TARGET)
    execute_process(COMMAND ${VSIM_CONFIG} --get-top-name OUTPUT_VARIABLE __VSIM_TOP_NAME)
    execute_process(COMMAND ${VSIM_CONFIG} --get-build-name OUTPUT_VARIABLE __VSIM_BUILD_NAME)
    execute_process(COMMAND ${VSIM_CONFIG} --get-use-trace OUTPUT_VARIABLE __VSIM_USE_TRACE)
    execute_process(COMMAND ${VSIM_CONFIG} --get-trace-name OUTPUT_VARIABLE __VSIM_TRACE_NAME)

    string(STRIP ${__VSIM_TARGET} VSIM_TARGET)
    string(STRIP ${__VSIM_TOP_NAME} VSIM_TOP_NAME)
    string(STRIP ${__VSIM_BUILD_NAME} VSIM_BUILD_NAME)
    string(STRIP ${__VSIM_USE_TRACE} VSIM_USE_TRACE)
    string(STRIP ${__VSIM_TRACE_NAME} VSIM_TRACE_NAME)

    set(VSIM_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${VSIM_BUILD_NAME})

    if (VSIM_TARGET STREQUAL "chisel")

      find_program(SBT NAMES sbt)

      if (SBT)

        # Install Chisel VTA package for DPI modules
        set(VTA_CHISEL_DIR ${VTA_DIR}/hardware/chisel)

        execute_process(WORKING_DIRECTORY ${VTA_CHISEL_DIR}
          COMMAND ${SBT} publishLocal RESULT_VARIABLE RETCODE)

        if (NOT RETCODE STREQUAL "0")
          message(FATAL_ERROR "[VSIM] sbt failed to install VTA scala package")
        endif()

        # Chisel - Scala to Verilog compilation
        set(VSIM_CHISEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/hardware/chisel)
        set(CHISEL_TARGET_DIR ${VSIM_BUILD_DIR}/chisel)
        set(CHISEL_OPT "test:runMain test.Elaborate --target-dir ${CHISEL_TARGET_DIR} --top-name ${VSIM_TOP_NAME}")

        execute_process(WORKING_DIRECTORY ${VSIM_CHISEL_DIR} COMMAND ${SBT} ${CHISEL_OPT} RESULT_VARIABLE RETCODE)

        if (NOT RETCODE STREQUAL "0")
          message(FATAL_ERROR "[VSIM] sbt failed to compile from Chisel to Verilog.")
        endif()

        file(GLOB VERILATOR_RTL_SRC ${CHISEL_TARGET_DIR}/*.v)

      else()
        message(FATAL_ERROR "[VSIM] sbt should be installed for Chisel")
      endif() # sbt

    elseif (VSIM_TARGET STREQUAL "verilog")

      set(VTA_VERILOG_DIR ${VTA_DIR}/hardware/chisel/src/main/resources/verilog)
      set(VSIM_VERILOG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/hardware/verilog)
      file(GLOB VERILATOR_RTL_SRC ${VTA_VERILOG_DIR}/*.v ${VSIM_VERILOG_DIR}/*.v)

    else()
      message(STATUS "[VSIM] target language can be only verilog or chisel...")
    endif() # VSIM_TARGET

    if (VSIM_TARGET STREQUAL "chisel" OR VSIM_TARGET STREQUAL "verilog")

      # Check if tracing can be enabled
      if (NOT VSIM_USE_TRACE STREQUAL "OFF")
        message(STATUS "[VSIM] Verilog enable tracing")
      else()
        message(STATUS "[VSIM] Verilator disable tracing")
      endif()

      # Verilator - Verilog to C++ compilation
      set(VERILATOR_TARGET_DIR ${VSIM_BUILD_DIR}/verilator)
      set(VERILATOR_OPT +define+RANDOMIZE_GARBAGE_ASSIGN +define+RANDOMIZE_REG_INIT)
      list(APPEND VERILATOR_OPT +define+RANDOMIZE_MEM_INIT --x-assign unique)
      list(APPEND VERILATOR_OPT --output-split 20000 --output-split-cfuncs 20000)
      list(APPEND VERILATOR_OPT --top-module ${VSIM_TOP_NAME} -Mdir ${VERILATOR_TARGET_DIR})
      list(APPEND VERILATOR_OPT --cc ${VERILATOR_RTL_SRC})

      if (NOT VSIM_USE_TRACE STREQUAL "OFF")
        list(APPEND VERILATOR_OPT --trace)
      endif()

      execute_process(COMMAND ${VERILATOR} ${VERILATOR_OPT} RESULT_VARIABLE RETCODE)

      if (NOT RETCODE STREQUAL "0")
        message(FATAL_ERROR "[VSIM] Verilator failed to compile Verilog to C++...")
      endif()

      # Build shared library (.so)
      set(VERILATOR_SRC_DIR ${VTA_DIR}/src/verilator)
      set(VERILATOR_INC_DIR /usr/local/share/verilator/include)
      set(VERILATOR_LIB_SRC ${VERILATOR_INC_DIR}/verilated.cpp ${VERILATOR_INC_DIR}/verilated_dpi.cpp)

      if (NOT VSIM_USE_TRACE STREQUAL "OFF")
        list(APPEND VERILATOR_LIB_SRC ${VERILATOR_INC_DIR}/verilated_vcd_c.cpp)
      endif()

      file(GLOB VERILATOR_GEN_SRC ${VERILATOR_TARGET_DIR}/*.cpp)
      file(GLOB VERILATOR_SRC ${VERILATOR_SRC_DIR}/sim.cc)
      add_library(vsim SHARED ${VERILATOR_LIB_SRC} ${VERILATOR_GEN_SRC} ${VERILATOR_SRC})

      set(VERILATOR_DEF VL_VSIM_NAME=V${VSIM_TOP_NAME} VL_PRINTF=printf VM_COVERAGE=0 VM_SC=0)
      if (NOT VSIM_USE_TRACE STREQUAL "OFF")
        list(APPEND VERILATOR_DEF VM_TRACE=1 VSIM_TRACE_FILE=${VSIM_BUILD_DIR}/${VSIM_TRACE_NAME}.vcd)
      else()
        list(APPEND VERILATOR_DEF VM_TRACE=0)
      endif()
      target_compile_definitions(vsim PRIVATE ${VERILATOR_DEF})
      target_compile_options(vsim PRIVATE -Wno-sign-compare -include V${VSIM_TOP_NAME}.h)
      target_include_directories(vsim PRIVATE ${VERILATOR_TARGET_DIR} ${VERILATOR_INC_DIR} ${VERILATOR_INC_DIR}/vltstd ${VTA_DIR}/include)

      if(APPLE)
        set_target_properties(vsim PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
      endif(APPLE)

    endif() # VSIM_TARGET STREQUAL "chisel" OR VSIM_TARGET STREQUAL "verilog"

  else()
    message(STATUS "[VSIM] could not find Python or Verilator, build is skipped...")
  endif() # VERILATOR
endif() # MSVC
