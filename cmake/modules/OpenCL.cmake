# OPENCL Module
find_package(OpenCL QUIET)

if(OpenCL_FOUND)
  # always set the includedir when cuda is available
  # avoid global retrigger of cmake
  include_directories(${OpenCL_INCLUDE_DIRS})
endif(OpenCL_FOUND)

if(USE_OPENCL)
  find_package(OpenCL REQUIRED)
  message(STATUS "Build with OpenCL support")
  file(GLOB RUNTIME_OPENCL_SRCS src/runtime/opencl/*.cc)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${OpenCL_LIBRARIES})
  list(APPEND RUNTIME_SRCS ${RUNTIME_OPENCL_SRCS})
  if($ENV{XILINX_SDX})
    include_directories($(XILINX_SDX)/runtime/include/1_2)
    list(APPEND TVM_RUNTIME_LINKER_LIBS $(XILINX_SDX)/runtime/lib/x86_64 $(XILINX_SDX)/lib/lnx64.o)
  endif()
else()
  list(APPEND COMPILER_SRCS src/codegen/opt/build_opencl_off.cc)
endif(USE_OPENCL)
