# LLVM rules
add_definitions(-DDMLC_USE_FOPEN64=0)

if(NOT USE_LLVM STREQUAL "OFF")
  find_llvm(${USE_LLVM})
  include_directories(${LLVM_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})
  message(STATUS "Build with LLVM " ${LLVM_PACKAGE_VERSION})
  message(STATUS "Set TVM_LLVM_VERSION=" ${TVM_LLVM_VERSION})
  # Set flags that are only needed for LLVM target
  add_definitions(-DTVM_LLVM_VERSION=${TVM_LLVM_VERSION})
  file(GLOB COMPILER_LLVM_SRCS src/codegen/llvm/*.cc)
  list(APPEND TVM_LINKER_LIBS ${LLVM_LIBS})
  list(APPEND COMPILER_SRCS ${COMPILER_LLVM_SRCS})
  if(NOT MSVC)
    set_source_files_properties(${COMPILER_LLVM_SRCS}
      PROPERTIES COMPILE_DEFINITIONS "DMLC_ENABLE_RTTI=0")
    set_source_files_properties(${COMPILER_LLVM_SRCS}
      PROPERTIES COMPILE_FLAGS "-fno-rtti")
  endif()
endif()
