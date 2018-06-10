#######################################################
# Enhanced version of find llvm that allows set of LLVM_CONFIG
# When LLVM_CONFIG_PATH is AUTO,
# it defaults to system find llvm
#
# Usage:
#   find_llvm(LLVM_CONFIG_PATH)
#
# Provide variables:
# - LLVM_INCLUDE_DIRS
# - LLVM_LIBS
# - LLVM_DEFINITIONS
# - LLVM_VERSION_CONCAT
#
macro(find_llvm)
  if(LLVM_CONFIG STREQUAL "")
    find_package(LLVM REQUIRED CONFIG)
    llvm_map_components_to_libnames(LLVM_LIBS all)
    list(REMOVE_ITEM LLVM_LIBS LTO)
    set(TVM_LLVM_VERSION ${LLVM_VERSION_MAJOR}${LLVM_VERSION_MINOR})
  else()
    # use llvm config
    message(STATUS "Use llvm-config=" ${LLVM_CONFIG})
    execute_process(COMMAND ${LLVM_CONFIG} --includedir
      OUTPUT_VARIABLE LLVM_INCLUDE_DIRS)
    execute_process(COMMAND ${LLVM_CONFIG} --libfiles
      OUTPUT_VARIABLE LLVM_LIBS)
    execute_process(COMMAND ${LLVM_CONFIG} --cxxflags
      OUTPUT_VARIABLE __llvm_cxxflags)
    execute_process(COMMAND ${LLVM_CONFIG} --version
      COMMAND cut -b 1,3
      OUTPUT_VARIABLE TVM_LLVM_VERSION)
    string(REGEX MATCHALL "(^| )-D[A-Za-z0-9_]*" LLVM_DEFINITIONS ${__llvm_cxxflags})
    string(STRIP ${LLVM_LIBS} LLVM_LIBS)
    separate_arguments(LLVM_LIBS)
    string(STRIP ${LLVM_INCLUDE_DIRS} LLVM_INCLUDE_DIRS)
    string(STRIP ${TVM_LLVM_VERSION} TVM_LLVM_VERSION)
  endif()
endmacro(find_llvm)
