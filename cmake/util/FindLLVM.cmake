#######################################################
# Enhanced version of find llvm.
#
# Usage:
#   find_llvm(${USE_LLVM})
#
# - When USE_LLVM=ON, use auto search
# - When USE_LLVM=/path/to/llvm-config, use corresponding config
#
# Provide variables:
# - LLVM_INCLUDE_DIRS
# - LLVM_LIBS
# - LLVM_DEFINITIONS
# - TVM_LLVM_VERISON
#
macro(find_llvm use_llvm)
  set(LLVM_CONFIG ${use_llvm})
  if(LLVM_CONFIG STREQUAL "ON")
    find_package(LLVM REQUIRED CONFIG)
    llvm_map_components_to_libnames(LLVM_LIBS all)
    list(REMOVE_ITEM LLVM_LIBS LTO)
    set(TVM_LLVM_VERSION ${LLVM_VERSION_MAJOR}${LLVM_VERSION_MINOR})
  elseif(NOT LLVM_CONFIG STREQUAL "OFF")
    # use llvm config
    message(STATUS "Use llvm-config=" ${LLVM_CONFIG})
    execute_process(COMMAND ${LLVM_CONFIG} --libfiles
      OUTPUT_VARIABLE __llvm_libfiles)
    execute_process(COMMAND ${LLVM_CONFIG} --system-libs
      OUTPUT_VARIABLE __llvm_system_libs)
    execute_process(COMMAND ${LLVM_CONFIG} --cxxflags
      OUTPUT_VARIABLE __llvm_cxxflags)
    execute_process(COMMAND ${LLVM_CONFIG} --version
      COMMAND cut -b 1,3
      OUTPUT_VARIABLE TVM_LLVM_VERSION)
    # definitions
    string(REGEX MATCHALL "(^| )-D[A-Za-z0-9_]*" LLVM_DEFINITIONS ${__llvm_cxxflags})
    # include dir
    string(REGEX MATCHALL "(^| )-I[A-Za-z0-9_/.\-]*" __llvm_include_flags ${__llvm_cxxflags})
    set(LLVM_INCLUDE_DIRS "")
    foreach(__flag IN ITEMS ${__llvm_include_flags})
      string(REGEX REPLACE "(^| )-I" "" __dir "${__flag}")
      list(APPEND LLVM_INCLUDE_DIRS "${__dir}")
    endforeach()
    message(STATUS ${LLVM_INCLUDE_DIRS})
    # libfiles
    string(STRIP ${__llvm_libfiles} __llvm_libfiles)
    string(STRIP ${__llvm_system_libs} __llvm_system_libs)
    set(LLVM_LIBS "${__llvm_libfiles} ${__llvm_system_libs}")
    separate_arguments(LLVM_LIBS)
    string(STRIP ${TVM_LLVM_VERSION} TVM_LLVM_VERSION)
  endif()
endmacro(find_llvm)
