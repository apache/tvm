# Usage:
#   tvm_source_group(<group> GLOB[_RECURSE] <globbing_expression>)
function(tvm_source_group group)
  cmake_parse_arguments(TVM_SOURCE_GROUP "" "" "GLOB;GLOB_RECURSE" ${ARGN})
  if(TVM_SOURCE_GROUP_GLOB)
    file(GLOB srcs1 ${TVM_SOURCE_GROUP_GLOB})
    source_group(${group} FILES ${srcs1})
  endif()

  if(TVM_SOURCE_GROUP_GLOB_RECURSE)
    file(GLOB_RECURSE srcs2 ${TVM_SOURCE_GROUP_GLOB_RECURSE})
    source_group(${group} FILES ${srcs2})
  endif()
endfunction()
