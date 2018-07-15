# CMake Build rules for VTA
find_program(PYTHON NAMES python python3 python3.6)

if(MSVC)
  message(STATUS "VTA build is skipped in Windows..")
elseif(PYTHON)
  set(VTA_CONFIG ${PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/vta/config/vta_config.py)

  if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/vta_config.json)
    message(STATUS "Use VTA config " ${CMAKE_CURRENT_BINARY_DIR}/vta_config.json)
    set(VTA_CONFIG ${PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/vta/config/vta_config.py
      --use-cfg=${CMAKE_CURRENT_BINARY_DIR}/vta_config.json)
  endif()

  execute_process(COMMAND ${VTA_CONFIG} --target OUTPUT_VARIABLE __vta_target)
  string(STRIP ${__vta_target} VTA_TARGET)

  message(STATUS "Build VTA runtime with target: " ${VTA_TARGET})

  execute_process(COMMAND ${VTA_CONFIG} --defs OUTPUT_VARIABLE __vta_defs)

  string(REGEX MATCHALL "(^| )-D[A-Za-z0-9_=.]*" VTA_DEFINITIONS "${__vta_defs}")

  file(GLOB VTA_RUNTIME_SRCS vta/src/*.cc)
  file(GLOB __vta_target_srcs vta/src/${VTA_TARGET}/*.cc)
  list(APPEND VTA_RUNTIME_SRCS ${__vta_target_srcs})

  add_library(vta SHARED ${VTA_RUNTIME_SRCS})

  target_include_directories(vta PUBLIC vta/include)

  foreach(__def ${VTA_DEFINITIONS})
    string(SUBSTRING ${__def} 3 -1 __strip_def)
    target_compile_definitions(vta PUBLIC ${__strip_def})
  endforeach()

  if(APPLE)
    set_target_properties(vta PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif(APPLE)

  # PYNQ rules
  if(${VTA_TARGET} STREQUAL "pynq")
    find_library(__sds_lib NAMES sds_lib PATHS /usr/lib)
    find_library(__dma_lib NAMES dma PATHS
      "/opt/python3.6/lib/python3.6/site-packages/pynq/drivers/"
      "/opt/python3.6/lib/python3.6/site-packages/pynq/lib/")
    target_link_libraries(vta ${__sds_lib} ${__dma_lib})
  endif()
else()
  message(STATUS "Cannot found python in env, VTA build is skipped..")
endif()
