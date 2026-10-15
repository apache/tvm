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

# The embedding project may explicitly install the reusable targets as well.
function(tvm_tirx_install)
  get_target_property(PROJECT_SOURCE_DIR tvm_tirx SOURCE_DIR)
  get_target_property(PROJECT_BINARY_DIR tvm_tirx BINARY_DIR)
  foreach(_variable _tirx_targets TVM_TIRX_VERSION TVM_TIRX_USE_CUDA TVM_TIRX_USE_ROCM
      TVM_TIRX_USE_THREADS TVM_FFI_USE_THREADS TVM_FFI_USE_LIBBACKTRACE BUILD_FOR_ANDROID)
    get_target_property(${_variable} tvm_tirx TVM_TIRX_INSTALL_${_variable})
  endforeach()
  include(GNUInstallDirs)
  if(TVM_TIRX_BUILD_PYTHON_MODULE)
    set(CMAKE_INSTALL_LIBDIR lib)
  endif()

  # Public headers only. Private implementation and compiler-rt headers stay in source.
  foreach(_include_root "${PROJECT_SOURCE_DIR}/include"
      "${PROJECT_SOURCE_DIR}/3rdparty/tvm-ffi/include"
      "${PROJECT_SOURCE_DIR}/3rdparty/tvm-ffi/3rdparty/dlpack/include")
    file(GLOB_RECURSE _headers CONFIGURE_DEPENDS RELATIVE "${_include_root}" "${_include_root}/*.h")
    foreach(_header IN LISTS _headers)
      get_filename_component(_directory "${_header}" DIRECTORY)
      install(FILES "${_include_root}/${_header}" DESTINATION "include/${_directory}")
    endforeach()
  endforeach()

  install(TARGETS ${_tirx_targets} EXPORT tvm_tirxTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})

  # Export the bundled FFI targets, so native consumers use the same dependency
  # without Python or a separately installed FFI CMake package.
  foreach(_kind header shared static)
    set_target_properties(tvm_ffi_${_kind} PROPERTIES EXPORT_NAME ${_kind})
  endforeach()
  set(TVM_TIRX_FFI_HAS_BACKTRACE OFF)
  if(TVM_FFI_USE_LIBBACKTRACE AND NOT MSVC)
    set(TVM_TIRX_FFI_HAS_BACKTRACE ON)
    get_target_property(_ffi_static_links tvm_ffi_static INTERFACE_LINK_LIBRARIES)
    string(REPLACE "$<LINK_ONLY:libbacktrace>"
      "$<BUILD_INTERFACE:$<LINK_ONLY:libbacktrace>>;$<INSTALL_INTERFACE:$<LINK_ONLY:tvm_ffi::backtrace>>"
      _ffi_static_links "${_ffi_static_links}")
    set_target_properties(tvm_ffi_static PROPERTIES INTERFACE_LINK_LIBRARIES "${_ffi_static_links}")
    install(FILES "${PROJECT_BINARY_DIR}/3rdparty/tvm-ffi/libbacktrace/lib/libbacktrace.a"
      DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()
  install(TARGETS tvm_ffi_header tvm_ffi_shared tvm_ffi_static EXPORT tvm_tirxFFITargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})

  include(CMakePackageConfigHelpers)
  configure_package_config_file("${PROJECT_SOURCE_DIR}/cmake/tvm_tirxConfig.cmake.in"
    "${PROJECT_BINARY_DIR}/tvm_tirxConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tvm_tirx"
    PATH_VARS CMAKE_INSTALL_LIBDIR)
  install(EXPORT tvm_tirxFFITargets NAMESPACE tvm_ffi:: DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tvm_tirx")
  install(EXPORT tvm_tirxTargets NAMESPACE tvm_tirx:: DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tvm_tirx")
  install(FILES "${PROJECT_BINARY_DIR}/tvm_tirxConfig.cmake" DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tvm_tirx")
  # Find helpers are used by the parent package's optional runtime integrations.
  install(FILES "${PROJECT_SOURCE_DIR}/3rdparty/tvm-ffi/cmake/Utils/Library.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tvm_tirx/utils/tvm_ffi")
  install(DIRECTORY "${PROJECT_SOURCE_DIR}/cmake/utils/" DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/tvm_tirx/utils"
    FILES_MATCHING PATTERN "Find*.cmake")

  if(TVM_TIRX_BUILD_PYTHON_MODULE)
    foreach(_extra_lib IN LISTS TVM_TIRX_PACKAGE_EXTRA_LIBS)
      if(NOT EXISTS "${_extra_lib}")
        message(FATAL_ERROR "TVM_TIRX_PACKAGE_EXTRA_LIBS entry does not exist: ${_extra_lib}")
      endif()
      install(FILES "${_extra_lib}" DESTINATION lib)
    endforeach()
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/licenses/" DESTINATION licenses)
    install(FILES "${PROJECT_SOURCE_DIR}/LICENSE" "${PROJECT_SOURCE_DIR}/NOTICE" DESTINATION .)
    install(FILES "${PROJECT_SOURCE_DIR}/3rdparty/tvm-ffi/LICENSE"
      DESTINATION licenses RENAME LICENSE-tvm-ffi.txt)
  endif()
endfunction()
