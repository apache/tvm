cmake_minimum_required(VERSION 3.2)

project(hexagon_api)

include(ExternalProject)

# Required variables:
#   ANDROID_ABI
#   ANDROID_PLATFORM
#   USE_ANDROID_TOOLCHAIN (Android toolchain .cmake file)
#   USE_HEXAGON_ARCH
#   USE_HEXAGON_SDK
#   USE_HEXAGON_TOOLCHAIN (Path to Hexagon toolchain ending with "Tools")
# Optional variable:
#   USE_OUTPUT_BINARY_DIR (Path to copy the output binaries to)
#   USE_HEXAGON_GTEST (Path to Hexagon specific gtest version)

set(TVM_SOURCE_DIR "${CMAKE_SOURCE_DIR}/../..")

if(DEFINED USE_OUTPUT_BINARY_DIR)
  set(HEXAGON_API_BINARY_DIR "${USE_OUTPUT_BINARY_DIR}")
else()
  set(HEXAGON_API_BINARY_DIR "${CMAKE_BINARY_DIR}/hexagon_rpc")
endif()
file(MAKE_DIRECTORY ${HEXAGON_API_BINARY_DIR})

if(DEFINED USE_HEXAGON_GTEST)
  if(EXISTS ${USE_HEXAGON_GTEST})
    message(STATUS "Found Hexagon gtest at ${USE_HEXAGON_GTEST}")
  else()
    message(WARNING "Could not find Hexagon gtest at ${USE_HEXAGON_GTEST}. Disabling Hexagon gtest support.")
    unset(USE_HEXAGON_GTEST)
  endif()
endif()

# Build X86 binaries:
# - tvm_rpc_x86

ExternalProject_Add(x86_tvm_runtime_rpc
  SOURCE_DIR "${TVM_SOURCE_DIR}"
  BUILD_COMMAND $(MAKE) runtime tvm_rpc
  CMAKE_ARGS
    "-DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
    "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
    "-DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}"
    "-DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}"
    "-DUSE_HEXAGON_TOOLCHAIN=${USE_HEXAGON_TOOLCHAIN}"
    "-DCMAKE_CXX_STANDARD=17"
    "-DUSE_LIBBACKTRACE=OFF"
    "-DUSE_RPC=ON"
    "-DUSE_CPP_RPC=ON"
    "-DUSE_HEXAGON=ON"
    "-DUSE_HEXAGON_RPC=ON"
    "-DBUILD_STATIC_RUNTIME=ON"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
)
ExternalProject_Get_Property(x86_tvm_runtime_rpc BINARY_DIR)
ExternalProject_Add_Step(x86_tvm_runtime_rpc copy_rpc_server
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BINARY_DIR}/tvm_rpc
    ${HEXAGON_API_BINARY_DIR}/tvm_rpc_x86
  DEPENDEES install
)

# Build Android binaries:
# - libtvm_runtime.so
# - tvm_rpc_android

ExternalProject_Add(android_tvm_runtime_rpc
  SOURCE_DIR "${TVM_SOURCE_DIR}"
  BUILD_COMMAND $(MAKE) runtime tvm_rpc
  CMAKE_ARGS
    "-DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}"
    "-DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}"
    "-DCMAKE_TOOLCHAIN_FILE=${USE_ANDROID_TOOLCHAIN}"
    "-DANDROID_PLATFORM=${ANDROID_PLATFORM}"
    "-DANDROID_ABI=${ANDROID_ABI}"
    "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
    "-DUSE_HEXAGON_ARCH=${USE_HEXAGON_ARCH}"
    "-DCMAKE_CXX_STANDARD=17"
    "-DUSE_LIBBACKTRACE=OFF"
    "-DUSE_RPC=ON"
    "-DUSE_CPP_RPC=ON"
    "-DUSE_HEXAGON=ON"
    "-DUSE_HEXAGON_RPC=ON"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    "-DUSE_ALTERNATIVE_LINKER=OFF"
    "-DUSE_RANDOM=ON"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
)

ExternalProject_Get_Property(android_tvm_runtime_rpc BINARY_DIR)
ExternalProject_Add_Step(android_tvm_runtime_rpc copy_runtime
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BINARY_DIR}/libtvm_runtime.so
    ${HEXAGON_API_BINARY_DIR}
  DEPENDEES install
)
ExternalProject_Add_Step(android_tvm_runtime_rpc copy_rpc_server
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BINARY_DIR}/tvm_rpc
    ${HEXAGON_API_BINARY_DIR}/tvm_rpc_android
  DEPENDEES install
)


# Build Hexagon binaries:
# - libhexagon_rpc_skel.so
# - libtvm_runtime.a
if(DEFINED USE_HEXAGON_GTEST)
  set(GTEST_FLAG "-DUSE_HEXAGON_GTEST=${USE_HEXAGON_GTEST}")
endif()

ExternalProject_Add(hexagon_tvm_runtime_rpc
  SOURCE_DIR "${TVM_SOURCE_DIR}"
  BUILD_COMMAND $(MAKE) runtime hexagon_rpc_sim
  CMAKE_ARGS
    "-DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}"
    "-DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}"
    "-DCMAKE_C_COMPILER=${USE_HEXAGON_TOOLCHAIN}/bin/hexagon-clang"
    "-DCMAKE_CXX_COMPILER=${USE_HEXAGON_TOOLCHAIN}/bin/hexagon-clang++"
    "-DUSE_HEXAGON_SDK=${USE_HEXAGON_SDK}"
    "-DUSE_HEXAGON_ARCH=${USE_HEXAGON_ARCH}"
    "-DUSE_HEXAGON_EXTERNAL_LIBS=${USE_HEXAGON_EXTERNAL_LIBS}"
    "-DHEXAGON_EXTERNAL_LIBS_SHA=${HEXAGON_EXTERNAL_LIBS_SHA}"
    "-DCMAKE_CXX_STANDARD=17"
    "-DUSE_LIBBACKTRACE=OFF"
    "-DUSE_RPC=OFF"
    "-DUSE_HEXAGON=ON"
    "-DUSE_HEXAGON_RPC=ON"
    "-DBUILD_STATIC_RUNTIME=ON"
    "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    "-DUSE_ALTERNATIVE_LINKER=OFF"
    "-DUSE_CUSTOM_LOGGING=ON"
    "-DUSE_HEXAGON_QHL=ON"
    "-DUSE_RANDOM=ON"
    "${GTEST_FLAG}"
  INSTALL_COMMAND ""
  BUILD_ALWAYS ON
)
ExternalProject_Get_Property(hexagon_tvm_runtime_rpc BINARY_DIR)
ExternalProject_Add_Step(hexagon_tvm_runtime_rpc copy_binaries
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${BINARY_DIR}/libtvm_runtime.a
    ${BINARY_DIR}/libhexagon_rpc_skel.so
    ${BINARY_DIR}/libhexagon_rpc_sim.so
    ${HEXAGON_API_BINARY_DIR}
  DEPENDEES install
)

configure_file("${TVM_SOURCE_DIR}/src/runtime/hexagon/rpc/android_bash.sh.template"
  ${HEXAGON_API_BINARY_DIR} COPYONLY)
