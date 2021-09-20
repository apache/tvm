if(NOT USE_VSI_NPU STREQUAL "OFF")

if(NOT TIM_VX_INSTALL_DIR OR NOT EXISTS ${TIM_VX_INSTALL_DIR})
message(FATAL_ERROR "TIM_VX_INSTALL_DIR should be set")
endif()

set(OVXLIB_API_ATTR "__attribute__\(\(visibility\(\"default\"\)\)\)")
add_definitions(-DOVXLIB_API=${OVXLIB_API_ATTR})
include_directories(${TIM_VX_INSTALL_DIR}/include)

list(APPEND TVM_LINKER_LIBS ${TIM_VX_INSTALL_DIR}/lib/libtim-vx.so)
list(APPEND TVM_RUNTIME_LINKER_LIBS ${TIM_VX_INSTALL_DIR}/lib/libtim-vx.so)

file(GLOB VSINPU_RUNTIME_CONTRIB_SRC
CONFIGURE_DEPENDS src/runtime/contrib/vsi_npu/vsi_npu_runtime.cc
)

list(APPEND RUNTIME_SRCS ${VSINPU_RUNTIME_CONTRIB_SRC})

file(GLOB COMPILER_VSI_NPU_SRCS
CONFIGURE_DEPENDS src/relay/backend/contrib/vsi_npu/*
)

list(APPEND COMPILER_VSI_NPU_SRCS src/relay/backend/contrib/vsi_npu/op_map/op_setup.cc)

list(APPEND COMPILER_SRCS ${COMPILER_VSI_NPU_SRCS})

list(APPEND RUNTIME_SRCS ${VSINPU_RUNTIME_CONTRIB_SRC})

endif(NOT USE_VSI_NPU STREQUAL "OFF")