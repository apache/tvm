# Path to mlir-hlo source folder.

if(DEFINED ENV{MLIR_HLO_SRC})
  set(MLIR_HLO_SRC $ENV{MLIR_HLO_SRC})
    if(EXISTS ${MLIR_HLO_SRC})
        message(STATUS "MLIR_HLO_SRC" ${MLIR_HLO_SRC})
  else()
      message(FATAL_ERROR "The path specified by MLIR_HLO_SRC does not exist: "
          ${MLIR_HLO_SRC})
  endif()
else()
    message(FATAL_ERROR "ENV variable MLIR_HLO_SRC not set")
endif()

# Path to mlir-hlo build folder
set(MLIR_HLO_BUILD "${MLIR_HLO_SRC}/build")
if(EXISTS ${MLIR_HLO_BUILD})
    message(STATUS "MLIR_HLO_BUILD " ${MLIR_HLO_BUILD})
else()
    message(FATAL_ERROR "Expecting MLIR_HLO_BUILD to exist: " ${MLIR_HLO_BUILD})
endif()

include_directories("${MLIR_HLO_SRC}/include")
include_directories("${MLIR_HLO_SRC}/lib")
include_directories("${MLIR_HLO_BUILD}/include")
include_directories("${MLIR_HLO_BUILD}/lib")

set(MLIR_HLO_LINK_FLAGS -lhlo -L${MLIR_HLO_SRC}/build)



# the following configs are for mlir hlo, moved to MHLO.cmake
# config for mhlo
set(LLVM_TARGET_DEFINITIONS ${MLIR_HLO_SRC}/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.td)
relay_mlir_tablegen(hlo_ops.h.inc -gen-op-decls "-I${MLIR_HLO_SRC}/pass")
relay_mlir_tablegen(hlo_ops.cc.inc -gen-op-defs "-I${MLIR_HLO_SRC}/pass")
set(GEN_DOC_FILE ${CMAKE_BINARY_DIR}/docs/ir/hlo.md)
add_public_tablegen_target(OMhloOpsIncGen)

# Header dependencies target for hlo_ops.h
add_custom_target(OMhloOpsInc
        DEPENDS
        OMhloOpsIncGen)

include_directories(${MLIR_HLO_BUILD}/include/mlir-hlo/utils/)
include_directories(${MLIR_HLO_BUILD}/include/mlir-hlo/Dialect/mhlo/IR/)
include_directories(${MLIR_HLO_BUILD}/include/mlir-hlo/Dialect/mhlo/transforms/)
include_directories(${MLIR_HLO_BUILD}/lib/utils/)
include_directories(${MLIR_HLO_BUILD}/lib/Dialect/mhlo/IR/)
include_directories(${MLIR_HLO_BUILD}/lib/Dialect/mhlo/transforms/)

add_library(MhloUtils
        ${MLIR_HLO_SRC}/lib/utils/broadcast_utils.cc
        ${MLIR_HLO_SRC}/lib/utils/convert_op_folder.cc
        ${MLIR_HLO_SRC}/lib/utils/cycle_detector.cc
        ${MLIR_HLO_SRC}/lib/utils/hlo_utils.cc
        )

add_library(OMhloOps
        ${MLIR_HLO_SRC}/lib/Dialect/mhlo/IR/hlo_ops.cc
        ${MLIR_HLO_SRC}/lib/Dialect/mhlo/IR/hlo_ops_base_structs.cc
        ${MLIR_HLO_SRC}/lib/Dialect/mhlo/IR/lhlo_ops.cc
        ${MLIR_HLO_SRC}/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h
        ${MLIR_HLO_SRC}/include/mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h
        ${MLIR_HLO_SRC}/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops_base_structs.h
        )
target_include_directories(OMhloOps
        PRIVATE
        ${MLIR_HLO_SRC}
        ${MLIR_HLO_BUILD})
add_dependencies(OMhloOps OMhloOpsIncGen)
target_link_libraries(OMhloOps)

add_relay_mlir_dialect_doc(hlo hlo_ops.td)


# All mlir relay libraries.
set(OMLibs
        OMhloOps
        RelayMlirTranslate
        RelayMlirTranslateRegistration
        MhloUtils)

# set(OMLibs ${OMLibs} PARENT_SCOPE)

