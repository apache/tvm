set(LLVM_TARGET_DEFINITIONS relay_ops.td)
relay_mlir_tablegen(relay_ops.h.inc -gen-op-decls "-I${RELAY_MLIR_SRC_ROOT}/pass")
relay_mlir_tablegen(relay_ops.cc.inc -gen-op-defs "-I${RELAY_MLIR_SRC_ROOT}/pass")
set(GEN_DOC_FILE ${CMAKE_BINARY_DIR}/docs/ir/relay_ops.md)
add_public_tablegen_target(OMRelayOpsIncGen)

# Header dependencies target for relay_ops.h
add_custom_target(OMRelayOpsInc
        DEPENDS OMRelayOpsIncGen)

add_library(OMRelayOps
        relay_ops.cc
        relay_ops.h)
target_include_directories(OMRelayOps
        PRIVATE
        ${RELAY_MLIR_SRC_ROOT}
        ${RELAY_MLIR_BIN_ROOT})
add_dependencies(OMRelayOps OMRelayOpsIncGen)
target_link_libraries(OMRelayOps)
    #OMMemoryEffectOpInterface)

add_relay_mlir_dialect_doc(relay relay_ops.td)


## the following configs are for mlir hlo, moved to MHLO.cmake
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
