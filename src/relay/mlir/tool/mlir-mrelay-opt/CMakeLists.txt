add_executable(mlir-mrelay-opt mlir_mrelay_opt.cpp)
add_dependencies(mlir-mrelay-opt OMRelayOpsInc OMhloOpsInc MhloUtils)

target_include_directories(mlir-mrelay-opt PRIVATE ${RELAY_MLIR_SRC_ROOT})
target_include_directories(mlir-mrelay-opt PRIVATE ${RELAY_MLIR_BIN_ROOT})

target_include_directories(mlir-mrelay-opt PRIVATE ${MLIR_HLO_SRC})
target_include_directories(mlir-mrelay-opt PRIVATE ${MLIR_HLO_SRC}/build)

target_link_libraries(mlir-mrelay-opt
        ${OMLibs}
        ${MLIRLibs})
