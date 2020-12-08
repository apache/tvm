add_library(RelayMlirTransform
        mhlo_to_mrelay.h
        mhlo_to_mrelay.cc)
target_include_directories(RelayMlirTransform
        PRIVATE
        ${RELAY_MLIR_SRC_ROOT}
        ${RELAY_MLIR_BIN_ROOT})