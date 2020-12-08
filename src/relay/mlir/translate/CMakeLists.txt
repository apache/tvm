add_library(RelayMlirTranslate
        relay_mlir_translate.h
        relay_mlir_translate.cc)
target_include_directories(RelayMlirTranslate
        PRIVATE
        ${RELAY_MLIR_SRC_ROOT}
        ${RELAY_MLIR_BIN_ROOT}
        ${RELAY_MLIR_SRC_ROOT})

