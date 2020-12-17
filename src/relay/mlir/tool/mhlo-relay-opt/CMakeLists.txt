add_library(RelayMlirTranslateRegistration
        relay_mlir_translate_registration.cc)
target_include_directories(RelayMlirTranslateRegistration
        PRIVATE
        ${RELAY_MLIR_SRC_ROOT}
        ${RELAY_MLIR_BIN_ROOT}
        ${RELAY_MLIR_SRC_ROOT})

# build opt tool
add_executable(relay-mlir-translate
        relay_mlir_translate_main.cc
        relay_mlir_translate_registration.cc)
add_dependencies(relay-mlir-translate
        RelayMlirTranslate
        RelayMlirTranslateRegistration
        MhloUtils)
target_include_directories(relay-mlir-translate
        PRIVATE
        ${RELAY_MLIR_SRC_ROOT}
        ${RELAY_MLIR_BIN_ROOT}
        ${RELAY_MLIR_SRC_ROOT})
target_link_libraries(relay-mlir-translate
        ${OMLibs}
        ${MLIRLibs}
        # ${TVMLibs}
        ${TVM_LINK_FLAGS})
