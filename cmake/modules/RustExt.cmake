if(USE_RUST_EXT)
    set(RUST_SRC_DIR "rust")
    set(CARGO_OUT_DIR "rust/target"
    set(COMPILER_EXT_PATH "${CARGO_OUT_DIR}/target/release/libcompiler_ext.dylib")

    add_custom_command(
        OUTPUT "${COMPILER_EXT_PATH}"
        COMMAND cargo build --release
        MAIN_DEPENDENCY "${RUST_SRC_DIR}"
        WORKING_DIRECTORY "${RUST_SRC_DIR}/compiler-ext")

    target_link_libraries(tvm "${COMPILER_EXT_PATH}" PRIVATE)
endif(USE_RUST_EXT)
