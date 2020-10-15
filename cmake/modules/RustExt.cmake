if(USE_RUST_EXT AND NOT USE_RUST_EXT EQUAL OFF)
    set(RUST_SRC_DIR "${CMAKE_SOURCE_DIR}/rust")
    set(CARGO_OUT_DIR "${CMAKE_SOURCE_DIR}/rust/target")

    if(USE_RUST_EXT STREQUAL "STATIC")
        set(COMPILER_EXT_PATH "${CARGO_OUT_DIR}/release/libcompiler_ext.a")
    elseif(USE_RUST_EXT STREQUAL "DYNAMIC")
        set(COMPILER_EXT_PATH "${CARGO_OUT_DIR}/release/libcompiler_ext.so")
    else()
        message(FATAL_ERROR "invalid setting for RUST_EXT")
    endif()

    add_custom_command(
        OUTPUT "${COMPILER_EXT_PATH}"
        COMMAND cargo build --release
        MAIN_DEPENDENCY "${RUST_SRC_DIR}"
        WORKING_DIRECTORY "${RUST_SRC_DIR}/compiler-ext")

    add_custom_target(rust_ext ALL DEPENDS "${COMPILER_EXT_PATH}")

    # TODO(@jroesch, @tkonolige): move this to CMake target
    # target_link_libraries(tvm "${COMPILER_EXT_PATH}" PRIVATE)
    list(APPEND TVM_LINKER_LIBS ${COMPILER_EXT_PATH})

    add_definitions(-DRUST_COMPILER_EXT=1)
endif()
