# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(USE_RUST_EXT)
    set(RUST_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rust")
    set(CARGO_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rust/target")

    if(USE_RUST_EXT STREQUAL "STATIC")
        set(COMPILER_EXT_PATH "${CARGO_OUT_DIR}/release/libcompiler_ext.a")
    elseif(USE_RUST_EXT STREQUAL "DYNAMIC")
        set(COMPILER_EXT_PATH "${CARGO_OUT_DIR}/release/libcompiler_ext.so")
    else()
        message(FATAL_ERROR "invalid setting for USE_RUST_EXT, STATIC, DYNAMIC or OFF")
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
