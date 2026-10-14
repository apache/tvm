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

# Core compiler sources, extended by the backend modules.
tvm_file_glob(GLOB_RECURSE COMPILER_SRCS
    src/ir/*.cc
    src/sym/*.cc
    src/te/*.cc
    src/tirx/*.cc
    src/s_tir/*.cc
    src/topi/*.cc
    src/driver/*.cc
    src/support/*.cc
    # Shared printer and TVMScript IR builder. Per-dialect printer/builder pieces
    # live under src/<dialect>/script/. List shared files explicitly so a dialect
    # implementation cannot be silently included through a recursive glob.
    src/script/ir_builder/base.cc
    src/script/ir_builder/ir.cc
    src/script/ir_builder/frame.cc
    src/script/printer/config.cc
    src/script/printer/doc.cc
    src/script/printer/doc_translator.cc
    src/script/printer/ir/call.cc
    src/script/printer/ir/utils.cc
    src/script/printer/ir/ir.cc
    src/script/printer/ir/misc.cc
    src/script/printer/ir/prim_type.cc
    src/script/printer/ir/prim_expr.cc
    src/script/printer/ir/stmt.cc
    src/script/printer/printer.cc
    src/script/printer/doc_printer.cc
    src/script/printer/script_printer.cc
    src/relax/ir/*.cc
    src/relax/op/*.cc
    src/relax/analysis/*.cc
    src/relax/transform/*.cc
    src/relax/backend/vm/*.cc
    src/relax/backend/adreno/*.cc
    src/relax/backend/task_extraction.cc
    src/relax/backend/pattern_registry.cc
    src/relax/utils.cc
    src/relax/distributed/*.cc
    src/relax/distributed/transform/*.cc
    src/relax/op/distributed/*.cc
    src/relax/script/*.cc
    src/relax/testing/*.cc
    )

tvm_file_glob(GLOB CODEGEN_SRCS
  src/target/*.cc
  src/target/source/*.cc
  src/target/canonicalizer/*.cc
  src/target/llvm/canonicalizer/*.cc
  src/backend/cuda/codegen/*.cc
  src/backend/cuda/op/*.cc
  src/backend/cuda/transforms/*.cc
  src/backend/cuda/tile/*.cc
  src/backend/cuda/layout/*.cc
  src/backend/hexagon/codegen/*.cc
  src/backend/metal/codegen/*.cc
  src/backend/metal/op/*.cc
  src/backend/opencl/codegen/*.cc
  src/backend/opencl/op/*.cc
  src/backend/rocm/codegen/*.cc
  src/backend/trn/codegen/*.cc
  src/backend/trn/op/*.cc
  src/backend/trn/transform/*.cc
  src/backend/trn/layout/*.cc
  src/backend/vulkan/codegen/target_kind.cc
  src/backend/vulkan/codegen/vulkan_fallback_module.cc
  src/backend/webgpu/codegen/*.cc
    )

list(APPEND COMPILER_SRCS ${CODEGEN_SRCS})
