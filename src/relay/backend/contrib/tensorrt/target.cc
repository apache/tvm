/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/relay/backend/contrib/tensorrt/target.cc
 * \brief Registers the "tensorrt" external codegen TargetKind.
 */

#include <tvm/target/target.h>

#include "./codegen.h"

namespace tvm {
namespace relay {
namespace contrib {
namespace tensorrt {

/*!
 * \brief This external codegen target can offload compilation to the TensorRT compiler.
 *  - Patterns: python/tvm/relay/op/contrib/tensorrt.py
 *  - Custom compiler: src/relay/backend/contrib/tensorrt/codegen.cc
 *  - Runtime: src/runtime/contrib/tensorrt/...
 */
TVM_REGISTER_TARGET_KIND("tensorrt", kDLCUDA)
    .set_attr<runtime::Bool>(tvm::attr::kIsExternalCodegen, runtime::Bool(true))
    .set_attr<tvm::transform::Pass>("RelayToTIR", CompileForTensorRT())
    // A array of three integers given the major, minor, and patch numbers for the supported
    // TensorRT compiler version. If empty will be auto-detected from linked library. Default empty.
    .add_attr_option<Array<runtime::Int>>("tensorrt_version", Array<runtime::Int>())
    // If true, the first tensor dimension for most operators is allowed to be Any and
    // TensorRT will assume it represents a batch dimension only known at inference time.
    // Fewer Relay operators are supported in implicit batch mode. Default true.
    .add_attr_option<runtime::Bool>("use_implicit_batch", runtime::Bool(true))
    // If true, excludes sub-graphs which do not have multiply-accumulate operations, even though
    // TensorRT supports them. ad. This is a simple heuristic to optimize the partitioning between
    // TensorRT and TVM. Not required if using Collage for partitioning. Defalut false.
    .add_attr_option<runtime::Bool>("remove_no_mac_subgraphs", runtime::Bool(false))
    // How many bytes of workspace size to allow each subgraph to use for TensorRT engine creation.
    // Default 1G.
    .add_attr_option<runtime::Int>("max_workspace_size", runtime::Int(1 << 30))
    // If true, allows TensorRT to automatically convert float32 operations to float16. Must also be
    // enabled if any float16 operations are in the model. Note that TensorRT may still choose a
    // higher-precision kernel if it results in overall lower runtime, or if no low-precision
    // implementation exists. Default false.
    .add_attr_option<runtime::Bool>("use_fp16", runtime::Bool(false))
    // If true, allows TensorRT to automatically convert float32 operations to uint8
    // (aka quantized). Default false.
    .add_attr_option<runtime::Bool>("use_uint8", runtime::Bool(false));

}  // namespace tensorrt
}  // namespace contrib
}  // namespace relay
}  // namespace tvm
