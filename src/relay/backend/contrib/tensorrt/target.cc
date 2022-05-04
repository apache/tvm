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

namespace tvm {
namespace relay {
namespace contrib {

/*!
 * \brief This external codegen target can offload compilation to the TensorRT compiler.
 *  - Patterns: python/tvm/relay/op/contrib/tensorrt.py
 *  - Custom compiler: src/relay/backend/contrib/tensorrt/codegen.cc
 *  - Runtime: src/runtime/contrib/tensorrt/ *.cc
 */
TVM_REGISTER_TARGET_KIND("tensorrt", kDLCUDA)
    .set_attr<Bool>(tvm::attr::kIsExternalCodegen, Bool(true));

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
