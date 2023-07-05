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
 * \file src/relay/backend/contrib/tensorrt/codegen.h
 * \brief The 'custom' compilation pass for TensorRT (invoked by the RelayToTIRTargetHook pass).
 */

#ifndef TVM_RELAY_BACKEND_CONTRIB_TENSORRT_CODEGEN_H_
#define TVM_RELAY_BACKEND_CONTRIB_TENSORRT_CODEGEN_H_

#include <tvm/ir/transform.h>

namespace tvm {
namespace relay {
namespace contrib {
namespace tensorrt {

/*!
 * \brief Returns the pass which replaces all calls to "Primitive" functions with a "Compiler"
 * attribute of "tensorrt" with calls to an extern which is implemented by a \p TensorRTRuntime
 * runtime module added to the IRModule's "external_mods" attribute.
 */
transform::Pass CompileForTensorRT();

}  // namespace tensorrt
}  // namespace contrib
}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_BACKEND_CONTRIB_TENSORRT_CODEGEN_H_
