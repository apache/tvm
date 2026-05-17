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
 * \file src/relax/backend/contrib/xnnpack/codegen.cc
 * \brief Phase 1 XNNPACK Relax external codegen skeleton.
 */

#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/expr.h>

namespace tvm {
namespace relax {
namespace contrib {

ffi::Array<ffi::Module> XNNPACKCompiler(ffi::Array<Function> functions,
                                        ffi::Map<ffi::String, ffi::Any> /*options*/,
                                        ffi::Map<Constant, ffi::String> /*constant_names*/) {
  if (functions.empty()) {
    return {};
  }

  TVM_FFI_THROW(InternalError)
      << "XNNPACK Relax codegen is registered, but Phase 1 does not support any operators. "
      << "Do not annotate Relax functions with Codegen=\"xnnpack\" until operator support is added.";
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("relax.ext.xnnpack", XNNPACKCompiler);
}

}  // namespace contrib
}  // namespace relax
}  // namespace tvm
