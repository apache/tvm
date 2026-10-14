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

/*! \file src/codegen/codegen.cc
 * \brief Shared dispatch from IR modules to target code generators.
 */
#include <tvm/codegen/codegen.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace codegen {

ffi::Module Build(IRModule mod, Target target) {
  if (transform::PassContext::Current()->GetConfig<bool>("tirx.disable_assert").value_or(false)) {
    mod = tirx::transform::SkipAssert()(mod);
  }

  // the build function.
  std::string build_f_name = "target.build." + target->kind->name;
  const auto bf = tvm::ffi::Function::GetGlobal(build_f_name);
  TVM_FFI_ICHECK(bf.has_value()) << build_f_name << " is not enabled";
  return (*bf)(mod, target).cast<ffi::Module>();
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("codegen.Build", Build);
}

}  // namespace codegen
}  // namespace tvm
