/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
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

#include "../../tirx/transform/unify_thread_binding.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>

namespace tvm {
namespace s_tir {
using namespace tvm::tirx;

namespace transform {

Pass UnifyThreadBinding() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = tirx::detail::ThreadBindingUnifier<StmtExprMutator>::Unify(std::move(f->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "s_tir.UnifyThreadBinding", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.UnifyThreadBinding", UnifyThreadBinding);
}

}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
