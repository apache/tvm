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
 * \file tir/ir/transform.cc
 * \brief TIR specific transformation passes.
 */
#include <tvm/node/repr_printer.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tir {
namespace transform {

PrimFuncPass::PrimFuncPass(
    runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)> pass_func,
    PassInfo pass_info) {
  auto n = make_object<PrimFuncPassNode>();
  n->pass_func = std::move(pass_func);
  n->pass_info = std::move(pass_info);
  data_ = std::move(n);
}

// Perform Module -> Module optimizations at the PrimFunc level.
IRModule PrimFuncPassNode::operator()(IRModule mod, const PassContext& pass_ctx) const {
  const PassInfo& pass_info = Info();
  ICHECK(mod.defined());
  pass_ctx.Trace(mod, pass_info, true);
  std::vector<ObjectRef> deleted_list;
  IRModuleNode* mod_ptr = mod.CopyOnWrite();
  auto* func_dict = mod_ptr->functions.CopyOnWrite();
  // directly loop over the underlying dict
  for (auto& kv : *func_dict) {
    // only picks up tir::PrimFunc
    if (kv.second->IsInstance<PrimFuncNode>()) {
      // move out the function so that it is the only copy.
      PrimFunc func = Downcast<PrimFunc>(std::move(kv.second));
      func = pass_func(std::move(func), mod, pass_ctx);
      kv.second = std::move(func);

      if (!kv.second.defined()) {
        deleted_list.push_back(kv.first);
      }
    }
  }

  // automatic removal of None
  for (const auto& gv : deleted_list) {
    func_dict->erase(gv);
  }
  pass_ctx.Trace(mod, pass_info, false);
  return mod;
}

Pass CreatePrimFuncPass(
    const runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)>& pass_func,
    int opt_level, String name, tvm::Array<String> required) {
  PassInfo pass_info = PassInfo(opt_level, name, required);
  return PrimFuncPass(pass_func, pass_info);
}

TVM_REGISTER_NODE_TYPE(PrimFuncPassNode);

TVM_REGISTER_GLOBAL("tir.transform.CreatePrimFuncPass")
    .set_body_typed(
        [](runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)> pass_func,
           PassInfo pass_info) { return PrimFuncPass(pass_func, pass_info); });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<PrimFuncPassNode>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const PrimFuncPassNode*>(ref.get());
      const PassInfo info = node->Info();
      p->stream << "PrimFuncPass(" << info->name << ", opt_level=" << info->opt_level << ")";
    });

}  // namespace transform
}  // namespace tir
}  // namespace tvm
