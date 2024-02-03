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

/*!
 * \brief Function level pass that applies transformations to all
 *        TIR functions within the module.
 */
class PrimFuncPassNode : public PassNode {
 public:
  /* \brief The pass meta data.*/
  PassInfo pass_info;

  /*! \brief The pass function called on each. */
  runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)> pass_func;

  void VisitAttrs(tvm::AttrVisitor* v) { v->Visit("pass_info", &pass_info); }

  /*!
   * \brief Run a function pass on given pass context.
   *
   * \param mod The module that an optimization pass is applied on.
   * \param pass_ctx The context that an optimization pass executes on.
   *
   * \return Return the updated module.
   */
  IRModule operator()(IRModule mod, const PassContext& pass_ctx) const final;

  /*!
   * \brief Get the pass information/meta data.
   */
  PassInfo Info() const override { return pass_info; }

  static constexpr const char* _type_key = "tir.PrimFuncPass";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrimFuncPassNode, PassNode);
};

class PrimFuncPass : public Pass {
 public:
  /*!
   * \brief The constructor
   * \param pass_func The packed function which implements a pass.
   * \param pass_info The pass info.
   */
  TVM_DLL PrimFuncPass(
      runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)> pass_func,
      PassInfo pass_info);

  TVM_DEFINE_OBJECT_REF_METHODS(PrimFuncPass, Pass, PrimFuncPassNode);
};

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
  ICHECK(mod.defined());
  std::vector<GlobalVar> deleted_list;

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
        deleted_list.push_back(Downcast<GlobalVar>(kv.first));
      }
    }
  }

  // Automatic removal of None.  This uses IRModuleNode::Remove
  // instead of manipulating func_dict directly, to ensure that both
  // the function map and the global_var_map_ are correctly updated.
  for (const auto& gv : deleted_list) {
    mod_ptr->Remove(gv);
  }
  return mod;
}

Pass CreatePrimFuncPass(
    const runtime::TypedPackedFunc<PrimFunc(PrimFunc, IRModule, PassContext)>& pass_func,
    int opt_level, String name, tvm::Array<String> required, bool traceable) {
  PassInfo pass_info = PassInfo(opt_level, name, required, traceable);
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
