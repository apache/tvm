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
 * \file src/contrib/msc/core/transform/set_byoc_attrs.cc
 * \brief Pass for fuse ShapeExpr.
 */

#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>

#include "../../../../relax/transform/utils.h"
#include "../utils.h"

namespace tvm {
namespace relax {

using namespace tvm::contrib::msc;

/*!
 * \brief Fuse Tuple and TupleGetItem to BYOC
 */
class ByocNameSetter : public ExprMutator {
 public:
  explicit ByocNameSetter(IRModule ctx_module, const String& target, const String& entry_name)
      : ExprMutator(ctx_module) {
    mod_ = ctx_module;
    target_ = target;
    entry_name_ = entry_name;
  }

  IRModule SetAttrs() {
    GlobalVar main_var;
    size_t func_cnt = 0;
    for (const auto& [gv, func] : mod_->functions) {
      if (gv->name_hint == entry_name_) {
        main_var = gv;
      } else {
        const auto& name_opt = func->GetAttr<runtime::String>(attr::kCodegen);
        if (name_opt.defined() && name_opt.value() == target_) {
          const auto& new_func = WithAttr(Downcast<Function>(func), "byoc_name",
                                          target_ + "_" + std::to_string(func_cnt));
          builder_->UpdateFunction(gv, new_func);
          func_cnt += 1;
        }
      }
    }
    return builder_->GetContextIRModule();
  }

 private:
  IRModule mod_;
  String target_;
  String entry_name_;
  Map<Function, Function> new_funcs_;
};

IRModule SetBYOCAttrs(IRModule mod, const String& target, const String& entry_name) {
  return ByocNameSetter(mod, target, entry_name).SetAttrs();
}

namespace transform {

Pass SetBYOCAttrs(const String& target, const String& entry_name) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relax::SetBYOCAttrs(m, target, entry_name); };
  return CreateModulePass(pass_func, 0, "SetBYOCAttrs", {});
}

TVM_REGISTER_GLOBAL("relax.transform.SetBYOCAttrs").set_body_typed(SetBYOCAttrs);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
