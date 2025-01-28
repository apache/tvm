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
 * \file src/relax/transform/attach_attr_layout_free_buffers.cc
 * \brief Attach layout_free_buffers for layout-free buffers.
 */

#include <tvm/relax/expr_functor.h>
#include <tvm/relax/transform.h>
#include <tvm/tir/stmt_functor.h>

namespace tvm {
namespace relax {

class AttrAttacher : public ExprMutator {
 public:
  static IRModule Transform(const IRModule& mod) {
    AttrAttacher mutator(mod);
    for (auto [gvar, func] : mod->functions) {
      if (func->IsInstance<relax::FunctionNode>()) {
        // clear the layout_free_exprs_ for each function
        mutator.layout_free_exprs_.clear();
        mutator.builder_->UpdateFunction(gvar, Downcast<BaseFunc>(mutator.VisitExpr(func)));
      }
    }
    return mutator.builder_->GetContextIRModule();
  }

 private:
  explicit AttrAttacher(IRModule mod) : ExprMutator(mod), mod_(mod) {}

  using ExprMutator::VisitExpr_;
  Expr VisitExpr_(const FunctionNode* op) final {
    if (auto opt_num_input = op->attrs.GetAttr<Integer>(attr::kNumInput)) {
      ICHECK(layout_free_exprs_.empty()) << "meet a non-global function with num_input attr";
      size_t num_input = opt_num_input.value()->value;
      for (size_t i = num_input; i < op->params.size(); i++) {
        layout_free_exprs_.insert(op->params[i].get());
      }
    }
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const ConstantNode* op) final {
    layout_free_exprs_.insert(op);
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const CallNode* op) final {
    static const Op& call_tir_op_ = Op::Get("relax.call_tir");
    Call call = Downcast<Call>(ExprMutator::VisitExpr_(op));
    if (call->op != call_tir_op_) {
      return call;
    }
    GlobalVar gv = Downcast<GlobalVar>(call->args[0]);
    Array<Expr> call_tir_args = Downcast<Tuple>(call->args[1])->fields;
    // Compute the layout free buffers
    Array<Integer> layout_free_buffers;
    for (size_t i = 0; i < call_tir_args.size(); i++) {
      if (layout_free_exprs_.count(call_tir_args[i].get())) {
        layout_free_buffers.push_back(Integer(i));
      }
    }
    // Attach the layout free buffers to the tir::PrimFunc
    tir::PrimFunc func = WithAttr(Downcast<tir::PrimFunc>(mod_->Lookup(gv)), "layout_free_buffers",
                                  layout_free_buffers);
    // Renew defs
    func = tir::RenewDefs(func);
    // Add the updated tir::PrimFunc in the IRModule
    // Note the blockbuilder would automatically combine the same tir function
    // So we don't need to worry about the duplicate insertion
    GlobalVar new_gv = builder_->AddFunction(func, gv->name_hint);
    // Create a new call node with the updated tir::PrimFunc
    auto n = make_object<CallNode>(*op);
    n->args = {new_gv, Tuple(call_tir_args)};
    return Call(n);
  }

 private:
  IRModule mod_;
  std::unordered_set<const ExprNode*> layout_free_exprs_;
};
namespace transform {

Pass AttachAttrLayoutFreeBuffers() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule mod, PassContext pc) { return AttrAttacher::Transform(mod); };
  auto pass = CreateModulePass(pass_func, 0, "_AttachAttrLayoutFreeBuffers", {});
  // Apply DeadCodeElimination to remove unused tir::PrimFunc
  return tvm::transform::Sequential({pass, DeadCodeElimination()}, "AttachAttrLayoutFreeBuffers");
}

TVM_REGISTER_GLOBAL("relax.transform.AttachAttrLayoutFreeBuffers")
    .set_body_typed(AttachAttrLayoutFreeBuffers);
}  // namespace transform
}  // namespace relax
}  // namespace tvm
