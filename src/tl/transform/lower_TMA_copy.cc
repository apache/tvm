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

/*!
 * \file lower_TMA_copy.cc
 * \brief Lower TMA copy for cuda GPU(sm90+)
 */

#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"
#include "../op/bulk_copy.h"
#include "../runtime/runtime.h"

namespace tvm {
namespace tl {

using namespace tir;

class LowerTMADescriptor : public StmtExprMutator {
 public:
  static PrimFunc Substitute(PrimFunc& f) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    LowerTMADescriptor substituter;
    fptr->body = substituter.VisitStmt(f->body);
    for (auto [call, var] : substituter.desc_map_) {
      // Should allocate 128 bytes for TensorMap on stack
      Call alloc_desc =
          Call(DataType::Handle(), builtin::tvm_stack_alloca(), {StringImm("arg_value"), 16});
      Array<PrimExpr> init_desc_args = {StringImm(tvm_tensormap_create), var};
      init_desc_args.insert(init_desc_args.end(), call->args.begin(), call->args.end());
      Call init_desc = Call(DataType::Handle(), builtin::tvm_call_packed(), init_desc_args);
      fptr->body = LetStmt(var, alloc_desc, SeqStmt({Evaluate(init_desc), fptr->body}));
    }
    return f;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    // Insert the prefetch TMA descriptor statement TO the beginning of the kernel
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        auto body = StmtExprMutator::VisitStmt(op->body);
        if (prefetch_calls_.empty()) {
          return AttrStmt(op->node, op->attr_key, op->value, body);
        } else {
          auto cond = EQ(iv->var, 0);
          auto init_stmt = IfThenElse(cond, SeqStmt(prefetch_calls_));
          prefetch_calls_.clear();
          return AttrStmt(op->node, op->attr_key, op->value, SeqStmt({init_stmt, body}));
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode* call) final {
    if (call->op.same_as(CreateTMADescriptorOp())) {
      Var var;
      auto iter = desc_map_.find(GetRef<Call>(call));
      if (iter != desc_map_.end()) {
        var = iter->second;
      } else {
        String name = call->args[2].as<Var>().value()->name_hint;
        var = Var(name + "_desc", PointerType(PrimType(cuTensorMapType()), "grid_constant"));
        desc_map_[GetRef<Call>(call)] = var;
        prefetch_calls_.push_back(Evaluate(Call(DataType::Handle(), builtin::call_extern(),
                                                {StringImm("tl::prefetch_tma_descriptor"), var})));
      }
      return var;
    } else {
      return StmtExprMutator::VisitExpr_(call);
    }
  }

 private:
  Array<Stmt> prefetch_calls_;
  std::unordered_map<Call, Var, StructuralHash, ExprDeepEqual> desc_map_;
  LowerTMADescriptor() = default;
};

using namespace tir::transform;

tvm::transform::Pass LowerTMADescriptor() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerTMADescriptor::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerTMADescriptor", {});
}

TVM_REGISTER_GLOBAL("tl.LowerTMADescriptor").set_body_typed(LowerTMADescriptor);

}  // namespace tl
}  // namespace tvm
