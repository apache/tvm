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
 * \file lower hopper intrin.cc
 * \brief Lower Hopper intrinsics cuda GPU(sm90+)
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

class LowerHopperIntrin : public StmtExprMutator {
 public:
  static PrimFunc Substitute(PrimFunc& f) {
    PrimFuncNode* fptr = f.CopyOnWrite();
    LowerHopperIntrin substituter;
    fptr->body = substituter.VisitStmt(f->body);
    for (auto [call, var] : substituter.desc_map_) {
      // Should allocate 128 bytes for TensorMap on stack
      Call alloc_desc =
          Call(DataType::Handle(), builtin::tvm_stack_alloca(), {StringImm("arg_value"), 16});
      Array<PrimExpr> init_desc_args;
      if (call->op.same_as(CreateTMADescriptorOp())) {
        init_desc_args.push_back(StringImm(tvm_tensormap_create_tiled));
      } else if (call->op.same_as(CreateTMAIm2ColDescriptorOp())) {
        init_desc_args.push_back(StringImm(tvm_tensormap_create_im2col));
      } else {
        CHECK(0) << call->op;
      }
      init_desc_args.push_back(var);
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
        if (prefetch_calls_.empty() && init_mbarrier_calls_.empty()) {
          return AttrStmt(op->node, op->attr_key, op->value, body);
        } else {
          Array<Stmt> stmt_seq;
          if (!init_mbarrier_calls_.empty()) {
            auto alloc_mbarrier = Evaluate(Call(DataType::Handle(), builtin::create_barriers(),
                                                {static_cast<int>(init_mbarrier_calls_.size())}));
            stmt_seq.push_back(alloc_mbarrier);
          }

          auto stmts = prefetch_calls_;
          stmts.insert(stmts.end(), init_mbarrier_calls_.begin(), init_mbarrier_calls_.end());
          auto init_stmt = IfThenElse(EQ(iv->var, 0), stmts.size() > 1 ? SeqStmt(stmts) : stmts[0]);
          stmt_seq.push_back(init_stmt);
          if (!init_mbarrier_calls_.empty()) {
            Stmt mem_sync = Evaluate(
                Call(DataType::Handle(), builtin::tvm_storage_sync(), {StringImm("shared")}));
            stmt_seq.push_back(mem_sync);
          }
          stmt_seq.push_back(body);

          prefetch_calls_.clear();
          init_mbarrier_calls_.clear();
          return AttrStmt(op->node, op->attr_key, op->value, SeqStmt(stmt_seq));
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  PrimExpr VisitExpr_(const CallNode* call) final {
    if (call->op.same_as(CreateTMADescriptorOp()) ||
        call->op.same_as(CreateTMAIm2ColDescriptorOp())) {
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
    } else if (call->op.same_as(CreateListofMBarrierOp())) {
      ICHECK(init_mbarrier_calls_.size() == 0);
      int num_barriers = static_cast<int>(call->args.size());
      for (int i = 0; i < num_barriers; i++) {
        PrimExpr mbarrier = Call(DataType::Handle(), GetMBarrierOp(), {i});
        init_mbarrier_calls_.push_back(
            Evaluate(Call(DataType::Handle(), builtin::ptx_init_barrier_thread_count(),
                          {mbarrier, call->args[i]})));
      }
      return 0;
    } else if (call->op.same_as(SyncThreadsPartialOp())) {
      int barrier_id = init_mbarrier_calls_.size();
      PrimExpr mbarrier = Call(DataType::Handle(), GetMBarrierOp(), {barrier_id});
      init_mbarrier_calls_.push_back(
          Evaluate(Call(DataType::Handle(), builtin::ptx_init_barrier_thread_count(),
                        {mbarrier, call->args[0]})));
      return Call(DataType::Handle(), SyncThreadsPartialOp(), {mbarrier});
    } else {
      return StmtExprMutator::VisitExpr_(call);
    }
  }

 private:
  Array<Stmt> prefetch_calls_;
  Array<Stmt> init_mbarrier_calls_;
  std::unordered_map<Call, Var, StructuralHash, ExprDeepEqual> desc_map_;
  LowerHopperIntrin() = default;
};

using namespace tir::transform;

tvm::transform::Pass LowerHopperIntrin() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerHopperIntrin::Substitute(f);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerHopperIntrin", {});
}

TVM_REGISTER_GLOBAL("tl.LowerHopperIntrin").set_body_typed(LowerHopperIntrin);

}  // namespace tl
}  // namespace tvm
