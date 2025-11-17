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
 * \file tir/transforms/canonicalize_loop.cc
 * \brief Canonicalize all loops to start from zero and step one.
 */
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tir/function.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <utility>

namespace tvm {
namespace tir {

class LoopCanonicalizer : public StmtExprMutator {
 public:
  LoopCanonicalizer() = default;

 private:
  Stmt VisitStmt_(const ForNode* op) final {
    if (is_zero(op->min) && op->HasTrivialStep()) {
      return StmtExprMutator::VisitStmt_(op);
    }
    arith::Analyzer analyzer;
    const auto* loop_var = op->loop_var.get();
    PrimExpr step = op->step.value_or(make_const(loop_var->dtype, 1));

    // report warning for negative step, since it would be a forever loop
    if (!analyzer.CanProveGreaterEqual(step, 1)) {
      // TODO(tvm): prove dynamic shaped step
      LOG(FATAL) << "Loop step for " << op->loop_var << " may not be positive: " << step;
    }

    new_iter_info_[loop_var] = std::make_pair(step, op->min);
    auto n = CopyOnWrite(op);
    n->body = VisitStmt(op->body);
    n->min = make_zero(loop_var->dtype);
    n->extent = analyzer.Simplify(ceildiv(op->extent, step));
    n->step = std::nullopt;
    new_iter_info_.erase(loop_var);
    return For(n);
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto it = new_iter_info_.find(op);
    if (it != new_iter_info_.end()) {
      const auto& [stride, offset] = it->second;
      return ffi::GetRef<Var>(op) * stride + offset;
    }
    return ffi::GetRef<Var>(op);
  }

  /*! \brief Map iter variable `x` to `x * stride + offset`. */
  std::unordered_map<const VarNode*, std::pair<PrimExpr, PrimExpr>> new_iter_info_;
};

PrimFunc CanonicalizeLoop(PrimFunc func) {
  PrimFuncNode* fptr = func.CopyOnWrite();
  fptr->body = LoopCanonicalizer()(func->body);
  return func;
}

namespace transform {

Pass CanonicalizeLoop() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return CanonicalizeLoop(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.CanonicalizeLoop", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tir.transform.CanonicalizeLoop", CanonicalizeLoop);
}

}  // namespace transform

}  // namespace tir
}  // namespace tvm
