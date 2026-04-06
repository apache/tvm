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
 * \file bind_parallel_loops_to_threads.cc
 * \brief Convert ForKind::kParallel loops to GPU thread bindings.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/target/target.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

namespace tvm {
namespace tirx {
namespace {

static bool IsGpuDeviceType(int dev_type) {
  return dev_type == kDLCUDA || dev_type == kDLROCM || dev_type == kDLOpenCL ||
         dev_type == kDLVulkan || dev_type == kDLMetal || dev_type == kDLWebGPU;
}

class ParallelLoopToThreadBindingMutator : public StmtExprMutator {
 public:
  explicit ParallelLoopToThreadBindingMutator(int64_t max_threads_per_block)
      : max_threads_per_block_(max_threads_per_block) {}

 private:
  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tirx::attr::thread_extent || op->attr_key == s_tir::attr::virtual_thread) {
      bool prev = in_thread_env_;
      in_thread_env_ = true;
      Stmt ret = StmtExprMutator::VisitStmt_(op);
      in_thread_env_ = prev;
      return ret;
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt TransformParallelFor(const ForNode* for_node) {
    if (in_thread_env_) {
      return ffi::GetRef<Stmt>(for_node);
    }

    DataType dtype = for_node->loop_var.dtype();
    PrimExpr min = cast(dtype, for_node->min);
    PrimExpr extent = cast(dtype, for_node->extent);
    PrimExpr max_threads = IntImm(dtype, max_threads_per_block_);
    PrimExpr num_blocks = ceildiv(extent, max_threads);

    Var tx_var("threadIdx.x", dtype);
    Var bx_var("blockIdx.x", dtype);
    IterVar tx_iter(Range::FromMinExtent(IntImm(dtype, 0), max_threads), tx_var,
                    IterVarType::kThreadIndex, "threadIdx.x");
    IterVar bx_iter(Range::FromMinExtent(IntImm(dtype, 0), num_blocks), bx_var,
                    IterVarType::kThreadIndex, "blockIdx.x");

    PrimExpr global_idx = cast(dtype, bx_var * max_threads + tx_var);
    PrimExpr mapped_idx = cast(dtype, min + global_idx);
    Stmt mapped_body = Substitute(for_node->body, {{Var(for_node->loop_var), mapped_idx}});
    mapped_body = IfThenElse(global_idx < extent, mapped_body, Evaluate(IntImm(DataType::Int(32), 0)));

    Stmt body_with_tx = AttrStmt(tx_iter, tirx::attr::thread_extent, max_threads, mapped_body);
    Stmt body_with_bx = AttrStmt(bx_iter, tirx::attr::thread_extent, num_blocks, body_with_tx);
    return body_with_bx;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kThreadBinding) {
      bool prev = in_thread_env_;
      in_thread_env_ = true;
      Stmt ret = StmtExprMutator::VisitStmt_(op);
      in_thread_env_ = prev;
      return ret;
    }
    if (op->kind != ForKind::kParallel) {
      return StmtExprMutator::VisitStmt_(op);
    }
    // First mutate inside this loop, then rewrite the current parallel loop.
    For updated = Downcast<For>(StmtExprMutator::VisitStmt_(op));
    return TransformParallelFor(updated.get());
  }

  int64_t max_threads_per_block_;
  bool in_thread_env_{false};
};

}  // namespace

namespace transform {

Pass BindParallelLoopsToThreads() {
  auto pass_func = [](PrimFunc f, IRModule m, PassContext ctx) {
    auto opt_target = f->GetAttr<Target>(tvm::attr::kTarget);
    Target target = opt_target.value_or(Target::Current(/*allow_none=*/true));
    if (!target.defined() || !IsGpuDeviceType(target->GetTargetDeviceType())) {
      return f;
    }

    int64_t max_threads_per_block = 1024;
    if (auto opt_max_threads = target->GetAttr<Integer>("max_num_threads")) {
      max_threads_per_block = opt_max_threads.value()->value;
    }

    PrimFuncNode* n = f.CopyOnWrite();
    n->body = ParallelLoopToThreadBindingMutator(max_threads_per_block)(n->body);
    return f;
  };

  return CreatePrimFuncPass(pass_func, 0, "tirx.BindParallelLoopsToThreads", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.BindParallelLoopsToThreads", BindParallelLoopsToThreads);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm

