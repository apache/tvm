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
 * \file layout_inference.cc
 * \brief infer the fragment/shared memory layout
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include <queue>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../op/parallel.h"
#include "loop_partition.h"
#include "loop_vectorize.h"
#include "common/loop_fusion_utils.h"

namespace tvm {
namespace tl {

using namespace tir;
using arith::IRMutatorWithAnalyzer;

struct LayoutInferenceResult {
  Map<Buffer, Layout> layout_map;
  Map<For, Fragment> for_map;
  Map<For, PrimExpr> predicate_map;
};

class BufferUseDefCollector : public StmtExprVisitor {
 public:
  BufferUseDefCollector() = default;

  LayoutInferenceResult Run() {
    Map<Buffer, Layout> layout_map = annotated_layout_map_;
    int num_infer = infer_list_.size();

    // maintain a bfs queue and infer common layout
    std::queue<int> q;
    std::vector<bool> in_queue(num_infer, true);
    for (int i = 0; i < num_infer; i++) q.push(i);

    auto run_infer_step = [&](int cur_infer_id, InferLevel level, bool update_queue) {
      auto& next = infer_list_[cur_infer_id];
      auto iter_var = thread_var_vec_[cur_infer_id];
      auto updates = next->InferLayout(
          LayoutInferArgs{target_, static_cast<size_t>(*as_const_int(iter_var->dom->extent)),
                          layout_map},
          level);
      for (const auto& [buffer, layout] : updates) {
        if (layout_map.count(buffer)) {
          ICHECK(StructuralEqual()(layout, layout_map[buffer]))
              << "Get different layout for " << buffer;
        } else {
          layout_map.Set(buffer, layout);
          if (!update_queue) continue;
          for (int idx : use_list_[buffer]) {
            if (!in_queue[idx] && idx != cur_infer_id) {
              in_queue[idx] = true;
              q.push(idx);
            }
          }
        }
      }
    };
    auto finish_infer_queue = [&]() {
      while (!q.empty()) {
        int cur_infer_id = q.front();
        q.pop();
        in_queue[cur_infer_id] = false;
        run_infer_step(cur_infer_id, InferLevel::kCommon, true);
      }
    };

    // step 1, infer strict layout
    for (int i = 0; i < num_infer; i++) {
      run_infer_step(i, InferLevel::kStrict, false);
    }

    // step2, infer common layout with bfs
    finish_infer_queue();

    // step 3, relax the infer constraint to free and rerun.
    for (int i = 0; i < num_infer; i++) {
      run_infer_step(i, InferLevel::kFree, true);
      finish_infer_queue();
    }

    // Check that all fragments have been inferred
    for (const auto& [buffer, _] : use_list_) {
      if (buffer.scope() == "local.fragment" && layout_map.count(buffer) == 0)
        LOG_ERROR << "The layout for fragment " << buffer << " can not be inferred correctly.";
    }

    // Collect the layout for for nodes
    Map<For, Fragment> for_map;
    Map<For, PrimExpr> predicate_map;
    for (auto& base_infer : infer_list_) {
      if (auto for_infer = dynamic_cast<ParallelOp*>(base_infer.get())) {
        ICHECK(for_infer->GetLoopLayout().defined())
            << "The Layout for Parallel for can not be infered correctly : \n"
            << for_infer->GetRoot();
        for_map.Set(for_infer->GetRoot(), for_infer->GetLoopLayout());
        if (auto predicate = for_infer->GetPredicate(thread_var_->var))
          predicate_map.Set(for_infer->GetRoot(), predicate.value());
      }
    }

    return {layout_map, for_map, predicate_map};
  }

  void Collect(const PrimFunc& f) {
    for (const auto& [_, buffer] : f->buffer_map) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined()) << "Layout_Inference: Require the target attribute";
    target_ = target.value();
    this->operator()(f->body);
  }

 private:
  void VisitExpr_(const CallNode* op) final {
    StmtExprVisitor::VisitExpr_(op);
    // Do not analysis the call node to the global function.
    if (op->op.as<GlobalVarNode>()) return;

    auto p = ParseOperator(GetRef<Call>(op), buffer_data_to_buffer_);
    if (p != nullptr) {
      for (const auto& arg : op->args) {
        if (auto buffer = getBufferFromAccessPtr(arg)) {
          addToUseList(buffer.value());
        }
      }
      infer_list_.push_back(std::move(p));
      thread_var_vec_.push_back(thread_var_);
    }
  }

  Optional<Buffer> getBufferFromAccessPtr(const PrimExpr& expr) {
    auto call = expr.as<CallNode>();
    if (call && call->op.same_as(builtin::tvm_access_ptr())) {
      auto var = call->args[1].as<Var>().value();
      return buffer_data_to_buffer_[var];
    }
    return NullOpt;
  }

  void addToUseList(const Buffer& buffer) {
    int infer_idx = infer_list_.size();
    if (use_list_.find(buffer) == use_list_.end()) {
      use_list_[buffer] = {};
    }
    use_list_[buffer].push_back(infer_idx);
  }

  void VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kParallel) {
      auto infer = std::make_unique<ParallelOp>(GetRef<For>(op));
      for (const auto& [buffer, _] : infer->GetIndiceMap()) {
        addToUseList(buffer);
      }
      infer_list_.push_back(std::move(infer));
      thread_var_vec_.push_back(thread_var_);
    } else {
      StmtExprVisitor::VisitStmt(op->body);
    }
  }

  void VisitStmt_(const BlockNode* op) final {
    for (auto buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    if (op->annotations.count(attr::kLayoutMap)) {
      auto map = op->annotations.Get(attr::kLayoutMap).as<Map<Var, Layout>>().value();
      for (const auto& [var, layout] : map) {
        auto buffer = buffer_data_to_buffer_[var];
        ICHECK(StructuralEqual()(layout->InputShape(), buffer->shape));
        annotated_layout_map_.Set(buffer, layout);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_var_ = iv;
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  std::vector<std::unique_ptr<Operator>> infer_list_;
  std::unordered_map<Buffer, std::vector<int>, ObjectPtrHash, ObjectPtrEqual> use_list_;
  IterVar thread_var_;
  std::vector<IterVar> thread_var_vec_;
  Target target_;
  LayoutMap annotated_layout_map_;
};

class LayoutInferencer : public IRMutatorWithAnalyzer {
 public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = ParallelLoopFuser::Fuse(f->body);
    BufferUseDefCollector collector;
    collector.Collect(f);
    auto result = collector.Run();
    LayoutInferencer substituter(result, &analyzer);
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

 private:
  LayoutInferencer(const LayoutInferenceResult result, arith::Analyzer* analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer), result_(result) {};

  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = Downcast<Block>(IRMutatorWithAnalyzer::VisitStmt_(op));

    for (auto buffer : block->alloc_buffers) {
      if (buffer.scope() == "local.framgent") {
        ICHECK(result_.layout_map.count(buffer))
            << "Cannot inference fragment layout for " << buffer;
      }
    }
    auto block_ptr = block.CopyOnWrite();
    block_ptr->annotations.Set(attr::kLayoutMap, result_.layout_map);
    return block;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    For for_node = Downcast<For>(IRMutatorWithAnalyzer::VisitStmt_(op));
    if (result_.for_map.count(GetRef<For>(op))) {
      auto loop_layout = result_.for_map[GetRef<For>(op)];
      for_node = PartitionLoop(for_node, thread_var_->var, analyzer_, loop_layout);
      for_node = VectorizeLoop(for_node);
      if (result_.predicate_map.count(GetRef<For>(op))) {
        return IfThenElse(result_.predicate_map[GetRef<For>(op)], for_node);
      } else {
        return for_node;
      }
    }
    return for_node;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      ICHECK_NE(iv->thread_tag.length(), 0U);
      if (iv->thread_tag == "threadIdx.x") {
        thread_var_ = iv;
      }
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

 private:
  const LayoutInferenceResult result_;
  IterVar thread_var_;
};

tvm::transform::Pass LayoutInference() {
  using namespace tir::transform;
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LayoutInferencer::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LayoutInference", {});
}

TVM_REGISTER_GLOBAL("tl.LayoutInference").set_body_typed(LayoutInference);

}  // namespace tl
}  // namespace tvm
