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

#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include <queue>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../layout/layout_infer.h"
#include "loop_partition.h"
#include "loop_vectorize.h"

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
      auto next = infer_list_[cur_infer_id];
      auto updates = next->Inference(layout_map, level);
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
      if (auto for_infer = std::dynamic_pointer_cast<ForNodeLayoutInfer>(base_infer)) {
        ICHECK(for_infer->GetLoopLayout().defined())
            << "The Layout for Parallel for can not be infered correctly : \n"
            << GetRef<For>(for_infer->GetRoot());
        for_map.Set(GetRef<For>(for_infer->GetRoot()), for_infer->GetLoopLayout());
        if (for_infer->GetPredicate().defined())
          predicate_map.Set(GetRef<For>(for_infer->GetRoot()), for_infer->GetPredicate());
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
    target_ = target.as<TargetNode>();
    this->operator()(f->body);
  }

 private:
  void VisitExpr_(const CallNode* op) final {
    StmtExprVisitor::VisitExpr_(op);
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> access_regions;
    std::shared_ptr<LayoutInferBase> p;
    ICHECK(thread_var_.defined());
    auto thread_block_size = as_const_int(thread_var_->dom->extent);
    ICHECK(thread_block_size);
    if (op->op.same_as(gemm())) {
      GemmArgs args = GemmArgs::Parse(op->args, buffer_data_to_buffer_);
      p = std::make_shared<GemmOpLayoutInfer>(args, *thread_block_size, target_);
      access_regions.insert({args.A, args.B, args.C});
    } else if (op->op.same_as(reduce())) {
      ReduceArgs args = ReduceArgs::Parse(op->args, buffer_data_to_buffer_);
      p = std::make_shared<ReduceOpLayoutInfer>(args, *thread_block_size);
      access_regions.insert({args.src, args.dst});
    }
    if (p) {
      infer_list_.push_back(p);
      for (const auto& buffer : access_regions) {
        addToUseList(buffer);
      }
    }
  }

  void addToUseList(const Buffer& buffer) {
    int infer_idx = infer_list_.size() - 1;
    if (use_list_.find(buffer) == use_list_.end()) {
      use_list_[buffer] = {};
    }
    use_list_[buffer].push_back(infer_idx);
  }

  void VisitStmt_(const ForNode* op) final {
    if (op->kind == ForKind::kParallel) {
      ICHECK(thread_var_.defined());
      auto infer = std::make_shared<ForNodeLayoutInfer>(op, thread_var_);
      infer_list_.push_back(infer);
      for (const auto& [buffer, _] : infer->GetIndiceMap()) {
        addToUseList(buffer);
      }
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
  std::vector<std::shared_ptr<LayoutInferBase>> infer_list_;
  std::unordered_map<Buffer, std::vector<int>, ObjectPtrHash, ObjectPtrEqual> use_list_;
  IterVar thread_var_;
  const TargetNode* target_;
  LayoutMap annotated_layout_map_;
};

class LayoutInferencer : public IRMutatorWithAnalyzer {
 public:
  static PrimFunc Substitute(PrimFunc f) {
    BufferUseDefCollector collector;
    collector.Collect(f);
    auto result = collector.Run();
    arith::Analyzer analyzer;
    LayoutInferencer substituter(result, &analyzer);
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

 private:
  LayoutInferencer(const LayoutInferenceResult result, arith::Analyzer* analyzer)
      : arith::IRMutatorWithAnalyzer(analyzer), result_(result) {
    for (const auto& [buffer, layout] : result_.layout_map) {
      new_alloc_.Set(buffer->data, makeBufferWithLayout(buffer, layout));
    }
  };

  Stmt VisitStmt_(const BlockNode* op) final {
    Block block = Downcast<Block>(IRMutatorWithAnalyzer::VisitStmt_(op));
    auto block_ptr = block.CopyOnWrite();
    Map<Var, Layout> new_layout_map;
    for (size_t i = 0; i < block_ptr->alloc_buffers.size(); i++) {
      const auto& buffer = block_ptr->alloc_buffers[i];
      if (new_alloc_.find(buffer->data) != new_alloc_.end()) {
        block_ptr->alloc_buffers.Set(i, new_alloc_[buffer->data]);
        new_layout_map.Set(new_alloc_[buffer->data]->data, result_.layout_map[buffer]);
      } else {
        ICHECK(buffer.scope() != "local.fragment")
            << "Cannot inference fragment layout for " << buffer;
      }
    }
    block_ptr->annotations.Set(attr::kLayoutMap, new_layout_map);
    return block;
  }

  Buffer makeBufferWithLayout(const Buffer& buffer, const Layout& layout) {
    const auto* ptr_type = TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
    Type new_type;
    // convert fragments to normal local buffer
    if (ptr_type->storage_scope == "local.fragment") {
      new_type = PointerType(ptr_type->element_type, "local");
    } else {
      new_type = buffer->data->type_annotation;
    }
    Var new_var;
    if (ptr_type->storage_scope == "global") {
      new_var = buffer->data;
    } else {
      new_var = Var(buffer->data->name_hint, new_type);
    }
    return Buffer(new_var, buffer->dtype, layout->OutputShape(), {}, buffer->elem_offset,
                  buffer->name, buffer->data_alignment, buffer->offset_factor, buffer->buffer_type);
  }

  PrimExpr VisitExpr_(const VarNode* var) final {
    if (new_alloc_.count(GetRef<Var>(var))) {
      return new_alloc_[GetRef<Var>(var)]->data;
    }
    return IRMutatorWithAnalyzer::VisitExpr_(var);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    if (new_alloc_.count(op->buffer->data)) {
      auto new_indices = result_.layout_map[op->buffer]->Forward(op->indices);
      auto new_buffer = new_alloc_[op->buffer->data];
      return BufferLoad(new_buffer, new_indices);
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (new_alloc_.count(op->buffer->data)) {
      auto new_indices = result_.layout_map[op->buffer]->Forward(op->indices);
      auto new_buffer = new_alloc_[op->buffer->data];
      auto new_value = VisitExpr(op->value);
      return BufferStore(new_buffer, new_value, new_indices);
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    Stmt body = IRMutatorWithAnalyzer::VisitStmt_(op);
    if (result_.for_map.count(GetRef<For>(op))) {
      auto loop_layout = result_.for_map[GetRef<For>(op)];
      auto stmt = PartitionLoop(body.as<ForNode>(), thread_var_->var, analyzer_,
                                result_.for_map[GetRef<For>(op)]);
      if (stmt.as<For>()) {
        stmt = VectorizeLoop(stmt.as<For>().value());
      }
      if (result_.predicate_map.count(GetRef<For>(op))) {
        stmt = IfThenElse(result_.predicate_map[GetRef<For>(op)], stmt);
      }
      return stmt;
    }
    return body;
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
  Map<Var, Buffer> new_alloc_;
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
