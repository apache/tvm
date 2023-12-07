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

#include "../arith/ir_mutator_with_analyzer.h"
#include "loop_partition.h"
#include "auto_vectorize.h"
#include "layout_infer.h"

namespace tvm {
namespace tir {

using namespace tl;
using arith::IRMutatorWithAnalyzer;

class BufferUseDefCollector : public StmtExprVisitor {
public:
  BufferUseDefCollector() = default;

  auto Run() -> std::pair<Map<Buffer, Layout>, Map<For, Fragment>> {
    Map<Buffer, Layout> layout_map;
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
          ICHECK(StructuralEqual()(layout, layout_map[buffer])) << "Get different layout for " << buffer;
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
    for (const auto& [buffer, _]: use_list_) {
      if (buffer.scope() == "local.fragment" && layout_map.count(buffer) == 0)
        LOG_ERROR << "The layout for fragment " << buffer << " can not be inferred correctly.";
    }

    // Collect the layout for for nodes
    Map<For, Fragment> for_map;
    for (auto& base_infer : infer_list_) {
      if (auto for_infer = std::dynamic_pointer_cast<ForNodeLayoutInfer>(base_infer)) {
        ICHECK(for_infer->GetLoopLayout().defined()) << "The Layout for Parallel for can not be infered correctly : \n"
          << GetRef<For>(for_infer->GetRoot());
        for_map.Set(GetRef<For>(for_infer->GetRoot()), for_infer->GetLoopLayout());
      }
    }
    return { layout_map, for_map };
  }

  void Collect(const PrimFunc& f) {
    for (const auto& [_, buffer] : f->buffer_map) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    this->operator()(f->body);
  }

private:
  void VisitExpr_(const CallNode* op) final {
    StmtExprVisitor::VisitExpr_(op);
    std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> access_regions;
    std::shared_ptr<LayoutInferBase> p;
    if (op->op.same_as(gemm())) {
      GemmArgs args = GemmArgs::Parse(op->args, buffer_data_to_buffer_);
      p = std::make_shared<GemmOpLayoutInfer>(args, thread_block_size_);
      access_regions.insert({ args.A, args.B, args.C });
    } else if (op->op.same_as(reduce())) {
      ReduceArgs args = ReduceArgs::Parse(op->args, buffer_data_to_buffer_);
      p = std::make_shared<ReduceOpLayoutInfer>(args, thread_block_size_);
      access_regions.insert({ args.src, args.dst });
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
      auto infer = std::make_shared<ForNodeLayoutInfer>(op, thread_block_size_);
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
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_block_size_ = iv->dom->extent.as<IntImmNode>()->value;
        ICHECK(thread_block_size_ % 32 == 0);
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  std::vector<std::shared_ptr<LayoutInferBase>> infer_list_;
  std::unordered_map<Buffer, std::vector<int>, ObjectPtrHash, ObjectPtrEqual> use_list_;
  size_t thread_block_size_ = 0;
};

class LayoutInferencer : public IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f) {
    BufferUseDefCollector collector;
    collector.Collect(f);
    Map<Buffer, Layout> layout_map;
    Map<For, Fragment> for_map;
    std::tie(layout_map, for_map) = collector.Run();
    arith::Analyzer analyzer;
    LayoutInferencer substituter(layout_map, for_map, &analyzer);
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  LayoutInferencer(const Map<Buffer, Layout>& layout_map, const Map<For, Fragment> for_map,
    arith::Analyzer* analyzer)
    : arith::IRMutatorWithAnalyzer(analyzer), layout_map_(layout_map), for_map_(for_map) {};

  Stmt VisitStmt_(const BlockNode* op) final {
    for (const auto& buffer : op->alloc_buffers) {
      if (layout_map_.find(buffer) != layout_map_.end()) {
        Layout layout = layout_map_[buffer];
        const auto* ptr_type = TVM_TYPE_AS(buffer->data->type_annotation, PointerTypeNode);
        Type new_type;
        if (ptr_type->storage_scope == "local.fragment") {
          new_type = PointerType(ptr_type->element_type, "local");
        } else {
          new_type = buffer->data->type_annotation;
        }
        Var new_var = Var(buffer->data->name_hint, new_type);
        new_alloc_.Set(buffer->data,
          Buffer(new_var, buffer->dtype, layout->OutputShape(), {},
            buffer->elem_offset, buffer->name, buffer->data_alignment,
            buffer->offset_factor, buffer->buffer_type));
      } else {
        ICHECK(buffer.scope() != "local.fragment")
          << "Cannot inference fragment layout for " << buffer;
      }
    }
    Block block = Downcast<Block>(IRMutatorWithAnalyzer::VisitStmt_(op));
    auto block_ptr = block.CopyOnWrite();
    Map<Var, Layout> new_layout_map;
    for (size_t i = 0; i < block_ptr->alloc_buffers.size(); i++) {
      const auto& buffer = block_ptr->alloc_buffers[i];
      if (new_alloc_.find(buffer->data) != new_alloc_.end()) {
        block_ptr->alloc_buffers.Set(i, new_alloc_[buffer->data]);
        new_layout_map.Set(new_alloc_[buffer->data]->data, layout_map_[buffer]);
      }
    }
    block_ptr->annotations.Set("layout_map", new_layout_map);
    return block;
  }

  PrimExpr VisitExpr_(const VarNode* var) final {
    if (new_alloc_.count(GetRef<Var>(var))) {
      return new_alloc_[GetRef<Var>(var)]->data;
    }
    return IRMutatorWithAnalyzer::VisitExpr_(var);
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    if (new_alloc_.count(op->buffer->data)) {
      auto new_indices = layout_map_[op->buffer]->Forward(op->indices);
      auto new_buffer = new_alloc_[op->buffer->data];
      return BufferLoad(new_buffer, new_indices);
    }
    return IRMutatorWithAnalyzer::VisitExpr_(op);
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    if (new_alloc_.count(op->buffer->data)) {
      auto new_indices = layout_map_[op->buffer]->Forward(op->indices);
      auto new_buffer = new_alloc_[op->buffer->data];
      auto new_value = VisitExpr(op->value);
      return BufferStore(new_buffer, new_value, new_indices);
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    Stmt body = IRMutatorWithAnalyzer::VisitStmt_(op);
    if (for_map_.find(GetRef<For>(op)) != for_map_.end()) {
      auto loop_layout = for_map_[GetRef<For>(op)];
      auto new_for = PartitionLoop(body.as<ForNode>(), thread_var_, analyzer_, for_map_[GetRef<For>(op)]);
      new_for = VectorizeLoop(new_for);
      return new_for;
    }
    return body;
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      ICHECK_NE(iv->thread_tag.length(), 0U);
      if (iv->thread_tag == "threadIdx.x") {
        thread_var_ = iv->var;
        thread_block_size_ = iv->dom->extent.as<IntImmNode>()->value;
        ICHECK(thread_block_size_ % 32 == 0);
      }
    }
    return IRMutatorWithAnalyzer::VisitStmt_(op);
  }

private:
  Map<Buffer, Layout> layout_map_;
  Map<For, Fragment> for_map_;
  Map<Var, Buffer> new_alloc_;
  Var thread_var_;
  size_t thread_block_size_ = 0;
};

namespace transform {

tvm::transform::Pass LayoutInference() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LayoutInferencer::Substitute(std::move(f));
    };
  return CreatePrimFuncPass(pass_func, 0, "tl.LayoutInference", {});
}

TVM_REGISTER_GLOBAL("tl.LayoutInference").set_body_typed(LayoutInference);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
