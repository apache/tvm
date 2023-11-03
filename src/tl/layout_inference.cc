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
#include "arith.h"
#include "helper.h"
#include "layout.h"
#include "loop_partition.h"
#include "auto_vectorize.h"
#include "op.h"

namespace tvm {
namespace tir {

using namespace tl;
using arith::IRMutatorWithAnalyzer;

class LayoutInferBase {
public:
  virtual Map<Buffer, Layout> Inference(const Map<Buffer, Layout>& layout_map,
    const Map<Var, Buffer>& buffer_data_to_buffer) = 0;
};

class CallNodeLayoutInfer : public LayoutInferBase {
public:
  CallNodeLayoutInfer(const CallNode* node, size_t block_size)
    : node_(node), block_size_(block_size) {};

  Map<Buffer, Layout> Inference(const Map<Buffer, Layout>& layout_map,
    const Map<Var, Buffer>& buffer_data_to_buffer) final {
    Map<Buffer, Layout> results;
    if (node_->op.same_as(gemm())) {
      auto args = GemmArgs::Parse(node_->args, buffer_data_to_buffer);
      ICHECK(args.C.scope() == "local.fragment");
      auto [warp_m, warp_n] = args.ComputeWarpPartition(block_size_ / 32);
      auto fragment = makeGemmFragmentC(args.M, args.N, args.M / warp_m, args.N / warp_n);
      results.Set(args.C, fragment);

      if (args.A.scope() == "shared" || args.A.scope() == "shared.dyn") {
        results.Set(args.A,
          makeGemmABLayout(*as_const_int(args.A->shape[0]),
            *as_const_int(args.A->shape[1]), args.A->dtype.bits(), args.trans_A ? 1 : 2));
      }
      if (args.B.scope() == "shared" || args.B.scope() == "shared.dyn") {
        results.Set(args.B,
          makeGemmABLayout(*as_const_int(args.B->shape[0]),
            *as_const_int(args.B->shape[1]), args.B->dtype.bits(), args.trans_B ? 2 : 1));
      }
      if (args.A.scope() == "local.fragment") {
        results.Set(args.A, makeGemmFragmentA(args.M, args.N, args.K, args.M / warp_m, args.N / warp_n));
      }
    } else if (node_->op.same_as(reduce())) {
      // ICHECK(0) << "TODO";
    } else if (node_->op.same_as(tl::copy())) {
      // auto args = CopyArgs::Parse(node_->args, buffer_data_to_buffer);
      // if (args.src.scope() == "local.fragment" && args.dst.scope() == "local.fragment") {
      //   if (layout_map.count(args.src) && !layout_map.count(args.dst)) {
      //     results.Set(args.dst, layout_map[args.src]);
      //   } else if (!layout_map.count(args.src) && layout_map.count(args.dst)) {
      //     results.Set(args.src, layout_map[args.dst]);
      //   }
      // }
    }
    return results;
  }

private:
  const CallNode* node_;
  size_t block_size_;
};

class ForNodeLayoutInfer : public LayoutInferBase, StmtExprVisitor {
public:
  ForNodeLayoutInfer(const ForNode* root) : root_(root) {
    VisitStmt_(root);
    // Check if the buffer indice matches full range
    for (const auto& [buffer, indices] : indice_map_) {
      Layout layout(loop_vars_, indices);
      ICHECK(StructuralEqual()(buffer->shape, layout->OutputShape()))
        << "Parallel for over fragment does not match full region, " << buffer->shape << " "
        << layout->OutputShape();
    }
  };

  Map<Buffer, Layout> Inference(const Map<Buffer, Layout>& layout_map,
    const Map<Var, Buffer>& buffer_data_to_buffer) final {
    Map<Buffer, Layout> results;
    Buffer source_buffer;
    Array<Buffer> dest_buffers;
    for (const auto& [buffer, _] : indice_map_) {
      if (!layout_map.count(buffer)) {
        dest_buffers.push_back(buffer);
      } else {
        auto frag = layout_map[buffer].as<Fragment>().value();
        if (is_one(frag->ReplicateExtent()) || buffer_is_write_.count(buffer))
          source_buffer = buffer;
      }
    }
    if (!source_buffer.defined()) return {};

    /*
      let rep_b = (loop_var) | (b_ind)
      ind_b_inv(b_vars, rep_b) = loop_var
      thd(loop_var) = thd_A(ind_A(loop_var))
      thd_b(b_ind, rep_b) = thd(ind_b_inv(b_vars, rep_b)))
    */
    auto src_layout = layout_map[source_buffer].as<Fragment>().value();
    if (IsCommonAccessIndice(source_buffer)) {
      loop_layout_ = src_layout;
    } else {
      PrimExpr loop_var_to_thread = src_layout->ForwardThread(indice_map_[source_buffer], {});
      loop_layout_ = Fragment(loop_vars_, {}, loop_var_to_thread, src_layout->thread_replicate_);
    }

    arith::Analyzer analyzer;
    loop_layout_->UpdateAnalyzer(&analyzer);

    for (const auto& buffer : dest_buffers) {
      Fragment dest_layout;
      if (IsCommonAccessIndice(buffer)) {
        dest_layout = loop_layout_;
      } else {
        PrimExpr rep_b = MakeFlattenedExpression(
          DivideUnusedIterators(indice_map_[buffer], loop_vars_, &analyzer));

        auto bijective_indice = indice_map_[buffer];
        bijective_indice.push_back(rep_b);
        Layout ind_inv = Layout(loop_vars_, bijective_indice)->Inverse();

        PrimExpr indice_rep_extent = ind_inv->InputShape().back();  // this is the size of rep_b
        PrimExpr loop_rep_extent = loop_layout_->ReplicateExtent();
        PrimExpr dest_buffer_rep_extent = indice_rep_extent * loop_rep_extent;
        IterVar rep = IterVar(Range(0, dest_buffer_rep_extent), Var("rep"), IterVarType::kDataPar);

        Array<IterVar> iter_vars;
        Array<PrimExpr> fwd;
        for (size_t i = 0; i + 1 < ind_inv->InputDim(); i++) {
          auto var = Var("i" + std::to_string(i));
          iter_vars.push_back(
            IterVar(Range(0, ind_inv->InputShape()[i]), var, IterVarType::kDataPar));
          fwd.push_back(var);
        }
        fwd.push_back(FloorMod(rep, indice_rep_extent));
        PrimExpr thd_b =
          loop_layout_->ForwardThread(ind_inv->Forward(fwd), FloorDiv(rep, indice_rep_extent));

        dest_layout = Fragment(iter_vars, {}, thd_b, rep)->CondenseReplicateVar();
      }
      results.Set(buffer, dest_layout);
    }
    return results;
  }

  Fragment GetLoopLayout() {
    return loop_layout_;
  }

  const ForNode* GetRoot() { return root_; }

  Map<Buffer, Array<PrimExpr>> GetIndiceMap() { return indice_map_; }

private:
  bool IsCommonAccessIndice(const Buffer& buffer) {
    auto common_indice = loop_vars_.Map([](const auto& iv) { return iv->var; });
    return StructuralEqual()(indice_map_[buffer], common_indice);
  }

  void VisitStmt_(const ForNode* op) final {
    ICHECK(op->kind == ForKind::kParallel);
    loop_vars_.push_back(IterVar(Range(op->min, op->extent), op->loop_var, IterVarType::kDataPar));
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BufferStoreNode* op) final {
    if (op->buffer.scope() == "local.fragment") {
      if (indice_map_.find(op->buffer) != indice_map_.end()) {
        ICHECK(StructuralEqual()(indice_map_.at(op->buffer), op->indices))
          << op->buffer << ": " << op->indices << " and " << indice_map_.at(op->buffer);
      } else {
        indice_map_.Set(op->buffer, op->indices);
      }
      buffer_is_write_.insert(op->buffer);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    if (op->buffer.scope() == "local.fragment") {
      if (indice_map_.find(op->buffer) != indice_map_.end()) {
        ICHECK(StructuralEqual()(indice_map_.at(op->buffer), op->indices))
          << op->buffer << ": " << op->indices << " and " << indice_map_.at(op->buffer);
      } else {
        indice_map_.Set(op->buffer, op->indices);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  const ForNode* root_;
  Map<Buffer, Array<PrimExpr>> indice_map_;
  std::unordered_set<Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_is_write_;
  Array<IterVar> loop_vars_;
  Fragment loop_layout_;
};

class BufferUseDefCollector : public StmtExprVisitor {
public:
  BufferUseDefCollector() = default;

  auto Run() -> std::pair<Map<Buffer, Layout>, Map<For, Fragment>> {
    Map<Buffer, Layout> layout_map;
    std::queue<int> q;
    std::vector<bool> in_queue(infer_list_.size(), true);
    for (int i = 0; i < int(infer_list_.size()); i++) {
      q.push(i);
    }
    while (!q.empty()) {
      int cur_infer_id = q.front();
      auto next = infer_list_[cur_infer_id];
      in_queue[cur_infer_id] = false;
      q.pop();
      auto updates = next->Inference(layout_map, buffer_data_to_buffer_);
      for (const auto& [buffer, layout] : updates) {
        if (layout_map.count(buffer)) {
          ICHECK(StructuralEqual()(layout, layout_map[buffer])) << "Get different layout for " << buffer;
        } else {
          layout_map.Set(buffer, layout);
          for (int idx : use_list_[buffer]) {
            if (!in_queue[idx] && idx != cur_infer_id) {
              in_queue[idx] = true;
              q.push(idx);
            }
          }
        }
      }
    }

    Map<For, Fragment> for_map;
    for (auto& base_infer : infer_list_) {
      if (auto for_infer = std::dynamic_pointer_cast<ForNodeLayoutInfer>(base_infer)) {
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
    for (auto arg : op->args) {
      if (auto c = arg.as<CallNode>()) {
        if (c->op.same_as(Op::Get("tl.region"))) {
          auto buffer_var = (c->args[0]).as<Var>().value();
          if (buffer_data_to_buffer_.count(buffer_var))
            access_regions.insert(buffer_data_to_buffer_.at(buffer_var));
        } else if (c->op.same_as(Op::Get("tir.tvm_access_ptr"))) {
          auto buffer_var = (c->args[1]).as<Var>().value();
          if (buffer_data_to_buffer_.count(buffer_var))
            access_regions.insert(buffer_data_to_buffer_.at(buffer_var));
        }
      }
    }
    if (!access_regions.empty()) {
      ICHECK(thread_block_size_ > 0);
      infer_list_.push_back(std::make_shared<CallNodeLayoutInfer>(op, thread_block_size_));
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
      auto infer = std::make_shared<ForNodeLayoutInfer>(op);
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
      if (loop_layout.defined()) {
        auto new_for = PartitionLoop(body.as<ForNode>(), thread_var_, analyzer_, for_map_[GetRef<For>(op)]);
        new_for = VectorizeLoop(new_for);
        return new_for;
      } else {
        auto new_for = VectorizeLoop(body.as<For>().value());
        new_for = PartitionLoop(new_for.get(), thread_var_, analyzer_, thread_block_size_);
        return new_for;
      }
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
