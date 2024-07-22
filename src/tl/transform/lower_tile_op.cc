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
 * \file lower_tile_op.cc
 * \brief Lower the tile op for further codegen.
 */

#include <tvm/tir/builtin.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>
#include <tvm/tir/utils.h>

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../layout/layout.h"
#include "../layout/utils.h"
#include "../op/op.h"
#include "loop_partition.h"

namespace tvm {
namespace tl {

using namespace tir;

static Buffer makeBufferWithLayout(const Buffer& buffer, const Layout& layout) {
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

class LowerTileOpPass : arith::IRMutatorWithAnalyzer {
 public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    LowerTileOpPass substituter(&analyzer);
    for (const auto& [_, buffer] : f->buffer_map) {
      substituter.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined()) << "LowerTileOpPass: Require the target attribute";
    substituter.target_ = target.value();
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

 private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Stmt VisitStmt_(const BlockNode* op) final {
    for (auto buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    Map<Var, Layout> vmap;
    if (op->annotations.count(attr::kLayoutMap)) {
      auto layout_map = op->annotations.at(attr::kLayoutMap).as<Map<Buffer, Layout>>().value();
      for (auto [buffer, layout] : layout_map) {
        buffer_remap_.Set(buffer, makeBufferWithLayout(buffer, layout));
        layout_map_.Set(buffer, layout);
      }
    }
    auto block = Downcast<Block>(arith::IRMutatorWithAnalyzer::VisitStmt_(op));
    auto block_ptr = block.CopyOnWrite();
    for (size_t i = 0; i < block->alloc_buffers.size(); i++) {
      auto buffer = block->alloc_buffers[i];
      if (buffer_remap_.count(buffer)) {
        block_ptr->alloc_buffers.Set(i, buffer_remap_[buffer]);
      }
    }
    for (const auto& buffer : workspaces_) block_ptr->alloc_buffers.push_back(buffer);
    workspaces_.clear();
    block_ptr->annotations.erase(attr::kLayoutMap);
    return block;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto load = Downcast<BufferLoad>(IRMutatorWithAnalyzer::VisitExpr_(op));
    if (buffer_remap_.count(load->buffer)) {
      auto new_indices = layout_map_[load->buffer]->Forward(load->indices);
      auto new_buffer = buffer_remap_[load->buffer];
      return BufferLoad(new_buffer, new_indices);
    }
    return load;
  }

  Stmt VisitStmt_(const BufferStoreNode* op) final {
    auto store = Downcast<BufferStore>(IRMutatorWithAnalyzer::VisitStmt_(op));
    if (buffer_remap_.count(store->buffer)) {
      auto new_indices = layout_map_[store->buffer]->Forward(store->indices);
      auto new_buffer = buffer_remap_[store->buffer];
      return BufferStore(new_buffer, store->value, new_indices);
    }
    return store;
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    auto var = Downcast<Var>(IRMutatorWithAnalyzer::VisitExpr_(op));
    if (buffer_data_to_buffer_.count(var)) {
      auto buffer = buffer_data_to_buffer_[var];
      if (buffer_remap_.count(buffer)) return buffer_remap_[buffer]->data;
    }
    return var;
  }

  Stmt VisitStmt_(const EvaluateNode* op) final {
    auto tile_op = ParseOperator(GetRef<Stmt>(op), buffer_data_to_buffer_);
    if (tile_op == nullptr) return IRMutatorWithAnalyzer::VisitStmt_(op);
    AddWorkspaceCallback callback = [this](int num_elem, DataType dtype) {
      auto workspace = decl_buffer({PrimExpr(num_elem)}, dtype, "workspace", "shared.dyn");
      workspaces_.push_back(workspace);
      return workspace.access_ptr(2);  // write
    };
    auto lowered = tile_op->Lower(
        LowerArgs{target_, thread_block_size_, thread_var_, callback, layout_map_, buffer_remap_},
        analyzer_);
    return IRMutatorWithAnalyzer::VisitStmt(lowered);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == tir::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      ICHECK_NE(iv->thread_tag.length(), 0U);
      if (iv->thread_tag == "threadIdx.x") {
        thread_var_ = iv->var;
        ICHECK(iv->dom->extent.as<IntImmNode>());
        thread_block_size_ = iv->dom->extent.as<IntImmNode>()->value;
      }
    }
    return arith::IRMutatorWithAnalyzer::VisitStmt_(op);
  }

  Target target_;
  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Layout> layout_map_;
  Map<Buffer, Buffer> buffer_remap_;
  Var thread_var_;
  size_t thread_block_size_ = 0;
  Array<Buffer> workspaces_;
};

namespace transform {

using namespace tir::transform;

tvm::transform::Pass LowerTileOp() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerTileOpPass::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerTileOp", {});
}

TVM_REGISTER_GLOBAL("tl.LowerTileOp").set_body_typed(LowerTileOp);
}  // namespace transform

}  // namespace tl
}  // namespace tvm
