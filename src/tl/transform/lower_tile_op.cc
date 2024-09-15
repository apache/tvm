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
    // Trace the buffer map for tvm_access_ptr
    substituter.buffer_map_.insert(f->buffer_map.begin(), f->buffer_map.end());
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
    // Record the mapping from buffer data var to buffer for later lookup
    for (auto buffer : op->alloc_buffers) {
      buffer_map_.insert({buffer->data, buffer});
    }
    for (auto match_buffer : op->match_buffers) {
      buffer_map_.insert({match_buffer->buffer->data, match_buffer->buffer});
    }
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


  int CheckAndGetBufferRowSize(Buffer buffer) {
    CHECK(buffer->shape.size() >= 2)
        << "The dimension of Buffer \"" << buffer->name << "\" with shape " << buffer->shape
        << " should be at least 2";

    auto dim = buffer->shape.size();
    auto buffer_row_size = buffer->shape[dim - 1].as<IntImmNode>()->value;
    // auto buffer_col_size = buffer->shape[dim - 2].as<IntImmNode>()->value;

    return buffer_row_size;
  }

  PrimExpr HandleAccessPtrAndOffset(PrimExpr access_ptr, Optional<PrimExpr> offset = NullOpt,
                                    DataType dtype = DataType::Int(32)) {
    // The 2th arg of T.tvm_access_ptr call is offset, we set it to 0 and accumulate it to
    // smem_offset
    CHECK(access_ptr->IsInstance<CallNode>())
        << "Invalid access ptr for permuted layout: " << access_ptr;
    auto access_ptr_call = Downcast<Call>(access_ptr);
    if (access_ptr_call->op.same_as(builtin::tvm_access_ptr())) {
      LOG(FATAL) << "Transformation for tvm_access_ptr is not implemented yet";
    } else if (access_ptr_call->op.same_as(builtin::address_of())) {
      BufferLoad load = Downcast<BufferLoad>(access_ptr_call->args[0]);
      Array<PrimExpr> indices = load->indices;
      Array<PrimExpr> shape = load->buffer->shape;
      // TODO(lei): Can improve this by using a more general way to handle multi-dimension buffer
      CHECK_EQ(indices.size(), 2) << "Currently only support 2D buffer";
      CHECK_EQ(shape.size(), 2) << "Currently only support 2D buffer";

      PrimExpr elem_offset = indices[0] * shape[1] + indices[1];
      PrimExpr smem_offset = elem_offset + (offset.defined() ? offset.value() : 0);

      auto new_buffer = buffer_remap_[load->buffer];

      auto buffer_map_iter = buffer_map_.find(Downcast<Var>(load->buffer->data));
      CHECK(buffer_map_iter != buffer_map_.end())
          << "The buffer corresponding to data Var " << access_ptr_call->args[0] << " is not found";

      int buffer_row_size = CheckAndGetBufferRowSize(buffer_map_iter->second);

      // Convert offset to 2-dimension, reindex it and convert it back
      PrimExpr row_idx = floordiv(smem_offset, buffer_row_size);
      PrimExpr col_idx = floormod(smem_offset, buffer_row_size);
      auto forward_indices = layout_map_[load->buffer]->Forward({row_idx, col_idx});
      auto new_offset =
          analyzer_->Simplify(forward_indices[0] * buffer_row_size + forward_indices[1]);
      Array<PrimExpr> new_indices = {PrimExpr(Div(new_offset, shape[1])),
                                     PrimExpr(Mod(new_offset, shape[1]))};

      auto new_access_ptr = access_ptr_call.CopyOnWrite();
      if (new_buffer->shape.size() == 2) {
        new_access_ptr->args.Set(0, BufferLoad(new_buffer, new_indices));
      } else {
        LOG(FATAL) << "Invalid buffer shape for permuted layout: " << new_buffer->shape;
      }
    } else {
      LOG(FATAL) << "Invalid access op for permuted layout: " << access_ptr;
    }

    return access_ptr_call;
  }

  PrimExpr VisitExpr_(const tir::CallNode* op) final {
    if (!op->op.same_as(builtin::ptx_ldmatrix()) && !op->op.same_as(builtin::mma_store())) {
      return Downcast<Call>(IRMutatorWithAnalyzer::VisitExpr_(op));
    } else {
      is_ptx_ = true;
    }
    // Rewrite from/to shared or shared.dyn to/from local
    auto call = Downcast<Call>(IRMutatorWithAnalyzer::VisitExpr_(op));
    if (call->op.same_as(builtin::ptx_ldmatrix())) {
      // form: T.ptx_ldmatrix(..., smem_ptr, smem_offset)
      // smem_ptr: T.tvm_access_ptr(ptype, data, offset, extent, rw_mask)
      // or T.address_of(buffer, offset)
      auto access_ptr = call->args[5];
      PrimExpr smem_offset = call->args[6];
      Call address_of_call = Downcast<Call>(access_ptr);
      if (!address_of_call->op.same_as(builtin::address_of())) {
        LOG(FATAL) << "Invalid access ptr for permuted layout: " << access_ptr;
      }
      BufferLoad load = Downcast<BufferLoad>(address_of_call->args[0]);

      if (buffer_remap_.count(load->buffer)) {
        auto new_access_ptr = HandleAccessPtrAndOffset(access_ptr, smem_offset, call->dtype);
        auto new_call = call.CopyOnWrite();
        new_call->args.Set(5, new_access_ptr);
        new_call->args.Set(6, IntImm(smem_offset->dtype, 0));
      }
    } else if (call->op.same_as(builtin::mma_store())) {
      // TODO(yixin): mma_store is not fully tested yet
      // because we will directly store result to Buffer instead of calling mma_store now
      auto access_ptr = call->args[2];
      auto new_access_ptr = HandleAccessPtrAndOffset(access_ptr, NullOpt, call->dtype);
      auto new_call = call.CopyOnWrite();
      new_call->args.Set(2, new_access_ptr);
    } else {
      LOG(FATAL) << "Invalid call node: " << call;
    }
    is_ptx_ = false;
    return call;
  }

  PrimExpr VisitExpr_(const BufferLoadNode* op) final {
    auto load = Downcast<BufferLoad>(IRMutatorWithAnalyzer::VisitExpr_(op));
    if (is_ptx_) {
      return load;
    }
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
  // For ptx Node, we need to remap the buffer and indices
  // By access CallNode instead of BufferLoad Node.
  bool is_ptx_{false};
  // Mapping from data Var of a Buffer to Buffer, for lookup
  std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual> buffer_map_;
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
