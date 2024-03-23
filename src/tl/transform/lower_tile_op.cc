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

#include "../../arith/ir_mutator_with_analyzer.h"
#include "../layout/layout.h"
#include "../layout/utils.h"
#include "../op/op.h"
#include "loop_partition.h"

namespace tvm {
namespace tl {

using namespace tir;

class LowerTileOpPass : arith::IRMutatorWithAnalyzer {
 public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    LowerTileOpPass substituter(&analyzer);
    for (const auto& [_, buffer] : f->buffer_map) {
      substituter.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    PrimFuncNode* fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

 private:
  using arith::IRMutatorWithAnalyzer::IRMutatorWithAnalyzer;

  Stmt VisitStmt_(const BlockNode* op) final {
    Map<Var, Layout> vmap;
    if (op->annotations.count(attr::kLayoutMap)) {
      vmap = op->annotations.at(attr::kLayoutMap).as<Map<Var, Layout>>().value();
    }
    for (auto buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
      if (vmap.count(buffer->data)) layout_map_.Set(buffer, vmap[buffer->data]);
    }
    auto ret = arith::IRMutatorWithAnalyzer::VisitStmt_(op);
    auto block = ret.as<Block>().value();
    if (!workspaces_.empty()) {
      auto block_ptr = block.CopyOnWrite();
      for (const auto& buffer : workspaces_) block_ptr->alloc_buffers.push_back(buffer);
      workspaces_.clear();
    }
    return block;
  }

  Stmt VisitStmt_(const EvaluateNode* node) final {
    auto op = ParseOperator(GetRef<Stmt>(node), buffer_data_to_buffer_);
    if (op == nullptr) return GetRef<Stmt>(node);
    AddWorkspaceCallback callback = [this](int num_elem, DataType dtype) {
      auto workspace = decl_buffer({PrimExpr(num_elem)}, dtype, "workspace", "shared.dyn");
      workspaces_.push_back(workspace);
      return workspace.access_ptr(2);  // write
    };
    auto lowered =
        op->Lower(LowerArgs{thread_block_size_, thread_var_, callback, layout_map_}, analyzer_);
    if (lowered.defined())
      return lowered;
    else
      return GetRef<Stmt>(node);
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

  Map<Var, Buffer> buffer_data_to_buffer_;
  Map<Buffer, Layout> layout_map_;
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
