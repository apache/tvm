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
 * \file tir/ir/script/script_complete.cc
 * \brief Used by TVM Script parser to expand incomplete TIR input
 */

#include <tvm/arith/int_set.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <utility>

namespace tvm {
namespace tir {

/*! \brief Generate surrounding loops automatically */
class ScriptCompleter : public StmtMutator {
 public:
  explicit ScriptCompleter(Map<Var, Buffer>* buffer_var_map, bool contain_root)
      : buffer_var_map_(buffer_var_map), contain_root_(contain_root) {}
  /*! \brief Whether the stmt contains at least one block. */
  bool contains_block = false;

 private:
  Map<Var, Buffer>* buffer_var_map_;
  bool contain_root_;
  bool visited_root_ = false;
  Stmt VisitStmt_(const BlockRealizeNode* op) override {
    contains_block = true;
    Stmt body = StmtMutator::VisitStmt_(op);
    if (!op->iter_values.empty() && !op->iter_values[0].dtype().is_int()) {
      auto block_with_binding = CopyOnWrite(Downcast<BlockRealize>(body).get());
      std::vector<PrimExpr> bindings;
      for (size_t i = 0; i < op->iter_values.size(); ++i) {
        bindings.push_back(Var("i" + std::to_string(i)));
      }
      block_with_binding->iter_values = bindings;
      body = BlockRealize(block_with_binding);
      for (int i = op->iter_values.size() - 1; i >= 0; --i) {
        body = For(Downcast<Var>(bindings[i]), op->block->iter_vars[i]->dom->min,
                   op->block->iter_vars[i]->dom->extent, {}, body);
      }
    }
    return body;
  }

  Stmt VisitStmt_(const BlockNode* op) override {
    bool is_root_block = contain_root_ && !visited_root_;
    visited_root_ = true;
    // Buffers allocated in the block can be accessed by its body.
    for (const auto& alloc_buffer : op->alloc_buffers) {
      buffer_var_map_->Set(alloc_buffer->data, alloc_buffer);
    }
    Block block = Downcast<Block>(StmtMutator::VisitStmt_(op));
    // Remove buffers allocated inside block to detect its access region
    for (const auto& alloc_buffer : op->alloc_buffers) {
      buffer_var_map_->erase(alloc_buffer->data);
    }
    // Get access detection mask
    // 0 for provided region, 1 and 3 for need detect read, 2 and 3 for need detect write
    int mask = 0;
    auto it = op->annotations.find(attr::script_parsing_detect_access);
    if (it != op->annotations.end()) {
      mask = Downcast<IntImm>((*it).second)->value;
    }
    // ignore root block or blocks which already has reads/writes regions
    if (mask != 0) {
      if (op->iter_vars.empty()) {
        // non-root opaque block is not allowed
        CHECK(is_root_block)
            << "ValueError: Can not auto detect buffer access region for an opaque block. Please "
               "annotate the access region manually.";
        return std::move(block);
      }
      auto access_region = GetBlockAccessRegion(block, *buffer_var_map_);
      const Array<BufferRegion>& reads = access_region[0];
      const Array<BufferRegion>& writes = access_region[1];
      const Array<BufferRegion>& opaque = access_region[2];
      CHECK(opaque.empty())
          << "ValueError: Can not auto detect buffer access region from tir.Load, tir.Store or "
             "direct access by buffer data. Please annotation the access region manually";
      auto n = CopyOnWrite(block.operator->());
      if (mask & 1) n->reads = reads;
      if (mask & 2) n->writes = writes;
      n->annotations = op->annotations;
      n->annotations.erase(attr::script_parsing_detect_access);
      return Block(n);
    } else {
      return std::move(block);
    }
  }
};

PrimFunc ScriptComplete(PrimFunc func, const Array<Buffer>& root_allocates) {
  Map<Var, Buffer> buffer_var_map;
  for (const auto& pair : func->buffer_map) {
    const Buffer& buffer = pair.second;
    buffer_var_map.Set(buffer->data, buffer);
  }
  for (const auto& alloc : root_allocates) {
    buffer_var_map.Set(alloc->data, alloc);
  }
  bool contain_root = root_allocates.empty() && func->body->IsInstance<BlockRealizeNode>() &&
                      Downcast<BlockRealize>(func->body)->block->iter_vars.empty();
  ScriptCompleter script_completer(&buffer_var_map, contain_root);
  // generate surrounding loops automatically
  Stmt res = script_completer(func->body);
  // generate root block automatically
  if (script_completer.contains_block && !contain_root) {
    res = Block({}, {}, {}, "root", res, NullOpt, root_allocates);
    res = BlockRealize({}, Bool(true), Downcast<Block>(res));
  }
  if (func->body.same_as(res)) {
    return func;
  } else {
    auto fptr = func.CopyOnWrite();
    fptr->body = res;
    return func;
  }
}

TVM_REGISTER_GLOBAL("script.Complete").set_body_typed(ScriptComplete);

}  // namespace tir
}  // namespace tvm
