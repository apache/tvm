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

#include <tvm/arith/analyzer.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../../arith/const_fold.h"
#include "../../arith/pattern_match.h"

namespace tvm {
namespace tir {

class PTXRewriter : public StmtMutator {
 public:
  Stmt VisitStmt_(const AllocateNode* allocate) final {
    if (!has_buffer_1) {
      has_buffer_1 = true;
      // addr[0] -> global_addr /  addr[1] -> local_addr
      addr_buffer = decl_buffer({IntImm(DataType::Int(32), 2)}, DataType::Int(32), "addr", "local");
      predicate_buffer =
          decl_buffer({IntImm(DataType::Int(32), 1)}, DataType::Bool(1), "predicate", "local");
    }
    Stmt result = StmtMutator::VisitStmt_(allocate);
    if (!has_buffer_2) {
      has_buffer_2 = true;
      result =
          Allocate(addr_buffer->data, addr_buffer->dtype, addr_buffer->shape, Bool(true), result);
      result = Allocate(predicate_buffer->data, predicate_buffer->dtype, predicate_buffer->shape,
                        Bool(true), result);
    }
    return result;
  }

  Stmt VisitStmt_(const BufferStoreNode* store) final {
    Stmt result = StmtMutator::VisitStmt_(store);
    Buffer load_buffer = store->buffer;
    PrimExpr load_value = store->value;
    // const BufferLoadNode* gload = load_value.as<BufferLoadNode>(); // take
    // the place of instance of
    const CallNode* call = load_value.as<CallNode>();
    if (call != nullptr) {
      const OpNode* op = call->op.as<OpNode>();
      if (op != nullptr && op->name == "tir.if_then_else") {
        const PrimExpr& predicate = call->args[0];
        const PrimExpr& lhs = call->args[1];
        const PrimExpr& rhs = call->args[2];
        PrimExpr global_addr, local_addr;
        const BufferLoadNode* load = lhs.as<BufferLoadNode>();
        PrimExpr imm_value = rhs;
        if (load == nullptr) {
          load = rhs.as<BufferLoadNode>();
          imm_value = lhs;
          if (load == nullptr) {
            return result;
          }
        }
        global_addr = load->indices[0];
        const RampNode* ramp = global_addr.as<RampNode>();
        if (ramp != nullptr) {
          return result;
        }
        local_addr = store->indices[0];
        BufferStore addr_store(addr_buffer, global_addr, {IntImm(DataType::Int(32), 0)});
        BufferStore local_addr_store(addr_buffer, local_addr, {IntImm(DataType::Int(32), 1)});
        BufferStore predicate_store(predicate_buffer, predicate, {IntImm(DataType::Int(32), 0)});
        PrimExpr new_lhs, new_rhs, new_predicate, new_indice;
        new_lhs =
            BufferLoad(load->buffer, {BufferLoad(addr_buffer, {IntImm(DataType::Int(32), 0)})});
        new_rhs = IntImm(DataType::Int(32), 0);
        new_predicate = BufferLoad(predicate_buffer, {IntImm(DataType::Int(32), 0)});
        new_indice = BufferLoad(addr_buffer, {IntImm(DataType::Int(32), 1)});
        BufferStore value_store(store->buffer, imm_value, {new_indice});
        Evaluate ptx_load(Call(store->buffer->dtype, tvm::tir::builtin::ptx_ldg32(),
                               {store->buffer->data, new_predicate, new_lhs, new_indice}));
        Array<Stmt> tmp_seq = {addr_store, local_addr_store, predicate_store, value_store,
                               ptx_load};
        SeqStmt seq_stmt = SeqStmt(tmp_seq);
        return seq_stmt;
      }
    }
    return result;
  }

  bool has_buffer_1 = false, has_buffer_2 = false;
  Buffer addr_buffer, predicate_buffer;
};

namespace transform {

Pass InjectPTXLDG32(bool enable_inject_ptx_intrin) {
  auto pass_func = [enable_inject_ptx_intrin](PrimFunc f, IRModule m, PassContext ctx) {
    if (enable_inject_ptx_intrin) {
      auto* n = f.CopyOnWrite();
      n->body = PTXRewriter()(n->body);
      // inject ptx
    }
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectPTXLDG32", {});
}

// The pass can now be invoked via the pass infrastructure, but we also add a
// Python binding for it
TVM_REGISTER_GLOBAL("tir.transform.InjectPTXLDG32").set_body_typed(InjectPTXLDG32);

}  // namespace transform
}  // namespace tir
}  // namespace tvm
