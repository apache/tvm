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
 * \brief Replace copy from global to shared with async copy
 * \file inject_ptx_async_copy.cc
 */
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../ir/buffer_common.h"
#include "storage_access.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace tir {

class PTXAsyncCopyInjector : public StmtMutator {
 public:
  Stmt VisitStmt_(const AttrStmtNode* attr) {
    if (attr->attr_key == tir::attr::async_scope) {
      in_async = true;
      auto body = this->VisitStmt(attr->body);
      in_async = false;
      return body;
    }
    return StmtMutator::VisitStmt_(attr);
  }

  Stmt VisitStmt_(const BufferStoreNode* store) {
    if (in_async && (store->buffer.scope() == "shared" || store->buffer.scope() == "shared.dyn")) {
      if (auto* load = store->value.as<BufferLoadNode>()) {
        if (load->buffer.scope() == "global") {
          ICHECK(load->indices.size() == 1 && store->indices.size() == 1);
	  ICHECK(load->indices[0]->dtype.lanes() == store->indices[0]->dtype.lanes());

	  const int indices_lanes = load->indices[0]->dtype.lanes();
          const int bytes = indices_lanes * load->buffer->dtype.bytes();

          if (bytes == 4 || bytes == 8 || bytes == 16) {
            auto dst_elem_type = GetPointerType(store->buffer->data->type_annotation);
            auto src_elem_type = GetPointerType(load->buffer->data->type_annotation);
            ICHECK(dst_elem_type.first && src_elem_type.first)
                << "Both store and load buffer should have a pointer type annotation.";

            int index_factor = 1;
            if (dst_elem_type != src_elem_type) {
              // The only case where src and dst have different dtypes is when the dst shared memory
              // is a byte buffer generated by merging dynamic shared memory.
              ICHECK(store->buffer.scope() == "shared.dyn");
              ICHECK(dst_elem_type.second == DataType::UInt(8));
              // BufferStore/Load have the "pointer reinterpret" semantics according to their
              // "value" dtype. Their "indices" are supposed to be applied after such pointer cast,
              // for example: ((*float16)(byte_buffer))[buffer->indices] = fp16_value;
              // To replace BufferStore/Load with cp.async, we need to multiply the store index by
              // the byte size of the "value" dtype, to get the correct offset into the byte buffer.
              index_factor = src_elem_type.second.bytes();
            }

            if (indices_lanes == 1) {
	      auto src_offset = load->indices[0];
	      auto dst_offset = store->indices[0];
              return Evaluate(
                  Call(store->buffer->dtype, tvm::tir::builtin::ptx_cp_async(),
                       {store->buffer->data, tir::Mul(dst_offset, PrimExpr(index_factor)),
                        load->buffer->data, src_offset, PrimExpr(bytes)}));
            }

	    // Only some vectorized indexing patterns are supported for now.
            auto src_offset = [=]() -> PrimExpr {
              if (load->indices[0]->IsInstance<RampNode>()) {
                return load->indices[0].as<RampNode>()->base;
              }
	      return PrimExpr();
            }();

            auto dst_offset = [=]() -> PrimExpr {
              if (store->indices[0].as<RampNode>()) {
                return store->indices[0].as<RampNode>()->base;
              } else if (store->indices[0].as<AddNode>()) {
                // The case where the dst buffer is a byte buffer generated by merging dynamic
                // shared memory.
                // A_shared.dyn[(ramp(...), 1, 8) + x8(17408))] = A_global[ramp(...),1, 8)]
                auto* add = store->indices[0].as<AddNode>();
                if (!add->a->IsInstance<RampNode>()) return PrimExpr();
                if (!add->b->IsInstance<BroadcastNode>()) return PrimExpr();
                return tir::Add(add->a.as<RampNode>()->base, add->b.as<BroadcastNode>()->value);
              }
              return PrimExpr();
            }();

            if (src_offset.defined() && dst_offset.defined()) {
              return Evaluate(
                  Call(store->buffer->dtype, tvm::tir::builtin::ptx_cp_async(),
                       {store->buffer->data, tir::Mul(dst_offset, PrimExpr(index_factor)),
                        load->buffer->data, src_offset, PrimExpr(bytes)}));
            }
          }
        }
      }
    }
    return StmtMutator::VisitStmt_(store);
  }

 private:
  bool in_async{false};
};

namespace transform {

Pass InjectPTXAsyncCopy() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto* n = f.CopyOnWrite();
    n->body = PTXAsyncCopyInjector()(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.InjectPTXAsyncCopy", {});
}

TVM_REGISTER_GLOBAL("tir.transform.InjectPTXAsyncCopy").set_body_typed(InjectPTXAsyncCopy);

}  // namespace transform

}  // namespace tir
}  // namespace tvm
