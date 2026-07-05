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
 * \file lower_tirx_opaque.cc
 * \brief Lower opaque constructs in TIRX programs. This is the tirx-specific
 *        counterpart of s_tirx::LowerOpaqueBlock, handling only the non-SBlock
 *        parts: AllocBuffer lowering, For(thread_binding) → AttrStmt(thread_extent),
 *        unit loop elimination, and pragma annotation handling.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace tirx {

/*!
 * \brief Lower opaque constructs for TIRX: AllocBuffer, thread bindings, unit loops.
 *
 * Unlike s_tirx::LowerOpaqueBlock, this pass does NOT handle SBlock/SBlockRealize,
 * since TIRX programs do not contain SBlock nodes.
 */
class TIRxOpaqueLower : public StmtExprMutator {
 public:
  static Stmt Rewrite(Stmt body) {
    TIRxOpaqueLower lower;
    lower.pool_sizes_ = CollectPoolSizes(body);
    return lower(std::move(body));
  }

 private:
  static std::unordered_map<Var, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> CollectPoolSizes(
      const Stmt& body) {
    class Collector : public StmtVisitor {
     public:
      void VisitStmt_(const AttrStmtNode* op) final {
        if (op->attr_key == "tirx.pool_max_bytes") {
          if (auto var = op->node.try_cast<Var>()) {
            const auto* n = op->value.as<IntImmNode>();
            TVM_FFI_ICHECK(n) << "TIRxError: tirx.pool_max_bytes must be IntImm";
            pool_sizes_[var.value()] = n->value;
          }
        }
        StmtVisitor::VisitStmt_(op);
      }

      std::unordered_map<Var, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> pool_sizes_;
    };

    Collector collector;
    collector(body);
    return std::move(collector.pool_sizes_);
  }

  Stmt VisitStmt_(const AttrStmtNode* op) final {
    if (op->attr_key == "tirx.pool_max_bytes") {
      // Strip the pool size AttrStmt after pre-collection in Rewrite().
      return VisitStmt(op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const AllocBufferNode* op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    op = stmt.as<AllocBufferNode>();
    TVM_FFI_ICHECK(op);

    Buffer alloc_buf = op->buffer;
    auto it = pool_sizes_.find(op->buffer->data);
    if (it != pool_sizes_.end()) {
      auto* n = alloc_buf.CopyOnWrite();
      n->shape = {IntImm::Int64(it->second)};
    }
    if (alloc_buf.same_as(op->buffer)) {
      return stmt;
    }
    auto n = CopyOnWrite(op);
    n->buffer = std::move(alloc_buf);
    return Stmt(n);
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // Step 1. Update unit loop info.
    PrimExpr min = this->VisitPrimExpr(op->min);
    PrimExpr extent = this->VisitPrimExpr(op->extent);
    if (is_one(extent) && op->annotations.empty()) {
      // handling unit loop
      unit_loop_vars_[op->loop_var] = min;
    }

    // Step 2. Visit recursively
    Stmt body = this->VisitStmt(op->body);

    // Step 3. Handle annotations
    std::vector<std::pair<std::string, PrimExpr>> pragma_attrs;
    ffi::Map<ffi::String, ffi::Any> new_annotations =
        HandleAnnotations(op->annotations, &pragma_attrs);
    // Step 4. Create new For loop accordingly
    if (op->kind == ForKind::kThreadBinding) {
      // Case 1. Thread binding → AttrStmt(thread_extent)
      TVM_FFI_ICHECK(op->thread_binding.defined());
      ffi::String thread_tag = op->thread_binding.value()->thread_tag;
      body = MakeLaunchThread(min, extent, op->loop_var, thread_tag, body);
    } else if (is_one(extent) && op->annotations.empty() &&
               !op->annotations.count(tirx::attr::irregular_loop_mark)) {
      // Case 2. Unit loop elimination
      return body;
    } else {
      // Case 3. An ordinary loop
      body = For(op->loop_var, std::move(min), std::move(extent), op->kind, std::move(body),
                 std::nullopt, new_annotations, op->step);
    }
    // Step 5. Insert nested attrs for pragma annotations
    for (auto it = pragma_attrs.rbegin(); it != pragma_attrs.rend(); ++it) {
      body = AttrStmt(op->loop_var, it->first, it->second, std::move(body));
    }
    return body;
  }

  Expr VisitExpr_(const VarNode* op) final {
    Var var = ffi::GetRef<Var>(op);
    auto it = unit_loop_vars_.find(var);
    if (it == unit_loop_vars_.end()) {
      return var;
    } else {
      PrimExpr expr = it->second;
      PrimType var_ty = var->ty.as_or_throw<PrimType>();
      if (expr.ty() != var_ty) {
        expr = tvm::cast(var_ty, std::move(expr));
      }
      return expr;
    }
  }

  static Stmt MakeLaunchThread(PrimExpr min, PrimExpr extent, Var var, ffi::String thread_tag,
                               Stmt body) {
    IterVar iter_var(/*dom=*/Range::FromMinExtent(min, extent),
                     /*var=*/std::move(var).as_or_throw<PrimVar>(),
                     /*iter_type=*/IterVarType::kThreadIndex,
                     /*thread_tag=*/thread_tag);
    ffi::String attr_key = (thread_tag == "vthread" || thread_tag == "vthread.x" ||
                            thread_tag == "vthread.y" || thread_tag == "vthread.z")
                               ? s_tir::attr::virtual_thread
                               : tirx::attr::thread_extent;
    return AttrStmt(/*node=*/std::move(iter_var),
                    /*attr_key=*/std::move(attr_key),
                    /*value=*/std::move(extent),
                    /*body=*/std::move(body));
  }

  /*! \brief Convert attr value from annotation map into PrimExpr. */
  PrimExpr ConvertAttrValue(const ffi::String& key, const Any& obj) {
    if (obj == nullptr) {
      return PrimExpr();
    } else if (auto expr = obj.try_cast<PrimExpr>()) {
      return expr.value();
    } else if (auto str = obj.try_cast<ffi::String>()) {
      return std::move(StringImm(str.value()));
    } else {
      LOG(FATAL) << "Illegal attribute of key " << key << ", value type " << obj.GetTypeKey()
                 << " not supported";
      return PrimExpr();
    }
  }

  /*!
   * \brief Handle loop annotation dict.
   * (1) if the attr key is prefixed by `pragma_`, move to ordered kv list
   *     (lowered to `AttrStmt` by legacy TE schedule convention).
   * (2) non-pragma loop annotations are preserved.
   * \return New annotation dict with preserved keys. Also update pragma attr pairs ordered by key.
   */
  ffi::Map<ffi::String, ffi::Any> HandleAnnotations(
      const ffi::Map<ffi::String, ffi::Any>& annotations,
      std::vector<std::pair<std::string, PrimExpr>>* pragma_attrs) {
    ffi::Map<ffi::String, ffi::Any> preserved_annotations;
    pragma_attrs->clear();
    for (const auto& kv : annotations) {
      const ffi::String& key = kv.first;
      if (tirx::attr::IsPragmaKey(key)) {
        pragma_attrs->emplace_back(key, ConvertAttrValue(key, kv.second));
      } else {
        // loop annotations are always preserved (no SBlock annotation dropping here)
        preserved_annotations.Set(key, kv.second);
      }
    }
    std::sort(pragma_attrs->begin(), pragma_attrs->end(),
              [](const auto& p1, const auto& p2) { return p1.first < p2.first; });
    return preserved_annotations;
  }

  /*! \brief Record the loop_var and loop start value of unit loops, whose extent is one. */
  std::unordered_map<Var, PrimExpr> unit_loop_vars_;
  /*! \brief Pool size annotations: buffer data var → size in bytes. */
  std::unordered_map<Var, int64_t, ffi::ObjectPtrHash, ffi::ObjectPtrEqual> pool_sizes_;
};

namespace transform {

Pass LowerTIRxOpaque() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    auto fptr = f.CopyOnWrite();
    fptr->body = TIRxOpaqueLower::Rewrite(std::move(fptr->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tirx.LowerTIRxOpaque", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.LowerTIRxOpaque", LowerTIRxOpaque);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
