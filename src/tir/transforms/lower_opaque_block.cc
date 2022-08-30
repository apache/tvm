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
 * \file lower_opaque_block.cc
 */

#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace tir {

/*!
 * \brief Remove Block to ensure that the TIR can not be scheduled again.
 */
class OpaqueBlockLower : public StmtExprMutator {
 private:
  Stmt VisitStmt_(const BlockRealizeNode* op) final {
    // We have convert blocks into opaque blocks in previous passes.
    ICHECK(op->iter_values.empty()) << "Non-opaque blocks are not allowed in FlattenBuffer. Please "
                                       "call pass ConvertBlocksToOpaque before.";
    // Step 1. Visit the body
    Block new_block = Downcast<Block>(this->VisitStmt(op->block));
    PrimExpr predicate = this->VisitExpr(op->predicate);
    // Step 2. Transform the `predicate` to if-then-else
    Stmt body = new_block->body;
    if (!is_one(predicate)) {
      body = IfThenElse(predicate, std::move(body));
    }
    // Step 3. Handle allocations in reverse order
    for (size_t i = new_block->alloc_buffers.size(); i > 0; --i) {
      const Buffer& buffer = new_block->alloc_buffers[i - 1];
      Array<PrimExpr> new_shape = buffer->shape;
      if (buffer->strides.size()) {
        ICHECK_EQ(buffer->shape.size(), buffer->strides.size());
        for (size_t i = buffer->strides.size() - 1; i > 0; --i) {
          ICHECK(is_zero(floormod(buffer->strides[i - 1], buffer->strides[i])));
          new_shape.Set(i, buffer->strides[i - 1] / buffer->strides[i]);
        }
      }
      body = Allocate(buffer->data, buffer->dtype, new_shape, const_true(), std::move(body));
    }
    // Step 4. Handle annotations, block annotations are not preserved by default.
    std::vector<std::pair<std::string, PrimExpr>> pragma_attrs;
    HandleAnnotations(new_block->annotations, &pragma_attrs, /*is_block=*/true);
    for (auto it = pragma_attrs.rbegin(); it != pragma_attrs.rend(); ++it) {
      body = AttrStmt(Integer(0), it->first, it->second, std::move(body));
    }
    return body;
  }

  Stmt VisitStmt_(const ForNode* op) final {
    // Step 1. Update unit loop info.
    PrimExpr min = this->VisitExpr(op->min);
    PrimExpr extent = this->VisitExpr(op->extent);
    if (is_one(extent) && op->annotations.empty()) {
      // handling unit loop
      unit_loop_vars_[op->loop_var] = min;
    }
    // Step 2. Visit recursively
    Stmt body = this->VisitStmt(op->body);
    // Step 3. Handle annotations
    std::vector<std::pair<std::string, PrimExpr>> pragma_attrs;
    Map<String, ObjectRef> new_annotations =
        HandleAnnotations(op->annotations, &pragma_attrs, /*is_block=*/false);
    // Step 4. Create new For loop accordingly
    if (op->kind == ForKind::kThreadBinding) {
      // Case 1. Thread binding
      ICHECK(op->thread_binding.defined());
      String thread_tag = op->thread_binding.value()->thread_tag;
      body = MakeLaunchThread(min, extent, op->loop_var, thread_tag, body);
    } else if (is_one(extent) && op->annotations.empty()) {
      // Case 2. Unit loop
      return body;
    } else {
      // Case 3. An ordinary loop
      body = For(op->loop_var, std::move(min), std::move(extent), op->kind, std::move(body),
                 NullOpt, new_annotations);
    }
    // Step 5. Insert nested attrs
    for (auto it = pragma_attrs.rbegin(); it != pragma_attrs.rend(); ++it) {
      body = AttrStmt(op->loop_var, it->first, it->second, std::move(body));
    }
    return body;
  }

  PrimExpr VisitExpr_(const VarNode* op) final {
    Var var = GetRef<Var>(op);
    auto it = unit_loop_vars_.find(var);
    if (it == unit_loop_vars_.end()) {
      return std::move(var);
    } else {
      PrimExpr expr = it->second;
      if (expr.dtype() != var.dtype()) {
        expr = tvm::cast(var.dtype(), std::move(expr));
      }
      return expr;
    }
  }

  static Stmt MakeLaunchThread(PrimExpr min, PrimExpr extent, Var var, String thread_tag,
                               Stmt body) {
    IterVar iter_var(/*dom=*/Range::FromMinExtent(min, extent),
                     /*var=*/std::move(var),
                     /*iter_type=*/IterVarType::kThreadIndex,
                     /*thread_tag=*/thread_tag);
    String attr_key = (thread_tag == "vthread" || thread_tag == "vthread.x" ||
                       thread_tag == "vthread.y" || thread_tag == "vthread.z")
                          ? attr::virtual_thread
                          : attr::thread_extent;
    return AttrStmt(/*node=*/std::move(iter_var),
                    /*attr_key=*/std::move(attr_key),
                    /*value=*/std::move(extent),
                    /*body=*/std::move(body));
  }

  /*! \brief Convert attr value from annotation map into PrimExpr. */
  PrimExpr ConvertAttrValue(const String& key, const ObjectRef& obj) {
    if (!obj.defined()) {
      return PrimExpr();
    } else if (const PrimExprNode* expr = obj.as<PrimExprNode>()) {
      return GetRef<PrimExpr>(expr);
    } else if (const StringObj* str = obj.as<StringObj>()) {
      return std::move(StringImm(str->data));
    } else {
      LOG(FATAL) << "Illegal attribute of key " << key << ", value type " << obj->GetTypeKey()
                 << " not supported";
      return PrimExpr();
    }
  }

  /*!
   * \brief Helper to handle annotation dict.
   * (1) if the attr key is prefixed by `pragma_`, move to ordered kv list. They
   * are lowered to `AttrStmt` by legacy TE schedule convention.
   * (2) the non-pragma loop annotations are preserved
   * (3) the non-pragma block annotations are dropped
   * \return New annotation dict with preserved keys. Also update pragma attr pairs ordered by key.
   */
  Map<String, ObjectRef> HandleAnnotations(
      const Map<String, ObjectRef>& annotations,
      std::vector<std::pair<std::string, PrimExpr>>* pragma_attrs, bool is_block) {
    Map<String, ObjectRef> preserved_annotations;
    pragma_attrs->clear();
    for (const auto& kv : annotations) {
      const String& key = kv.first;
      if (attr::IsPragmaKey(key)) {
        pragma_attrs->emplace_back(key, ConvertAttrValue(key, kv.second));
      } else if (!is_block) {
        // the loop annotation is preserved
        preserved_annotations.Set(key, kv.second);
      }
    }
    std::sort(pragma_attrs->begin(), pragma_attrs->end(),
              [](const auto& p1, const auto& p2) { return p1.first < p2.first; });
    return preserved_annotations;
  }

  /*! \brief Record the loop_var and loop start value of unit loops, whose extent is one. */
  std::unordered_map<Var, PrimExpr, ObjectPtrHash, ObjectPtrEqual> unit_loop_vars_;

  /*! \brief Attr keys to preserve into loop annotations. */
  std::unordered_set<std::string> preserved_annotations_;
};

PrimFunc LowerOpaqueBlock(PrimFunc f) {
  // Only apply this pass to TIR that is not from TE schedules
  if (!IsFromLegacyTESchedule(f)) {
    auto fptr = f.CopyOnWrite();
    fptr->body = OpaqueBlockLower()(std::move(fptr->body));
    return f;
  } else {
    return f;
  }
}

namespace transform {

Pass LowerOpaqueBlock() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    return LowerOpaqueBlock(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tir.LowerOpaqueBlock", {});
}

TVM_REGISTER_GLOBAL("tir.transform.LowerOpaqueBlock").set_body_typed(LowerOpaqueBlock);
}  // namespace transform

}  // namespace tir
}  // namespace tvm
