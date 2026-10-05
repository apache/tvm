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

#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/s_tir/stmt_functor.h>
#include <tvm/s_tir/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace s_tir {
using namespace tvm::prim;
using namespace tvm::tirx;

/*!
 * \brief Remove SBlock to ensure that the TIR can not be scheduled again.
 */
class OpaqueBlockLower : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;

  static Stmt Rewrite(Stmt body) {
    auto lower = ffi::make_object<OpaqueBlockLower>();
    lower->storage_align_ = CollectStorageAlignAnnotation(body);
    return lower->Mutate(body, InplaceMode::kAllow).ValueOrUnchanged(std::move(body));
  }

 private:
  UnchangedOr<Stmt> Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode) final {
    // Attribute metadata can reference a unit loop variable that is replaced.
    // Rewrite it before descending into body-local definitions.
    auto node = this->Mutate(op->node, inplace_mode);
    auto value = this->Mutate(op->value, inplace_mode);
    auto body = this->Mutate(op->body, inplace_mode);
    if (node.UnchangedOrSameAs(op->node) && value.UnchangedOrSameAs(op->value) &&
        body.UnchangedOrSameAs(op->body)) {
      return ffi::Unchanged();
    }
    return AttrStmt(std::move(node).ValueOrUnchanged(op->node), op->attr_key,
                    std::move(value).ValueOrUnchanged(op->value),
                    std::move(body).ValueOrUnchanged(op->body), op->span);
  }

  UnchangedOr<Stmt> Mutate_(const SBlockRealizeNode* op, InplaceMode inplace_mode) final {
    // We have convert blocks into opaque blocks in previous passes.
    TVM_FFI_ICHECK(op->iter_values.empty())
        << "Non-opaque blocks are not allowed in FlattenBuffer. Please "
           "call pass ConvertBlocksToOpaque before.";
    // Block policy belongs to the loops inside the block after opaque lowering.
    auto enclosing_policy = unroll_policy_;
    UpdateUnrollPolicy(op->block->annotations);
    SBlock new_block =
        this->Mutate(op->block, inplace_mode).ValueOrUnchanged(op->block).as_or_throw<SBlock>();
    unroll_policy_ = std::move(enclosing_policy);
    PrimExpr predicate = this->Mutate(op->predicate, inplace_mode).ValueOrUnchanged(op->predicate);
    // Step 2. Transform the `predicate` to if-then-else
    Stmt body = new_block->body;
    if (!is_one(predicate)) {
      body = IfThenElse(predicate, std::move(body));
    }
    // Step 3. Handle allocations in reverse order
    for (size_t i = new_block->alloc_buffers.size(); i > 0; --i) {
      const TensorVar& buffer = new_block->alloc_buffers[i - 1];
      ffi::Map<ffi::String, ffi::Any> allocate_annotations;
      auto it = storage_align_.find(buffer.var());
      if (it != storage_align_.end()) {
        StorageAlignAnnotation allocate_aligns;
        for (auto tuple : it->second) {
          tuple.Set<0>(-1);
          allocate_aligns.push_back(tuple);
        }
        allocate_annotations.Set(s_tir::attr::buffer_dim_align, allocate_aligns);
      }
      allocate_annotations.Set(tirx::attr::buffer_data_alignment,
                               IntImm::Int32(buffer->data_alignment));
      allocate_annotations.Set(tirx::attr::buffer_allocated_addr, buffer->allocated_addr);
      body = SeqStmt::Flatten(
          Bind(buffer.var(), Call(buffer.type(), tirx::builtin::alloc_tensor(),
                                  {tvm::Tuple(buffer->shape), DataTypeImm(buffer->dtype->dtype),
                                   StringImm(buffer.scope())},
                                  DictAttrs(allocate_annotations))),
          std::move(body));
    }
    // Step 4. Handle annotations, block annotations are not preserved by default.
    std::vector<std::pair<std::string, Expr>> pragma_attrs;
    HandleAnnotations(new_block->annotations, &pragma_attrs, /*is_block=*/true);
    for (auto it = pragma_attrs.rbegin(); it != pragma_attrs.rend(); ++it) {
      body = AttrStmt(0, it->first, it->second, std::move(body));
    }
    return body;
  }

  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final {
    // Step 1. Update unit loop info.
    PrimExpr min = this->Mutate(op->min, inplace_mode).ValueOrUnchanged(op->min);
    PrimExpr extent = this->Mutate(op->extent, inplace_mode).ValueOrUnchanged(op->extent);
    auto step = this->Mutate(op->step, inplace_mode)
                    .as_or_throw<UnchangedOr<ffi::Optional<PrimExpr>>>()
                    .ValueOrUnchanged(op->step);
    auto previous_remap = VarRemapGet(op->loop_var);
    if (op->kind != ForKind::kThreadBinding && is_one(extent) && op->annotations.empty()) {
      // handling unit loop
      VarRemapSet(op->loop_var, prim::cast(op->loop_var.ty(), min));
    }

    // Keep policy on surviving descendants when this owner is lowered away.
    auto enclosing_policy = unroll_policy_;
    UpdateUnrollPolicy(op->annotations);
    auto annotations = op->annotations;
    for (const auto& kv : unroll_policy_) {
      annotations.Set(kv.first, kv.second);
    }

    // Step 2. Annotations may refer to the loop's own variable. Rewrite them
    // before visiting body-local definitions.
    std::vector<std::pair<std::string, Expr>> pragma_attrs;
    ffi::Map<ffi::String, ffi::Any> new_annotations =
        HandleAnnotations(annotations, &pragma_attrs, /*is_block=*/false);

    // Step 3. Visit recursively.
    Stmt body = this->Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
    unroll_policy_ = std::move(enclosing_policy);
    VarRemapSet(op->loop_var, previous_remap);

    // Step 4. Keep thread-binding loops until LowerThreadBinding.
    if (op->kind != ForKind::kThreadBinding && is_one(extent) && op->annotations.empty() &&
        !op->annotations.count(s_tir::attr::irregular_loop_mark)) {
      return body;
    }
    if (op->kind != ForKind::kThreadBinding) {
      body = For(op->loop_var, min, extent, op->kind, std::move(body), op->thread_binding,
                 new_annotations, step);
    }
    // Step 5. Insert nested attrs inside thread-binding scope.
    for (auto it = pragma_attrs.rbegin(); it != pragma_attrs.rend(); ++it) {
      body = AttrStmt(op->loop_var, it->first, it->second, std::move(body));
    }
    if (op->kind == ForKind::kThreadBinding) {
      body = For(op->loop_var, std::move(min), std::move(extent), op->kind, std::move(body),
                 op->thread_binding, std::move(new_annotations), std::move(step), op->span);
    }
    return body;
  }

  void UpdateUnrollPolicy(const ffi::Map<ffi::String, ffi::Any>& annotations) {
    for (const char* key : {tirx::attr::auto_unroll_max_step, tirx::attr::unroll_explicit}) {
      if (auto value = annotations.Get(key); value.has_value() && value.value() != nullptr) {
        unroll_policy_.Set(key, value.value());
      }
    }
  }

  // Effective policy is materialized on each surviving loop, preserving nested overrides.
  ffi::Map<ffi::String, ffi::Any> unroll_policy_;

  /*! \brief Convert attr value from annotation map into Expr. */
  Expr ConvertAttrValue(const ffi::String& key, const Any& obj) {
    if (auto expr = obj.try_cast<Expr>()) {
      return expr.value();
    } else if (auto str = obj.try_cast<ffi::String>()) {
      return std::move(StringImm(str.value()));
    } else {
      TVM_FFI_THROW(InternalError) << "Illegal attribute of key " << key << ", value type "
                                   << obj.GetTypeKey() << " not supported";
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
  ffi::Map<ffi::String, ffi::Any> HandleAnnotations(
      const ffi::Map<ffi::String, ffi::Any>& annotations,
      std::vector<std::pair<std::string, Expr>>* pragma_attrs, bool is_block) {
    ffi::Map<ffi::String, ffi::Any> preserved_annotations;
    pragma_attrs->clear();
    for (const auto& kv : annotations) {
      const ffi::String& key = kv.first;
      if ((key == tirx::attr::auto_unroll_max_step || key == tirx::attr::unroll_explicit) &&
          kv.second == nullptr) {
        continue;
      }
      if (tirx::attr::IsPragmaKey(key)) {
        if (kv.second == nullptr) {
          continue;
        }

        auto value = this->Mutate(kv.second, InplaceMode::kDisallow).ValueOrUnchanged(kv.second);
        pragma_attrs->emplace_back(key, ConvertAttrValue(key, value));
      } else if (!is_block) {
        // Preserve the annotation while remapping enclosing launch bindings,
        // including expressions nested in annotation containers.
        auto value = this->Mutate(kv.second, InplaceMode::kDisallow).ValueOrUnchanged(kv.second);
        preserved_annotations.Set(key, value);
      }
    }
    std::sort(pragma_attrs->begin(), pragma_attrs->end(),
              [](const auto& p1, const auto& p2) { return p1.first < p2.first; });
    return preserved_annotations;
  }

  /*! \brief Record the loop_var and loop start value of unit loops, whose extent is one. */

  /*! \brief Attr keys to preserve into loop annotations. */
  std::unordered_set<std::string> preserved_annotations_;

  /*! \brief The map from buffer var to its storage alignment information. */
  std::unordered_map<Var, StorageAlignAnnotation> storage_align_;
};

namespace transform {

Pass LowerOpaqueBlock() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    auto fptr = f.CopyOnWrite();
    fptr->body = OpaqueBlockLower::Rewrite(std::move(fptr->body).value());
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "s_tir.LowerOpaqueBlock", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("s_tir.transform.LowerOpaqueBlock", LowerOpaqueBlock);
}
}  // namespace transform

}  // namespace s_tir
}  // namespace tvm
