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
 *        parts: AllocTensor lowering, For(thread_binding) → RegionStmt(launch_thread),
 *        unit loop elimination, and pragma annotation handling.
 */

#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "ir_utils.h"

namespace tvm {
namespace tirx {
using namespace tvm::prim;

/*!
 * \brief Lower opaque constructs for TIRX: AllocTensor, thread bindings, unit loops.
 *
 * Unlike s_tirx::LowerOpaqueBlock, this pass does NOT handle SBlock/SBlockRealize,
 * since TIRX programs do not contain SBlock nodes.
 */
class TIRxOpaqueLower : public StmtExprMutator {
 public:
  using StmtExprMutator::Mutate;
  using StmtExprMutator::Mutate_;
  static Stmt Rewrite(Stmt body) {
    return ffi::make_object<TIRxOpaqueLower>()
        ->Mutate(body, InplaceMode::kAllow)
        .ValueOrUnchanged(body);
  }

 private:
  UnchangedOr<Stmt> Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode) final {
    // Attribute subjects may contain references to the loop binding being
    // replaced, including expressions and IterVar fields. Rewrite metadata
    // before the body introduces any additional local remaps.
    auto node = Mutate(op->node, inplace_mode);
    auto value = Mutate(op->value, inplace_mode);
    auto body = Mutate(op->body, inplace_mode);
    if (node.UnchangedOrSameAs(op->node) && value.UnchangedOrSameAs(op->value) &&
        body.UnchangedOrSameAs(op->body)) {
      return ffi::Unchanged();
    }
    return AttrStmt(std::move(node).ValueOrUnchanged(op->node), op->attr_key,
                    std::move(value).ValueOrUnchanged(op->value),
                    std::move(body).ValueOrUnchanged(op->body), op->span);
  }

  UnchangedOr<Stmt> Mutate_(const ForNode* op, InplaceMode inplace_mode) final {
    // Step 1. Update unit loop info.
    PrimExpr min = this->Mutate(op->min, inplace_mode).ValueOrUnchanged(op->min);
    PrimExpr extent = this->Mutate(op->extent, inplace_mode).ValueOrUnchanged(op->extent);
    auto step = this->Mutate(op->step, inplace_mode)
                    .as_or_throw<UnchangedOr<ffi::Optional<PrimExpr>>>()
                    .ValueOrUnchanged(op->step);
    ffi::Any previous_remap = VarRemapGet(op->loop_var);
    PrimVar launch_var(ffi::UnsafeInit{});
    if (op->kind == ForKind::kThreadBinding) {
      TVM_FFI_ICHECK(is_zero(min)) << "Thread binding must have zero minimum";
      launch_var = PrimVar(op->loop_var->name, extent.ty());
      VarRemapSet(op->loop_var, prim::cast(op->loop_var.ty(), launch_var));
    } else if (is_one(extent) && op->annotations.empty()) {
      VarRemapSet(op->loop_var, prim::cast(op->loop_var.ty(), min));
    }

    // Annotations share the loop binding's scope. Mutate them before the body
    // so body-local definitions cannot escape into annotation expressions.
    auto annotations = this->Mutate(op->annotations, inplace_mode)
                           .as_or_throw<UnchangedOr<ffi::Map<ffi::String, ffi::Any>>>()
                           .ValueOrUnchanged(op->annotations);
    Stmt body = this->Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
    VarRemapSet(op->loop_var, previous_remap);

    // Step 3. Handle annotations
    std::vector<std::pair<std::string, Expr>> pragma_attrs;
    ffi::Map<ffi::String, ffi::Any> new_annotations = HandleAnnotations(annotations, &pragma_attrs);
    if (op->kind != ForKind::kThreadBinding) {
      // A pragma that uses this loop's own Var must remain inside its scope.
      // Pragmas using only enclosing bindings retain their wrapper position.
      std::vector<std::pair<std::string, Expr>> outer_pragmas;
      for (auto it = pragma_attrs.rbegin(); it != pragma_attrs.rend(); ++it) {
        auto uses_loop_var = [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
          return var.same_as(op->loop_var) ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(var))
                                           : ffi::WalkResult::Advance();
        };
        if (ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(it->second, uses_loop_var).has_value()) {
          body = AttrStmt(op->loop_var, it->first, it->second, std::move(body));
        } else {
          outer_pragmas.push_back(*it);
        }
      }
      std::reverse(outer_pragmas.begin(), outer_pragmas.end());
      pragma_attrs = std::move(outer_pragmas);
    }
    // Step 4. Create new For loop accordingly
    if (op->kind == ForKind::kThreadBinding) {
      // Case 1. Thread binding → RegionStmt(launch_thread)
      TVM_FFI_ICHECK(!op->annotations.count("loop_partition_hint") ||
                     op->annotations.at("loop_partition_hint") == nullptr)
          << "Run LoopPartition before opaque lowering of a thread-binding loop with "
             "loop_partition_hint";
      TVM_FFI_ICHECK(op->thread_binding.has_value());
    } else if (is_one(extent) && op->annotations.empty()) {
      // Case 2. Unit loop elimination
      return body;
    } else {
      // Case 3. An ordinary loop
      body = For(op->loop_var, std::move(min), std::move(extent), op->kind, std::move(body),
                 std::nullopt, new_annotations, step);
    }
    // Step 5. Insert nested attrs for pragma annotations
    for (auto it = pragma_attrs.rbegin(); it != pragma_attrs.rend(); ++it) {
      Var var = op->kind == ForKind::kThreadBinding ? Var(launch_var) : Var(op->loop_var);
      body = AttrStmt(var, it->first, it->second, std::move(body));
    }
    if (op->kind == ForKind::kThreadBinding) {
      return LaunchThread(op->thread_binding.value()->thread_tag, extent, launch_var, body,
                          op->span);
    }
    return body;
  }

  /*! \brief Convert attr value from annotation map into Expr. */
  Expr ConvertAttrValue(const ffi::String& key, const Any& obj) {
    if (auto expr = obj.try_cast<Expr>()) {
      return expr.value();
    } else if (auto str = obj.try_cast<ffi::String>()) {
      return std::move(StringImm(str.value()));
    } else {
      LOG(FATAL) << "Illegal attribute of key " << key << ", value type " << obj.GetTypeKey()
                 << " not supported";
    }
  }

  /*!
   * \brief Handle loop annotation dict.
   * (1) if the attr key is prefixed by `pragma_`, move to ordered kv list
   *     (lowered to `AttrStmt` by legacy TE schedule convention), except for
   *     `pragma_unroll`, whose bool-or-integer value must remain on the loop.
   * (2) non-pragma loop annotations are preserved.
   * \return New annotation dict with preserved keys. Also update pragma attr pairs ordered by key.
   */
  ffi::Map<ffi::String, ffi::Any> HandleAnnotations(
      const ffi::Map<ffi::String, ffi::Any>& annotations,
      std::vector<std::pair<std::string, Expr>>* pragma_attrs) {
    ffi::Map<ffi::String, ffi::Any> preserved_annotations;
    pragma_attrs->clear();
    for (const auto& kv : annotations) {
      const ffi::String& key = kv.first;
      if (key == "pragma_unroll") {
        if (kv.second != nullptr) {
          preserved_annotations.Set(key, kv.second);
        }
      } else if (tirx::attr::IsPragmaKey(key)) {
        if (kv.second == nullptr) {
          continue;
        }

        pragma_attrs->emplace_back(key, ConvertAttrValue(key, kv.second));
      } else {
        // Loop annotations are always preserved
        preserved_annotations.Set(key, kv.second);
      }
    }
    std::sort(pragma_attrs->begin(), pragma_attrs->end(),
              [](const auto& p1, const auto& p2) { return p1.first < p2.first; });
    return preserved_annotations;
  }

  /*! \brief Record the loop_var and loop start value of unit loops, whose extent is one. */
};

namespace transform {

Pass LowerTIRxOpaque() {
  auto pass_func = [=](PrimFunc f, IRModule m, PassContext ctx) {
    if (!f->body.has_value()) return f;
    auto fptr = f.CopyOnWrite();
    fptr->body = TIRxOpaqueLower::Rewrite(std::move(fptr->body).value());
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
