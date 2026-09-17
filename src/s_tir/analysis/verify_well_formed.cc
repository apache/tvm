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

#include "../../tirx/analysis/verify_well_formed.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/analysis.h>
#include <tvm/s_tir/stmt_functor.h>

#include "../ir/tir_visitor_with_path.h"

namespace tvm {
namespace s_tir {
using tirx::BufferRegion;
using tirx::ForNode;
using tirx::PrimFunc;

/*! \brief Verify all Expr inside the block does not contain:
 *    1. loop vars outside the current block.
 *    2. block vars of parent blocks.
 */
class BlockVarAccessVerifier : public StmtExprVisitor {
 public:
  static bool Verify(const PrimFunc& func, bool assert_mode) {
    auto verifier = ffi::make_object<BlockVarAccessVerifier>(assert_mode);
    verifier->Visit(func->body);
    return !verifier->has_error_;
  }

  explicit BlockVarAccessVerifier(bool assert_mode) : assert_mode_(assert_mode) {}

 private:
  ffi::Optional<VisitInterrupt> Visit(ffi::AnyView stmt) final {
    if (!has_error_) {
      return StmtExprVisitor::Visit(stmt);
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const VarNode* op) final {
    auto it = loop_vars_.find(op);
    if (it != loop_vars_.end() && it->second < block_stack_.size()) {
      has_error_ = true;
      if (assert_mode_) {
        if (it->second == 0) {
          TVM_FFI_THROW(InternalError)
              << "Well-formedness check failed: "
              << "Loop iterator var " << op->name << " is defined outside of any block, "
              << "but is used inside the non-opaque current block \""
              << block_stack_.back()->name_hint << "\".";
        } else {
          TVM_FFI_THROW(InternalError)
              << "Well-formedness check failed: "
              << "Loop iterator var " << op->name << " is defined in block \""
              << block_stack_[it->second - 1]->name_hint << "\", "
              << "but is used inside the non-opaque current block \""
              << block_stack_.back()->name_hint << "\".";
        }
      }
    }
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const ForNode* op) final {
    TVM_FFI_ICHECK(loop_vars_.find(op->loop_var.get()) == loop_vars_.end());
    loop_vars_[op->loop_var.get()] = block_stack_.size();
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(StmtExprVisitor::Visit_(op));
    loop_vars_.erase(op->loop_var.get());
    return std::nullopt;
  }

  ffi::Optional<VisitInterrupt> Visit_(const SBlockNode* op) final {
    // Do not check boundary if it's a opaque block.
    bool is_non_opaque = op->iter_vars.size();
    if (is_non_opaque) {
      block_stack_.push_back(op);
    }

    // Step 0. Skip block iter var's domain

    // Step 1. Visit read/write regions
    auto fvisit_buffer_region = [this](const BufferRegion& s) -> ffi::Optional<VisitInterrupt> {
      for (const auto& range : s->region) {
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(range->min));
        TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(range->extent));
      }
      return std::nullopt;
    };
    for (const auto& region : op->reads) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(fvisit_buffer_region(region));
    }
    for (const auto& region : op->writes) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(fvisit_buffer_region(region));
    }

    // Step 2. Visit match buffers
    for (const auto& match_buffer_region : op->match_buffers) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(fvisit_buffer_region(match_buffer_region->source));
    }

    // Step 3. Visit init and body
    if (op->init.has_value()) {
      TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->init.value()));
    }
    TVM_FFI_S_VISIT_MAYBE_EARLY_RETURN(this->Visit(op->body));

    if (is_non_opaque) {
      block_stack_.pop_back();
    }
    return std::nullopt;
  }

 private:
  /*! \brief The map from outside loop vars to its corresponding block level. */
  std::unordered_map<const VarNode*, size_t> loop_vars_;
  /*! \brief Whether it's in assert mode. */
  bool assert_mode_;
  /*! \brief Current nested block stack level. */
  std::vector<const SBlockNode*> block_stack_;
  /*! \brief Whether there is error. */
  bool has_error_{false};
};

bool VerifyWellFormed(const tirx::PrimFunc& func, bool assert_mode) {
  return BlockVarAccessVerifier::Verify(func, assert_mode) &&
         tirx::VerifyWellFormedCommon<TIRVisitorWithPath>(func, assert_mode);
}

bool VerifyWellFormed(const IRModule& mod, bool assert_mode) {
  for (const auto& [gvar, base_func] : mod->functions) {
    if (auto func = base_func.as<tirx::PrimFunc>()) {
      if (!BlockVarAccessVerifier::Verify(func.value(), assert_mode)) return false;
    }
  }
  return tirx::VerifyWellFormedCommon<TIRVisitorWithPath>(mod, assert_mode);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::GlobalDef().def(
      "s_tir.analysis.VerifyWellFormed", [](const ffi::ObjectRef& obj, bool assert_mode) {
        if (auto func = obj.as<tirx::PrimFunc>()) {
          return s_tir::VerifyWellFormed(func.value(), assert_mode);
        }
        if (auto mod = obj.as<IRModule>()) {
          return s_tir::VerifyWellFormed(mod.value(), assert_mode);
        }
        TVM_FFI_THROW(TypeError) << "Expected a PrimFunc or IRModule, but received "
                                 << obj->GetTypeKey();
      });
}
}  // namespace s_tir
}  // namespace tvm
