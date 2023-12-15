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
 * \file tir/analysis/verify_well_formed.cc
 * \brief Check if schedulable tir is well-formed.
 */

#include <tvm/runtime/registry.h>
#include <tvm/tir/stmt.h>
#include <tvm/tir/stmt_functor.h>

#include <exception>
#include <optional>
#include <variant>

#include "../ir/functor_common.h"
#include "../ir/tir_visitor_with_path.h"
#include "tvm/ir/module.h"

namespace tvm {
namespace tir {

namespace {

template <typename DerivedVerifier>
class Verifier : protected TIRVisitorWithPath {
 public:
  template <typename TirNodeRef>
  static bool Verify(const TirNodeRef& node, bool assert_on_error) {
    DerivedVerifier verifier(assert_on_error);
    verifier(node);
    return !verifier.has_error_;
  }

 protected:
  Verifier(bool assert_on_error) : assert_on_error_(assert_on_error) {}

  /* \brief Helper class to handle the bool-or-assert handles
   *
   * Each verifier can either return a boolean, or assert on failure.
   * To avoid needing to duplicate this logic at every step, the
   * Verify() method can be used.  Similar to `LOG(FATAL)` or
   * `LOG(DEBUG)`, it returns an object that can accept streamed
   * context information.
   *
   * If the error should be raised, then the context is collected
   * identically to `LOG(FATAL)`.  If a boolean is returned, or if the
   * condition passes, then the streamed context is discarded.
   *
   * Usage:
   *
   *     Verify(value == expected_value)
   *            << "ValueError: " << value
   *            << " was not the expected value of " << expected_value;
   */
  class VerifyStream {
   public:
    VerifyStream(bool log_fatal) {
      if (log_fatal) {
        log_.emplace();
      }
    }

    VerifyStream(const VerifyStream&) = delete;
    VerifyStream& operator=(const VerifyStream&) = delete;
    VerifyStream(VerifyStream&& other) { std::swap(log_, other.log_); }
    VerifyStream& operator=(VerifyStream&& other) {
      std::swap(log_, other.log_);
      return *this;
    }

    template <typename T>
    VerifyStream& operator<<(T&& t) {
      if (log_.has_value()) {
        log_.value() << std::forward<T>(t);
      }
      return *this;
    }

    ~VerifyStream() noexcept(false) {
      if (log_.has_value()) {
        LOG(FATAL) << log_->str();
      }
    }

    std::optional<std::ostringstream> log_{std::nullopt};
  };

  // TODO(Lunderberg): Add the filename/linenum with
  // std::source_location when C++20 is available.
  VerifyStream Verify(bool condition) {
    has_error_ = has_error_ || !condition;
    return VerifyStream(!condition && assert_on_error_);
  }

  bool assert_on_error_;
  bool has_error_{false};
};

}  // namespace

/*! \brief Verify all Expr inside the block does not contain:
 *    1. loop vars outside the current block.
 *    2. block vars of parent blocks.
 */
class BlockVarAccessVerifier : public StmtExprVisitor {
 public:
  static bool Verify(const PrimFunc& func, bool assert_mode) {
    BlockVarAccessVerifier verifier(assert_mode);
    verifier(func->body);
    return !verifier.has_error_;
  }

 private:
  explicit BlockVarAccessVerifier(bool assert_mode) : assert_mode_(assert_mode) {}

  void VisitStmt(const Stmt& stmt) final {
    if (!has_error_) {
      StmtExprVisitor::VisitStmt(stmt);
    }
  }

  void VisitExpr(const PrimExpr& expr) final {
    if (!has_error_) {
      StmtExprVisitor::VisitExpr(expr);
    }
  }

  void VisitExpr_(const VarNode* op) final {
    auto it = loop_vars_.find(op);
    if (it != loop_vars_.end() && it->second < block_stack_.size()) {
      has_error_ = true;
      if (assert_mode_) {
        if (it->second == 0) {
          LOG(FATAL) << "Well-formedness check failed: "
                     << "Loop iterator var " << op->name_hint
                     << " is defined outside of any block, "
                     << "but is used inside the non-opaque current block \""
                     << block_stack_.back()->name_hint << "\".";
        } else {
          LOG(FATAL) << "Well-formedness check failed: "
                     << "Loop iterator var " << op->name_hint << " is defined in block \""
                     << block_stack_[it->second - 1]->name_hint << "\", "
                     << "but is used inside the non-opaque current block \""
                     << block_stack_.back()->name_hint << "\".";
        }
      }
    }
  }

  void VisitStmt_(const ForNode* op) final {
    ICHECK(loop_vars_.find(op->loop_var.get()) == loop_vars_.end());
    loop_vars_[op->loop_var.get()] = block_stack_.size();
    StmtExprVisitor::VisitStmt_(op);
    loop_vars_.erase(op->loop_var.get());
  }

  void VisitStmt_(const BlockNode* op) final {
    // Do not check boundary if it's a opaque block.
    bool is_non_opaque = op->iter_vars.size();
    if (is_non_opaque) {
      block_stack_.push_back(op);
    }

    // Step 0. Skip block iter var's domain

    // Step 1. Visit read/write regions
    auto fvisit_buffer_region = [this](const BufferRegion& s) {
      for (const auto& range : s->region) {
        this->VisitExpr(range->min);
        this->VisitExpr(range->extent);
      }
    };
    VisitArray(op->reads, fvisit_buffer_region);
    VisitArray(op->writes, fvisit_buffer_region);

    // Step 2. Visit match buffers
    VisitArray(op->match_buffers,
               [fvisit_buffer_region](const MatchBufferRegion& match_buffer_region) {
                 fvisit_buffer_region(match_buffer_region->source);
               });

    // Step 3. Visit init and body
    if (op->init.defined()) {
      this->VisitStmt(op->init.value());
    }
    this->VisitStmt(op->body);

    if (is_non_opaque) {
      block_stack_.pop_back();
    }
  }

 private:
  /*! \brief The map from outside loop vars to its corresponding block level. */
  std::unordered_map<const VarNode*, size_t> loop_vars_;
  /*! \brief Whether it's in assert mode. */
  bool assert_mode_;
  /*! \brief Current nested block stack level. */
  std::vector<const BlockNode*> block_stack_;
  /*! \brief Whether there is error. */
  bool has_error_{false};
};

class UndefinedVarVerifier : public Verifier<UndefinedVarVerifier> {
 public:
  // Until templated-this arrives in C++23, the CRTP can't inject a
  // constructor into the child class.  Therefore, must explicitly add
  // the constructor.
  using Verifier::Verifier;

 private:
  void EnterDef(const Var& var, ObjectPath path) override {
    {
      auto it = currently_defined_.find(var);
      Verify(it == currently_defined_.end())
          << "ValueError: "
          << "TIR is ill-formed, "
          << "due to multiple nested definitions of variable " << var
          << ".  It was first defined at " << it->second << ", and was re-defined at " << path;
    }

    {
      auto it = previously_defined_.find(var);
      Verify(it == previously_defined_.end())
          << "ValueError: "
          << "TIR is ill-formed, "
          << "due to multiple definitions of variable " << var << ".  It was first defined at "
          << it->second << ", and was later re-defined at " << path;
    }

    currently_defined_.insert({var, path});
  }

  void ExitDef(const Var& var, ObjectPath path) override {
    auto active_def = currently_defined_.find(var);

    currently_defined_.erase(active_def);
    previously_defined_.insert({var, path});
  }

  void VisitExpr_(const VarNode* op, ObjectPath path) override {
    auto var = GetRef<Var>(op);

    auto prev_def = previously_defined_.find(var);
    Verify(prev_def == previously_defined_.end())
        << "ValueError: "
        << "Invalid use of variable " << var << " at " << path << ".  "
        << "While this variable was previously defined at " << prev_def->second
        << ", this definition is no longer in-scope.";

    auto active_def = currently_defined_.find(var);
    Verify(active_def != currently_defined_.end())
        << "ValueError: "
        << "Invalid use of undefined variable " << var << " at " << path;
  }

  std::unordered_map<Var, ObjectPath, ObjectPtrHash, ObjectPtrEqual> currently_defined_;
  std::unordered_map<Var, ObjectPath, ObjectPtrHash, ObjectPtrEqual> previously_defined_;
};

bool VerifyWellFormed(const PrimFunc& func, bool assert_mode) {
  if (!BlockVarAccessVerifier::Verify(func, assert_mode)) {
    return false;
  }

  if (!UndefinedVarVerifier::Verify(func, assert_mode)) return false;

  // TODO(Siyuan): add more checks here.
  return true;
}

bool VerifyWellFormed(const IRModule& mod, bool assert_mode) {
  for (const auto& [gvar, base_func] : mod->functions) {
    if (auto prim_func = base_func.as<PrimFunc>()) {
      bool res = VerifyWellFormed(prim_func.value(), assert_mode);
      if (!res) {
        return false;
      }
    }
  }

  if (!UndefinedVarVerifier::Verify(mod, assert_mode)) return false;

  return true;
}

TVM_REGISTER_GLOBAL("tir.analysis.VerifyWellFormed")
    .set_body_typed([](const ObjectRef& obj, bool assert_mode) {
      if (auto opt = obj.as<PrimFunc>()) {
        return VerifyWellFormed(opt.value(), assert_mode);
      } else if (auto opt = obj.as<IRModule>()) {
        return VerifyWellFormed(opt.value(), assert_mode);
      } else {
        LOG(FATAL) << "Expected VerifyWellFormed argument to be a PrimFunc or IRModule, but found "
                   << obj->GetTypeKey();
      }
    });

}  // namespace tir
}  // namespace tvm
