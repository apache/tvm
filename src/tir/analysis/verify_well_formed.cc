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

#include <sstream>

#include "../ir/functor_common.h"
#include "tvm/ir/module.h"

namespace tvm {
namespace tir {

template <typename Derived>
class BaseVerifier : public StmtExprVisitor {
 public:
  static bool Verify(const PrimFunc& func, bool assert_mode) {
    Derived verifier(func, assert_mode);
    verifier(func->body);
    return !verifier.has_error_;
  }

 protected:
  explicit BaseVerifier(const PrimFunc&, bool assert_mode) : assert_mode_(assert_mode) {}

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

  /*! \brief Whether it's in assert mode. */
  bool assert_mode_;
  /*! \brief Whether there is error. */
  bool has_error_{false};
};

/*! \brief Verify all Expr inside the block does not contain:
 *    1. loop vars outside the current block.
 *    2. block vars of parent blocks.
 */
class BlockVarAccessVerifier : public BaseVerifier<BlockVarAccessVerifier> {
 private:
  using BaseVerifier::BaseVerifier;

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
  /*! \brief Current nested block stack level. */
  std::vector<const BlockNode*> block_stack_;
};

class UndefinedBufferAccessVerifier : public BaseVerifier<UndefinedBufferAccessVerifier> {
 private:
  using Parent = BaseVerifier<UndefinedBufferAccessVerifier>;
  friend class BaseVerifier<UndefinedBufferAccessVerifier>;

  UndefinedBufferAccessVerifier(const PrimFunc& func, bool assert_mode)
      : BaseVerifier(func, assert_mode) {
    for (const Var& param : func->params) {
      if (auto opt = func->buffer_map.Get(param)) {
        global_defs_.emplace_back(this, opt.value(), NullOpt);
      }
    }
  }

  // Buffer definition sites
  void VisitStmt_(const BufferRealizeNode* op) final {
    Context context(this, op->buffer, GetRef<Stmt>(op));
    Parent::VisitStmt_(op);
  }
  void VisitStmt_(const DeclBufferNode* op) final {
    Context context(this, op->buffer, GetRef<Stmt>(op));
    Parent::VisitStmt_(op);
  }
  void VisitStmt_(const BlockNode* op) final {
    std::vector<Context> context;
    for (const auto& buf : op->alloc_buffers) {
      context.emplace_back(this, buf, GetRef<Stmt>(op));
    }
    for (const auto& match : op->match_buffers) {
      context.emplace_back(this, match->buffer, GetRef<Stmt>(op));
    }
    Parent::VisitStmt_(op);
  }

  // Buffer usage sites
  void VisitExpr_(const BufferLoadNode* op) final {
    AssertDefined(op->buffer, op);
    Parent::VisitExpr_(op);
  }
  void VisitStmt_(const BufferStoreNode* op) final {
    AssertDefined(op->buffer, op);
    Parent::VisitStmt_(op);
  }

  // AttrStmt, which may be either usage or definition, depending on
  // the attribute.
  void VisitStmt_(const AttrStmtNode* op) final {
    std::vector<Context> context;

    if (op->attr_key == attr::buffer_bind_scope) {
      Array<ObjectRef> arr = Downcast<Array<ObjectRef>>(op->node);
      ICHECK_EQ(arr.size(), 2U);
      Buffer source = Downcast<Buffer>(arr[0]);
      Buffer target = Downcast<Buffer>(arr[1]);
      AssertDefined(target, op);
      context.emplace_back(this, source, GetRef<Stmt>(op));
    } else if (auto node = op->node.as<Buffer>()) {
      AssertDefined(node.value(), op);
    }
    Parent::VisitStmt_(op);
  }

  // A context manager for scoped buffer definitions
  struct Context {
    Context(UndefinedBufferAccessVerifier* self, const Buffer& buffer, Optional<Stmt> definition)
        : self_(self), buffer_(buffer) {
      if (auto it = self_->definition_site_.find(buffer_); it != self_->definition_site_.end()) {
        Optional<Stmt> prev = (*it).second;
        self_->has_error_ = true;
        if (self_->assert_mode_) {
          auto& fatal = LOG(FATAL);
          fatal << "Buffer " << buffer << " was defined multiple times.  "
                << "The first definition occurred ";
          if (prev) {
            fatal << " in the " << prev->GetTypeKey() << ", " << prev << ".  ";
          } else {
            fatal << " in the PrimFunc's buffer_map.  ";
          }
          fatal << "The second definition occurred ";
          if (definition) {
            fatal << " in the " << definition->GetTypeKey() << ", " << definition << ".";
          } else {
            fatal << " in the PrimFunc's buffer_map.";
          }
        }
      }
      self_->definition_site_.Set(buffer_, definition);
    }
    ~Context() {
      if (self_) {
        self_->definition_site_.erase(buffer_);
      }
    }
    Context& operator=(const Context&) = delete;
    Context(const Context&) = delete;
    Context& operator=(Context&& other) {
      swap(std::move(other));
      return *this;
    }
    Context(Context&& other) { swap(std::move(other)); }

    void swap(Context&& other) {
      std::swap(self_, other.self_);
      std::swap(buffer_, other.buffer_);
    }

    UndefinedBufferAccessVerifier* self_{nullptr};
    Buffer buffer_;
  };

  void AssertDefined(const Buffer& buffer, const Object* usage) {
    auto it = definition_site_.find(buffer);
    if (it == definition_site_.end()) {
      has_error_ = true;
      if (assert_mode_) {
        Array<Buffer> defined_bufs;
        for (const auto& [buf, definition] : definition_site_) {
          defined_bufs.push_back(buf);
        }
        LOG(FATAL) << "Buffer " << buffer << "@" << buffer.get() << " was accessed as part of "
                   << GetRef<ObjectRef>(usage) << ", without a definition.  "
                   << "At this location, buffers " << defined_bufs << " are defined";
      }
    }
  }

  // A lookup table for currently-defined buffers.  The
  // `Optional<Stmt>` contains either the location at which the buffer
  // is defined, or NullOpt if the buffer was defined in the
  // `buffer_map`.
  Map<Buffer, Optional<Stmt>> definition_site_;

  // A container for buffer definitions in the `buffer_map`.
  std::vector<Context> global_defs_;
};

bool VerifyWellFormed(const PrimFunc& func, bool assert_mode) {
  if (!BlockVarAccessVerifier::Verify(func, assert_mode)) {
    return false;
  }
  if (!UndefinedBufferAccessVerifier::Verify(func, assert_mode)) {
    return false;
  }
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
