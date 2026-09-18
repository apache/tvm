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
 * \file ir_utils.cc
 * \brief Helper functions to construct and compose IR nodes.
 */
#include "ir_utils.h"

#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/structural_visit.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/scope_stack.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/layout.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace tvm {
namespace tirx {
using namespace tvm::prim;

Stmt MergeNest(const std::vector<Stmt>& nest, Stmt body) {
  // use reverse iteration
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    Stmt s = *ri;
    if (const auto* for_ = s.as<ForNode>()) {
      auto n = ffi::make_object<ForNode>(*for_);
      TVM_FFI_ICHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* bind = s.as<BindNode>()) {
      // Bind has no body -- prepend it before the accumulated body in a SeqStmt.
      body = SeqStmt::Flatten(ffi::GetRef<Stmt>(bind), body);
    } else if (const auto* attr = s.as<AttrStmtNode>()) {
      auto n = ffi::make_object<AttrStmtNode>(*attr);
      TVM_FFI_ICHECK(is_no_op(n->body));
      n->body = body;
      body = Stmt(n);
    } else if (const auto* ite = s.as<IfThenElseNode>()) {
      auto n = ffi::make_object<IfThenElseNode>(*ite);
      TVM_FFI_ICHECK(is_no_op(n->then_case));
      TVM_FFI_ICHECK(!n->else_case);
      n->then_case = body;
      body = Stmt(n);
    } else if (const auto* seq = s.as<SeqStmtNode>()) {
      auto n = ffi::make_object<SeqStmtNode>(*seq);
      TVM_FFI_ICHECK(n->size() != 0 && is_no_op(n->seq[n->size() - 1]));
      n->seq.Set(n->size() - 1, body);
      body = Stmt(n);
    } else if (s.as<AssertStmtNode>()) {
      body = SeqStmt({s, body});
    } else if (s.as<AllocBufferNode>() || s.as<DeclBufferNode>()) {
      body = SeqStmt::Flatten(s, body);
    } else {
      TVM_FFI_THROW(InternalError) << "not supported nest type";
    }
  }
  return body;
}

Stmt MergeNest(const std::vector<std::vector<Stmt>>& nest, Stmt body) {
  for (auto ri = nest.rbegin(); ri != nest.rend(); ++ri) {
    body = MergeNest(*ri, body);
  }
  return body;
}

PrimFunc IRConvertSSA::VisitPrimFunc(PrimFunc func) {
  // Remap parameters, if they were used in another function.
  // Function-scope remaps use function_scope_var_remap_ (not the scope stack),
  // because they persist across the entire function body.
  auto params = func->params.Map([&](const tirx::Var& var) -> tirx::Var {
    if (defined_.count(var.get())) {
      Var new_var = MakeNewVar(var);
      PushVarRemap(var, new_var);
      return new_var;
    } else {
      defined_.insert(var.get());
      return var;
    }
  });

  // Remap implicitly defined buffer parameters
  {
    std::unordered_set<const VarNode*> defined_params;
    for (const auto& var : func->params) {
      defined_params.insert(var.get());
    }
    std::unordered_set<const VarNode*> defined_match_vars;
    for (const Var& param : func->params) {
      auto buffer = param.as<BufferVar>();
      if (!buffer) continue;
      auto check_var = [&](const Var& var) {
        const VarNode* var_ptr = var.get();
        if (defined_params.count(var_ptr)) return;
        if (!defined_match_vars.insert(var_ptr).second) return;

        // Buffer-parameter shape vars use "match" semantics: first occurrence
        // defines the var, subsequent occurrences (in other buffers) are
        // just consistent uses of the same var -- not redefinitions.
        if (defined_.count(var_ptr)) {
          Var new_var = MakeNewVar(var);
          PushVarRemap(var, new_var);
        } else {
          defined_.insert(var_ptr);
        }
      };
      auto walk_fn = [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
        check_var(var);
        return ffi::WalkResult::Advance();
      };
      for (const auto& dim : buffer.value()->shape) {
        ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(dim, walk_fn);
      }
      for (const auto& stride : buffer.value()->strides) {
        if (auto var = stride.as<Var>()) check_var(var.value());
      }
      if (auto var = buffer.value()->elem_offset.as<Var>()) check_var(var.value());
    }
  }

  // Update the buffer parameters, based on the redefined parameters
  bool buffer_params_changed = false;
  for (size_t i = 0; i < func->params.size(); ++i) {
    if (auto buffer = func->params[i].as<BufferVar>()) {
      BufferVar new_buffer = GetRemappedBuffer(buffer.value());
      if (!new_buffer.same_as(buffer.value()) || !params[i].same_as(new_buffer)) {
        buffer_params_changed = true;
        params.Set(i, new_buffer.var());
      }
    }
  }

  auto attrs = [&]() -> DictAttrs {
    ffi::Map<ffi::String, ffi::Any> dict;
    bool made_change = false;

    for (const auto& [key, old_value] : func->attrs->dict) {
      auto value = old_value;
      if (auto expr = value.as<PrimExpr>()) {
        value = Mutate(expr.value(), InplaceMode::kDisallow).ValueOrUnchanged(expr.value());
      } else if (auto* stmt = value.as<StmtNode>()) {
        value = Mutate(ffi::GetRef<Stmt>(stmt), InplaceMode::kDisallow)
                    .ValueOrUnchanged(ffi::GetRef<Stmt>(stmt));
      }

      made_change = made_change || !value.same_as(old_value);
      dict.Set(key, value);
    }

    if (made_change) {
      return DictAttrs(dict);
    } else {
      return func->attrs;
    }
  }();

  auto body_result = Mutate(func->body, InplaceMode::kDisallow);
  bool body_unchanged = body_result.UnchangedOrSameAs(func->body);
  auto body = std::move(body_result).ValueOrUnchanged(func->body);

  // If anything changed, update the returned function
  if (!params.same_as(func->params) || buffer_params_changed || !attrs.same_as(func->attrs) ||
      !body_unchanged) {
    func = PrimFunc(params, body, func->ret_type, attrs);
  }

  // Pop function-scope remaps in reverse order
  PopAllRemapsInCurrentScope();
  function_scope_var_remap_.clear();
  return func;
}

UnchangedOr<Expr> IRConvertSSA::Mutate_(const VarNode* op, InplaceMode inplace_mode) {
  Var var = ffi::GetRef<Var>(op);
  Var mapped = GetRemappedVar(var);
  if (!mapped.same_as(var)) return mapped;
  return StmtExprMutator::Mutate_(op, inplace_mode);
}

UnchangedOr<PrimExpr> IRConvertSSA::Mutate_(const prim::LetNode* op, InplaceMode inplace_mode) {
  const Var& v = op->var;
  if (defined_.count(v.get())) {
    PrimExpr value = this->Mutate(op->value, inplace_mode).ValueOrUnchanged(op->value);
    Var new_var = MakeNewVar(v);
    PushVarRemap(v, new_var);
    PrimExpr body = this->Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body);
    PopVarRemap(v, new_var);
    return prim::Let(new_var, value, body);
  } else {
    defined_.insert(v.get());
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }
}

UnchangedOr<PrimExpr> IRConvertSSA::Mutate_(const TensorLoadNode* op, InplaceMode inplace_mode) {
  auto node = StmtExprMutator::Mutate_(op, inplace_mode)
                  .ValueOrUnchanged(ffi::GetRef<PrimExpr>(op))
                  .as_or_throw<TensorLoad>();
  auto output = VisitBufferAccess(std::move(node));
  return output;
}

UnchangedOr<Stmt> IRConvertSSA::Mutate_(const BufferStoreNode* op, InplaceMode inplace_mode) {
  auto node = StmtExprMutator::Mutate_(op, inplace_mode)
                  .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                  .as_or_throw<BufferStore>();
  auto output = VisitBufferAccess(std::move(node));
  return output;
}

UnchangedOr<Stmt> IRConvertSSA::Mutate_(const DeclBufferNode* op, InplaceMode inplace_mode) {
  Var v = op->buffer.var();
  if (defined_.count(v.get())) {
    Var new_var = MakeNewVar(v);
    PushVarRemap(v, new_var);
  } else {
    defined_.insert(v.get());
  }
  DeclBuffer decl = StmtExprMutator::Mutate_(op, inplace_mode)
                        .ValueOrUnchanged(ffi::GetRef<Stmt>(op))
                        .as_or_throw<DeclBuffer>();
  BufferVar new_buffer = GetRemappedBuffer(decl->buffer);
  if (!new_buffer.same_as(decl->buffer)) {
    decl.CopyOnWrite()->buffer = std::move(new_buffer);
  }
  return decl;
}

Stmt IRConvertSSA::WithScope(const std::function<Stmt()>& body) {
  return scope_.WithNewScope(body);
}

Var IRConvertSSA::DefineVar(Var var) {
  if (defined_.count(var.get())) {
    Var new_var = MakeNewVar(var);
    PushVarRemap(var, new_var);
    return new_var;
  }
  defined_.insert(var.get());
  return var;
}

BufferStore IRConvertSSA::VisitBufferAccess(BufferStore node) {
  BufferVar new_buf = GetRemappedBuffer(node->buffer);
  if (!new_buf.same_as(node->buffer)) {
    auto writer = node.CopyOnWrite();
    writer->buffer = new_buf;
  }

  return node;
}

TensorLoad IRConvertSSA::VisitBufferAccess(TensorLoad node) {
  BufferVar buffer = node->source.as_or_throw<BufferVar>();
  BufferVar new_buf = GetRemappedBuffer(buffer);
  if (new_buf.same_as(buffer)) {
    return node;
  }
  return BufferLoad(new_buf, node->indices, node->span);
}

Var IRConvertSSA::GetRemappedVar(Var var) {
  if (auto it = scoped_var_remap_.find(var.get());
      it != scoped_var_remap_.end() && it->second.size()) {
    return it->second.back();
  } else if (auto it = function_scope_var_remap_.find(var.get());
             it != function_scope_var_remap_.end()) {
    return it->second;
  } else {
    return var;
  }
}

BufferVar IRConvertSSA::GetRemappedBuffer(BufferVar buf) {
  // Determine the buffer var that should be in the updated buffer,
  // given the current scope.  If no redefines are present, then the
  // buffer var is unchanged.
  Var new_buffer_var = GetRemappedVar(buf.var());
  PrimExpr elem_offset =
      Mutate(buf->elem_offset, InplaceMode::kDisallow).ValueOrUnchanged(buf->elem_offset);
  auto visit_expr = [this](const PrimExpr& expr) {
    return Mutate(expr, InplaceMode::kDisallow).ValueOrUnchanged(expr);
  };
  ffi::Array<PrimExpr> shape = buf->shape.Map(visit_expr);
  ffi::Array<PrimExpr> strides = buf->strides.Map(visit_expr);

  // Rewrite the layout's per-iter extent/stride expressions in lockstep
  // with the shape. If we don't, SSA-renamed shape vars end up as fresh
  // Vars while the layout still references the original, producing
  // structurally-unequal buffers whose shape and layout disagree (e.g.,
  // test_dynamic_launch_thread).
  ffi::Optional<Layout> new_layout = buf->layout;
  bool layout_changed = false;
  if (buf->layout.has_value()) {
    if (auto opt_tile = buf->layout.value().as<TileLayoutNode>()) {
      auto remap_iter = [&](const Iter& it) -> Iter {
        PrimExpr new_extent =
            Mutate(it->extent, InplaceMode::kDisallow).ValueOrUnchanged(it->extent);
        PrimExpr new_stride =
            Mutate(it->stride, InplaceMode::kDisallow).ValueOrUnchanged(it->stride);
        if (new_extent.same_as(it->extent) && new_stride.same_as(it->stride)) {
          return it;
        }
        return Iter(new_extent, new_stride, it->axis);
      };
      auto new_shard = opt_tile->shard.Map(remap_iter);
      auto new_replica = opt_tile->replica.Map(remap_iter);
      if (!new_shard.same_as(opt_tile->shard) || !new_replica.same_as(opt_tile->replica)) {
        new_layout = TileLayout(new_shard, new_replica, opt_tile->offset);
        layout_changed = true;
      }
    }
  }

  // If no mapping is required, return the original buffer.
  if (new_buffer_var.same_as(buf.var()) && elem_offset.same_as(buf->elem_offset) &&
      shape.same_as(buf->shape) && strides.same_as(buf->strides) && !layout_changed) {
    return buf;
  }

  // If the current scope already has a mapping of this buffer, use
  // the mapped buffer.
  auto key = buf.get();
  std::vector<BufferVar>& buffers = buf_remap_[key];
  if (buffers.size() && buffers.back().same_as(new_buffer_var)) {
    return buffers.back();
  }

  // When only the buffer's identity changed, the remapped Var already has
  // the desired BufferType.  Reuse that exact Var so the definition and all
  // subsequent uses remain in SSA.
  if (const auto* type = new_buffer_var->ty.as<BufferTypeNode>()) {
    BufferVar candidate(new_buffer_var);
    if (shape.same_as(type->shape) && strides.same_as(type->strides) &&
        elem_offset.same_as(type->elem_offset) && !layout_changed) {
      buffers.push_back(candidate);
      return candidate;
    }
  }

  // Otherwise, make and return a new buffer object that uses the
  // new buffer, pushing it onto the scoped stack of existing
  // buffers.  This will be popped when the new_buffer_var
  // redefinition is popped.
  auto type = CopyBufferType(buf);
  type->shape = shape;
  type->strides = strides;
  type->elem_offset = elem_offset;
  if (layout_changed) {
    type->layout = std::move(new_layout);
  }
  BufferVar new_buf = RebuildBufferVar(buf, std::move(type), new_buffer_var->name);

  // A BufferVar's metadata lives in its Var type.  If rewriting the
  // metadata required a fresh Var, make it the active remap as well.  This
  // keeps BufferLoad/BufferStore and ordinary Var uses (such as
  // buffer_data) on the same identity.
  auto it = scoped_var_remap_.find(buf.get());
  if (it != scoped_var_remap_.end() && it->second.size() &&
      it->second.back().same_as(new_buffer_var)) {
    it->second.back() = new_buf.var();
  } else if (auto function_it = function_scope_var_remap_.find(buf.get());
             function_it != function_scope_var_remap_.end() &&
             function_it->second.same_as(new_buffer_var)) {
    function_it->second = new_buf.var();
  } else {
    PushVarRemap(buf.var(), new_buf.var());
  }
  buffers.push_back(new_buf);
  return new_buf;
}

UnchangedOr<Stmt> IRConvertSSA::Mutate_(const BindNode* op, InplaceMode inplace_mode) {
  // Bind var remaps are tracked in the current scope so they persist
  // across SeqStmt siblings and are cleaned up when the enclosing
  // body-carrying statement's scope exits.
  const Var& v = op->var;
  if (defined_.count(v.get())) {
    Expr value = this->Mutate(op->value, inplace_mode).ValueOrUnchanged(op->value);
    Var new_var = MakeNewVar(v);
    PushVarRemap(v, new_var);
    return Bind(new_var, value);
  } else {
    defined_.insert(v.get());
    return StmtExprMutator::Mutate_(op, inplace_mode);
  }
}

UnchangedOr<Stmt> IRConvertSSA::Mutate_(const IfThenElseNode* op, InplaceMode inplace_mode) {
  // Each branch gets its own scope so Bind remaps in one branch
  // do not leak into the other.
  auto condition_result = Mutate(op->condition, inplace_mode);
  bool condition_unchanged = condition_result.UnchangedOrSameAs(op->condition);
  PrimExpr condition = std::move(condition_result).ValueOrUnchanged(op->condition);
  Stmt then_case = scope_.WithNewScope([&]() -> Stmt {
    return Mutate(op->then_case, inplace_mode).ValueOrUnchanged(op->then_case);
  });
  ffi::Optional<Stmt> else_case;
  if (op->else_case) {
    else_case = scope_.WithNewScope([&]() -> Stmt {
      return Mutate(op->else_case.value(), inplace_mode).ValueOrUnchanged(op->else_case.value());
    });
  }
  if (condition_unchanged && then_case.same_as(op->then_case) && else_case.same_as(op->else_case)) {
    return ffi::Unchanged();
  }
  return IfThenElse(condition, then_case, else_case);
}

UnchangedOr<Stmt> IRConvertSSA::Mutate_(const ForNode* op, InplaceMode inplace_mode) {
  const Var& v = op->loop_var;
  if (defined_.count(v.get())) {
    return scope_.WithNewScope([&]() -> Stmt {
      Var new_var = MakeNewVar(v);
      PushVarRemap(v, new_var);
      Stmt stmt =
          StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
      auto n = ffi::make_object<ForNode>(*stmt.as<ForNode>());
      n->loop_var = new_var.as_or_throw<PrimVar>();
      return For(n);
    });
  } else {
    defined_.insert(v.get());
    return scope_.WithNewScope([&]() -> Stmt {
      return StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    });
  }
}

UnchangedOr<Stmt> IRConvertSSA::Mutate_(const WhileNode* op, InplaceMode inplace_mode) {
  return scope_.WithNewScope([&]() -> Stmt {
    return StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
  });
}

UnchangedOr<Stmt> IRConvertSSA::Mutate_(const AllocBufferNode* op, InplaceMode inplace_mode) {
  Var v = op->buffer.var();
  if (defined_.count(v.get())) {
    Var new_var = MakeNewVar(v);
    PushVarRemap(v, new_var);
  } else {
    defined_.insert(v.get());
  }
  Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
  op = stmt.as<AllocBufferNode>();
  // Use GetRemappedBuffer so that the AllocBuffer's buffer is the same
  // object as the one used by BufferStore/TensorLoad in subsequent siblings.
  BufferVar new_buf = GetRemappedBuffer(op->buffer);
  if (!new_buf.same_as(op->buffer)) {
    auto node = stmt.as_or_throw<AllocBuffer>();
    node.CopyOnWrite()->buffer = std::move(new_buf);
    return node;
  }
  return stmt;
}

UnchangedOr<Stmt> IRConvertSSA::Mutate_(const AttrStmtNode* op, InplaceMode inplace_mode) {
  if (const IterVarNode* iter_var = op->node.as<IterVarNode>()) {
    Range dom = iter_var->dom;
    if (dom.defined()) {
      // Retain the original domain while comparing and rebuilding its replacement.
      auto min = Mutate(dom->min, InplaceMode::kDisallow).ValueOrUnchanged(dom->min);
      auto extent = Mutate(dom->extent, InplaceMode::kDisallow).ValueOrUnchanged(dom->extent);
      if (!min.same_as(iter_var->dom->min) || !extent.same_as(iter_var->dom->extent)) {
        dom = Range::FromMinExtent(min, extent);
      }
    }

    Var var = iter_var->var;
    bool delayed_define = false;
    if (auto it = function_scope_var_remap_.find(var.get());
        it != function_scope_var_remap_.end()) {
      var = it->second;
    } else if (defined_.count(var.get())) {
      Var new_var(var->name, var->ty);

      function_scope_var_remap_.insert({var.get(), new_var});
      var = new_var;
    } else {
      // The AttrStmt refers to an undefined variable.  This is
      // allowed for some attributes, such as
      // "pragma_parallel_launch_point", which annotates a variable
      // that is about to occur in a ForNode.  In these cases, the
      // ForNode and the AttrStmt must continue using the same
      // variable defintion.
      //
      // However, other AttrStmt, such as "thread_extent", act as
      // points of definition for the variable they annotate.  If
      // the variable has not been defined after visiting the body,
      // we should mark it as defined before exiting.  This ensures
      // correct de-duplication between multiple functions.
      //
      // This implementation may be simplified in the future by
      // moving "pragma_parallel_launch_point" to be an annotation
      // on the `ForNode`, rather than an `AttrStmt`.
      delayed_define = true;
    }

    IterVar new_iter_var;
    if (dom.same_as(iter_var->dom) && var.same_as(iter_var->var)) {
      new_iter_var = ffi::GetRef<IterVar>(iter_var);
    } else {
      new_iter_var = IterVar(dom, var.as_or_throw<PrimVar>(), iter_var->iter_type,
                             iter_var->thread_tag, iter_var->span);
    }
    auto value_result = Mutate(op->value, inplace_mode);
    bool value_unchanged = value_result.UnchangedOrSameAs(op->value);
    auto value = std::move(value_result).ValueOrUnchanged(op->value);
    auto body = scope_.WithNewScope(
        [&]() -> Stmt { return Mutate(op->body, inplace_mode).ValueOrUnchanged(op->body); });

    Stmt output;
    if (new_iter_var.get() == iter_var && body.same_as(op->body) && value_unchanged) {
      output = ffi::GetRef<Stmt>(op);
    } else {
      output = AttrStmt(new_iter_var, op->attr_key, value, body, iter_var->span);
    }

    if (delayed_define) {
      if (!defined_.count(var.get())) {
        function_scope_var_remap_.insert({var.get(), var});
        defined_.insert(var.get());
      }
    }

    return output;

  } else if (const VarNode* v = op->node.as<VarNode>()) {
    Stmt stmt = scope_.WithNewScope([&]() -> Stmt {
      return StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    });
    op = stmt.as<AttrStmtNode>();
    if (scoped_var_remap_.count(v) && scoped_var_remap_[v].size() != 0) {
      return AttrStmt(scoped_var_remap_[v].back(), op->attr_key, op->value, op->body);
    } else {
      return stmt;
    }
  } else {
    return scope_.WithNewScope([&]() -> Stmt {
      return StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
    });
  }
}

bool IRConvertSSA::BufferDependsOnVar(const BufferVar& buffer, const VarNode* var) {
  if (buffer.get() == var) return true;

  auto uses_var = [var](const PrimExpr& expr) {
    auto walkfn = [var](const Var& candidate) -> ffi::Expected<ffi::WalkResult> {
      return candidate.get() == var ? ffi::WalkResult::Interrupt(ffi::VisitInterrupt(candidate))
                                    : ffi::WalkResult::Advance();
    };
    return expr.defined() &&
           ffi::StructuralWalk<ffi::WalkOrder::kPreOrder>(expr, walkfn).has_value();
  };
  if (uses_var(buffer->elem_offset)) return true;
  for (const PrimExpr& dim : buffer->shape) {
    if (uses_var(dim)) return true;
  }
  for (const PrimExpr& stride : buffer->strides) {
    if (uses_var(stride)) return true;
  }
  if (buffer->layout.has_value()) {
    if (const auto* tile_layout = buffer->layout.value().as<TileLayoutNode>()) {
      for (const Iter& iter : tile_layout->shard) {
        if (uses_var(iter->extent) || uses_var(iter->stride)) return true;
      }
      for (const Iter& iter : tile_layout->replica) {
        if (uses_var(iter->extent) || uses_var(iter->stride)) return true;
      }
    }
  }
  return false;
}

Var IRConvertSSA::MakeNewVar(const Var& old_var) { return Var(old_var->name, old_var->ty); }

void IRConvertSSA::PushVarRemap(const Var& old_var, const Var& new_var) {
  scoped_var_remap_[old_var.get()].push_back(new_var);
  auto& level = scope_.Current();
  level.parent = this;
  level.push_back({old_var, new_var});
}

void IRConvertSSA::PopVarRemap(const Var& old_var, const Var& new_var) {
  scoped_var_remap_[old_var.get()].pop_back();
  for (auto& kv : buf_remap_) {
    std::vector<BufferVar>& buffers = kv.second;
    if (buffers.size() && BufferDependsOnVar(buffers.back(), new_var.get())) {
      buffers.pop_back();
    }
  }
  // Also remove from the current scope's tracking vector
  auto& current = scope_.Current();
  if (current.size() && current.back().new_var.same_as(new_var)) {
    current.pop_back();
  }
}

void IRConvertSSA::PopAllRemapsInCurrentScope() {
  auto& current = scope_.Current();
  while (current.size()) {
    auto& remap = current.back();
    scoped_var_remap_[remap.old_var.get()].pop_back();
    for (auto& kv : buf_remap_) {
      std::vector<BufferVar>& buffers = kv.second;
      if (buffers.size() && BufferDependsOnVar(buffers.back(), remap.new_var.get())) {
        buffers.pop_back();
      }
    }
    current.pop_back();
  }
}

Stmt ConvertSSA(Stmt stmt) {
  return ffi::make_object<IRConvertSSA>()->Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(stmt);
}

ffi::String GetPtrStorageScope(Var buffer_var) {
  if (const auto* buffer_type = buffer_var->ty.as<BufferTypeNode>()) {
    return buffer_type->storage_scope;
  }
  const auto* ptr_type = buffer_var->ty.as<PointerTypeNode>();
  TVM_FFI_ICHECK(ptr_type)
      << "The provided variable is neither a pointer nor a buffer-typed variable";
  return ptr_type->storage_scope;
}

ffi::Array<PrimExpr> GetBufferAllocationShape(const BufferVar& buffer) {
  ffi::Array<PrimExpr> alloc_shape = buffer->shape;
  if (buffer->strides.size()) {
    TVM_FFI_ICHECK_EQ(buffer->shape.size(), buffer->strides.size());
    for (size_t i = buffer->strides.size() - 1; i > 0; --i) {
      TVM_FFI_ICHECK(
          sym::Analyzer()->CanProveEqual(floormod(buffer->strides[i - 1], buffer->strides[i]), 0));
      alloc_shape.Set(i, buffer->strides[i - 1] / buffer->strides[i]);
    }
  }
  return alloc_shape;
}

// Attribute strings are the metadata protocol shared by lowered and schedulable statements.
std::pair<PrimExpr, PrimExpr> GetAsyncWaitAttributes(const AttrStmtNode* op) {
  TVM_FFI_ICHECK(op && op->attr_key == tvm::tirx::attr::async_wait_queue_scope);
  auto inner = op->body.as<AttrStmtNode>();
  TVM_FFI_ICHECK(inner && inner->attr_key == tvm::tirx::attr::async_wait_inflight_count);
  return std::make_pair(op->value, inner->value);
}

int Stoi(const std::string& str) {
  try {
    return std::stoi(str);
  } catch (std::invalid_argument& e) {
    TVM_FFI_THROW(InternalError) << "Cannot convert \"" << str << "\" to int";
    throw;
  }
}

std::pair<int32_t, int32_t> GetWmmaFragmentDimSize(const std::string& shape_str,
                                                   const std::string& scope) {
  size_t m, n, k;
  size_t last_pos = 0, pos = 0;
  pos = shape_str.find(", ", last_pos);
  m = Stoi(shape_str.substr(last_pos, pos - last_pos));
  last_pos = pos + 2;
  pos = shape_str.find(", ", last_pos);
  n = Stoi(shape_str.substr(last_pos, pos - last_pos));
  last_pos = pos + 2;
  k = Stoi(shape_str.substr(last_pos, shape_str.length() - last_pos));
  if (scope == "wmma.matrix_a") {
    return std::pair<int32_t, int32_t>(m, k);
  } else if (scope == "wmma.matrix_b") {
    return std::pair<int32_t, int32_t>(k, n);
  } else if (scope == "wmma.accumulator") {
    return std::pair<int32_t, int32_t>(m, n);
  }
  return std::pair<int32_t, int32_t>(0, 0);
}

std::optional<bool> IsHostFunc(const PrimFunc& func) {
  if (func->HasNonzeroAttr(tvm::tirx::attr::kIsHostFunc)) {
    return true;
  } else if (auto target = func->GetAttr<Target>(tvm::attr::kTarget)) {
    return target.value()->HasKey("cpu");
  } else {
    return std::nullopt;
  }
}

IRModule IRConvertSSA::VisitIRModule(IRModule mod) {
  ffi::Map<GlobalVar, BaseFunc> functions;
  bool made_change = false;
  for (auto [gvar, base_func] : mod->functions) {
    if (auto* ptr = base_func.as<tirx::PrimFuncNode>()) {
      auto updated = VisitPrimFunc(ffi::GetRef<tirx::PrimFunc>(ptr));
      if (!updated.same_as(base_func)) {
        made_change = true;
        base_func = updated;
      }
    }
    functions.Set(gvar, base_func);
  }
  if (made_change) {
    mod.CopyOnWrite()->functions = std::move(functions);
  }
  return mod;
}

namespace transform {
Pass ConvertSSA() {
  auto pass_func = [](IRModule mod, PassContext ctx) {
    return ffi::make_object<tirx::IRConvertSSA>()->VisitIRModule(std::move(mod));
  };
  return tvm::transform::CreateModulePass(pass_func, 0, "tirx.ConvertSSA", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.transform.ConvertSSA", ConvertSSA);
}

}  // namespace transform
}  // namespace tirx
}  // namespace tvm
