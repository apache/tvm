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
  std::unordered_set<const VarNode*> parameter_symbols;
  // Define explicit parameters before the symbolic values in their types.
  for (const Var& param : func->params) {
    DefineVar(param);
    parameter_symbols.insert(param.get());
  }
  for (const Var& param : func->params) {
    ffi::StructuralWalk<ffi::WalkOrder::kPostOrder>(
        param->ty, [&](const Var& var) -> ffi::Expected<ffi::WalkResult> {
          if (!parameter_symbols.count(var.get())) {
            parameter_symbols.insert(var.get());
            DefineVar(var);
          }
          return ffi::WalkResult::Advance();
        });
  }
  auto params = func->params.Map([&](const Var& var) {
    Var mapped = GetRemappedVar(var);
    Type type = Mutate(var->ty, InplaceMode::kDisallow)
                    .as_or_throw<UnchangedOr<Type>>()
                    .ValueOrUnchanged(var->ty);
    if (!type.same_as(mapped->ty)) {
      mapped = mapped.CopyWithType(type);
      PushVarRemap(var, mapped);
    }
    return mapped;
  });

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
  if (!params.same_as(func->params) || !attrs.same_as(func->attrs) || !body_unchanged) {
    func = PrimFunc(params, body, func->ret_type, attrs);
  }

  // Pop function-scope remaps in reverse order
  PopAllRemapsInCurrentScope();
  function_scope_var_remap_.clear();
  return func;
}

UnchangedOr<Expr> IRConvertSSA::Mutate_(const VarNode* op, InplaceMode inplace_mode) {
  Var var = ffi::GetRef<Var>(op);
  if (def_region_kind() != kTVMFFIDefRegionKindNone) {
    if (def_region_kind() == kTVMFFIDefRegionKindPattern) {
      Var mapped = GetRemappedVar(var);
      if (!mapped.same_as(var) || defined_.count(var.get())) return mapped;
    }
    return DefineVar(var);
  }
  Var mapped = GetRemappedVar(var);
  if (!mapped.same_as(var)) return mapped;
  return ffi::Unchanged();
}

UnchangedOr<PrimExpr> IRConvertSSA::Mutate_(const prim::LetNode* op, InplaceMode inplace_mode) {
  ffi::Any previous_remap = VarRemapGet(op->var);
  PrimExpr result = scope_.WithNewScope([&] {
    return StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<PrimExpr>(op));
  });
  VarRemapSet(op->var, previous_remap);
  return result;
}

Stmt IRConvertSSA::WithScope(const std::function<Stmt()>& body) {
  return scope_.WithNewScope(body);
}

Var IRConvertSSA::DefineVar(Var var) {
  Type type = WithDefRegionKind(kTVMFFIDefRegionKindNone, [&] {
    return Mutate(var->ty, InplaceMode::kDisallow)
        .as_or_throw<UnchangedOr<Type>>()
        .ValueOrUnchanged(var->ty);
  });
  Var result = defined_.count(var.get()) ? MakeNewVar(var) : var;
  defined_.insert(var.get());
  if (!type.same_as(result->ty)) result = result.CopyWithType(type);
  if (!result.same_as(var)) PushVarRemap(var, result);
  return result;
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

UnchangedOr<Stmt> IRConvertSSA::Mutate_(const BindNode* op, InplaceMode inplace_mode) {
  Var var = op->var;
  ffi::Any previous_remap = VarRemapGet(var);
  // The ordinary Bind path rewrites the RHS before defining the Var and propagates its type.
  Stmt stmt = StmtExprMutator::Mutate_(op, inplace_mode).ValueOrUnchanged(ffi::GetRef<Stmt>(op));
  Var bound_var = stmt.as<BindNode>()->var;
  if (!bound_var.same_as(GetRemappedVar(var))) PushVarRemap(var, bound_var);
  VarRemapSet(var, previous_remap);
  return stmt;
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

UnchangedOr<Stmt> IRConvertSSA::Mutate_(const RegionStmtNode* op, InplaceMode inplace_mode) {
  auto attrs = Mutate(op->attrs, InplaceMode::kDisallow)
                   .as_or_throw<UnchangedOr<DictAttrs>>()
                   .ValueOrUnchanged(op->attrs);
  auto args = Mutate(op->args, InplaceMode::kDisallow)
                  .as_or_throw<UnchangedOr<ffi::Array<Expr>>>()
                  .ValueOrUnchanged(op->args);
  ffi::Array<Var> params;
  Stmt body = scope_.WithNewScope([&]() -> Stmt {
    for (const Var& var : op->body_params) params.push_back(DefineVar(var));
    return Mutate(op->body, InplaceMode::kDisallow).ValueOrUnchanged(op->body);
  });
  ffi::Array<Var> results;
  for (const Var& var : op->result_vars) results.push_back(DefineVar(var));
  return RegionStmt(op->op, std::move(args), std::move(params), std::move(attrs), std::move(body),
                    std::move(results), op->span);
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
      // Preserve the annotated variable's identity for later definitions
      // and independent functions, without introducing a lexical body binding.
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

    Stmt output = ffi::GetRef<Stmt>(op);
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

Var IRConvertSSA::MakeNewVar(const Var& old_var) {
  return Var(old_var->name, old_var->ty, old_var->span);
}

void IRConvertSSA::PushVarRemap(const Var& old_var, const Var& new_var) {
  scoped_var_remap_[old_var.get()].push_back(new_var);
  auto& level = scope_.Current();
  level.parent = this;
  level.push_back({old_var, new_var});
}

void IRConvertSSA::PopAllRemapsInCurrentScope() {
  auto& current = scope_.Current();
  while (current.size()) {
    auto& remap = current.back();
    scoped_var_remap_[remap.old_var.get()].pop_back();
    current.pop_back();
  }
}

Stmt ConvertSSA(Stmt stmt) {
  return ffi::make_object<IRConvertSSA>()->Mutate(stmt, InplaceMode::kAllow).ValueOrUnchanged(stmt);
}

ffi::String GetPtrStorageScope(Var buffer_var) {
  if (const auto* buffer_type = buffer_var->ty.as<TensorTypeNode>()) {
    return buffer_type->storage_scope;
  }
  const auto* ptr_type = buffer_var->ty.as<PointerTypeNode>();
  TVM_FFI_ICHECK(ptr_type)
      << "The provided variable is neither a pointer nor a buffer-typed variable";
  return ptr_type->storage_scope;
}

ffi::Array<PrimExpr> GetBufferAllocationShape(const TensorVar& buffer) {
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
