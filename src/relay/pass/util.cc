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
 * Copyright (c) 2018 by Contributors
 *
 * \file util.cc
 *
 * \brief Utility functions for Relay.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include "pass_util.h"
#include "../ir/type_functor.h"

namespace tvm {
namespace relay {

template<typename T>
struct InsertionSet {
  std::unordered_set<T, NodeHash, NodeEqual> set;
  std::vector<T> data;
  void Insert(const T& t) {
    if (set.count(t) == 0) {
      set.insert(t);
      data.push_back(t);
    }
  }
};

class TypeVarTVisitor : public TypeVisitor {
 public:
  TypeVarTVisitor(
      InsertionSet<TypeVar>* type_vars,
      InsertionSet<TypeVar>* bound_type_vars)
    : type_vars_(type_vars), bound_type_vars_(bound_type_vars) { }

  void VisitType_(const TypeVarNode* tp) final {
    TypeVar var = GetRef<TypeVar>(tp);
    type_vars_->Insert(var);
  }

  void VisitType_(const FuncTypeNode* f) final {
    for (auto type_param : f->type_params) {
      type_vars_->Insert(type_param);
      bound_type_vars_->Insert(type_param);
    }
    TypeVisitor::VisitType_(f);
  }

 private:
  InsertionSet<TypeVar>* type_vars_;
  InsertionSet<TypeVar>* bound_type_vars_;
};

class TypeVarEVisitor : private ExprVisitor {
 public:
  explicit TypeVarEVisitor(const Module& mod) : mod_(mod) {}

  Array<TypeVar> CollectFree() {
    Array<TypeVar> ret;
    for (const auto& v : type_vars_.data) {
      if (bound_type_vars_.set.count(v) == 0) {
        ret.push_back(v);
      }
    }
    return ret;
  }

  Array<TypeVar> CollectBound() {
    Array<TypeVar> ret;
    for (const auto& v : bound_type_vars_.data) {
      ret.push_back(v);
    }
    return ret;
  }

  Array<TypeVar> CollectAll() {
    Array<TypeVar> ret;
    for (const auto& v : type_vars_.data) {
      ret.push_back(v);
    }
    return ret;
  }

  Array<TypeVar> Free(const Expr& expr) {
    VisitExpr(expr);
    return CollectFree();
  }

  Array<TypeVar> Free(const Type& type) {
    VisitType(type);
    return CollectFree();
  }

  Array<TypeVar> Bound(const Expr& expr) {
    VisitExpr(expr);
    return CollectBound();
  }

  Array<TypeVar> Bound(const Type& type) {
    VisitType(type);
    return CollectBound();
  }

  Array<TypeVar> All(const Expr& expr) {
    VisitExpr(expr);
    return CollectAll();
  }

  Array<TypeVar> All(const Type& type) {
    VisitType(type);
    return CollectAll();
  }

  void VisitExpr_(const FunctionNode* f) final {
    for (const auto& tp : f->type_params) {
      type_vars_.Insert(tp);
      bound_type_vars_.Insert(tp);
    }
    ExprVisitor::VisitExpr_(f);
  }

  void VisitExpr_(const ConstructorNode* cn) final {
    // for constructors, type vars will be bound in the module
    auto data = mod_->LookupDef(cn->belong_to);
    for (const auto& tv : data->type_vars) {
      type_vars_.Insert(tv);
      bound_type_vars_.Insert(tv);
    }
    ExprVisitor::VisitExpr_(cn);
  }

  void VisitType(const Type& t) final {
    TypeVarTVisitor(&type_vars_, &bound_type_vars_)
        .VisitType(t);
  }

 private:
  InsertionSet<TypeVar> type_vars_;
  InsertionSet<TypeVar> bound_type_vars_;
  const Module& mod_;
};

class VarVisitor : protected ExprVisitor, protected PatternVisitor {
 public:
  Array<Var> Free(const Expr& expr) {
    this->VisitExpr(expr);
    Array<Var> ret;
    for (const auto& v : vars_.data) {
      if (bound_vars_.set.count(v) == 0) {
        ret.push_back(v);
      }
    }
    return ret;
  }

  Array<Var> Collect() {
    Array<Var> ret;
    for (const auto& v : bound_vars_.data) {
      ret.push_back(v);
    }
    return ret;
  }

  Array<Var> Bound(const Expr& expr) {
    this->VisitExpr(expr);
    return Collect();
  }

  Array<Var> Bound(const Pattern& pat) {
    this->VisitPattern(pat);
    return Collect();
  }

  Array<Var> All(const Expr& expr) {
    this->VisitExpr(expr);
    Array<Var> ret;
    for (const auto& v : vars_.data) {
      ret.push_back(v);
    }
    return ret;
  }

  void MarkBounded(const Var& v) {
    bound_vars_.Insert(v);
    vars_.Insert(v);
  }

  void VisitExpr_(const VarNode* var) final {
    vars_.Insert(GetRef<Var>(var));
  }

  void VisitExpr_(const FunctionNode* op) final {
    for (const auto& param : op->params) {
      MarkBounded(param);
    }
    VisitExpr(op->body);
  }

  void VisitExpr_(const LetNode* op) final {
    MarkBounded(op->var);
    VisitExpr(op->value);
    VisitExpr(op->body);
  }

  void VisitPattern(const Pattern& p) final {
    PatternVisitor::VisitPattern(p);
  }

  void VisitPattern_(const PatternVarNode* op) final {
    MarkBounded(op->var);
  }

 private:
  InsertionSet<Var> vars_;
  InsertionSet<Var> bound_vars_;
};

tvm::Array<TypeVar> FreeTypeVars(const Expr& expr, const Module& mod) {
  return TypeVarEVisitor(mod).Free(expr);
}

tvm::Array<TypeVar> FreeTypeVars(const Type& type, const Module& mod) {
  return TypeVarEVisitor(mod).Free(type);
}

tvm::Array<TypeVar> BoundTypeVars(const Expr& expr, const Module& mod) {
  return TypeVarEVisitor(mod).Bound(expr);
}

tvm::Array<TypeVar> BoundTypeVars(const Type& type, const Module& mod) {
  return TypeVarEVisitor(mod).Bound(type);
}

tvm::Array<TypeVar> AllTypeVars(const Expr& expr, const Module& mod) {
  return TypeVarEVisitor(mod).All(expr);
}

tvm::Array<TypeVar> AllTypeVars(const Type& type, const Module& mod) {
  return TypeVarEVisitor(mod).All(type);
}

tvm::Array<Var> FreeVars(const Expr& expr) {
  return VarVisitor().Free(expr);
}

tvm::Array<Var> BoundVars(const Expr& expr) {
  return VarVisitor().Bound(expr);
}

tvm::Array<Var> BoundVars(const Pattern& pat) {
  return VarVisitor().Bound(pat);
}

tvm::Array<Var> AllVars(const Expr& expr) {
  return VarVisitor().All(expr);
}

TVM_REGISTER_API("relay._analysis.free_vars")
.set_body_typed(FreeVars);

TVM_REGISTER_API("relay._analysis.bound_vars")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
      NodeRef x = args[0];
      if (x.as_derived<ExprNode>()) {
        *ret = BoundVars(Downcast<Expr>(x));
      } else {
        *ret = BoundVars(Downcast<Pattern>(x));
      }
    });

TVM_REGISTER_API("relay._analysis.all_vars")
.set_body_typed(AllVars);

TVM_REGISTER_API("relay._analysis.free_type_vars")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    NodeRef x = args[0];
    Module mod = args[1];
    if (x.as_derived<TypeNode>()) {
      *ret = FreeTypeVars(Downcast<Type>(x), mod);
    } else {
      *ret = FreeTypeVars(Downcast<Expr>(x), mod);
    }
  });

TVM_REGISTER_API("relay._analysis.bound_type_vars")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
      NodeRef x = args[0];
      Module mod = args[1];
      if (x.as_derived<TypeNode>()) {
        *ret = BoundTypeVars(Downcast<Type>(x), mod);
      } else {
        *ret = BoundTypeVars(Downcast<Expr>(x), mod);
      }
    });

TVM_REGISTER_API("relay._analysis.all_type_vars")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
      NodeRef x = args[0];
      Module mod = args[1];
      if (x.as_derived<TypeNode>()) {
        *ret = AllTypeVars(Downcast<Type>(x), mod);
      } else {
        *ret = AllTypeVars(Downcast<Expr>(x), mod);
      }
    });

/*!
 * \brief Get reference counter of each internal ExprNode in body.
 * \param body The body expression.
 * \return The reference count mapping.
 */
std::unordered_map<const Node*, size_t>
GetExprRefCount(const Expr& body) {
  class ExprRefCounter : private ExprVisitor {
   public:
    std::unordered_map<const Node*, size_t>
    Get(const Expr& body) {
      this->VisitExpr(body);
      return std::move(this->visit_counter_);
    }
  };
  return ExprRefCounter().Get(body);
}

template <typename T>
bool IsNDArrayAllGreaterEqual(const runtime::NDArray& tensor, T value) {
  CHECK_EQ(tensor->ctx.device_type, kDLCPU);
  CHECK(tensor->strides == nullptr);
  CHECK_EQ(tensor->byte_offset, 0);
  const T* data = static_cast<const T*>(tensor->data);
  int64_t num_elems = 1;
  for (int i = 0; i < tensor->ndim; ++i) {
    num_elems *= tensor->shape[i];
  }

  for (int64_t i = 0; i < num_elems; i++) {
    if (*data < value) {
      return false;
    }
    data++;
  }
  return true;
}

bool IsAllPositiveConstant(const Expr& expr) {
  // peel through a few common transform ops.
  static const auto& expand_dims = Op::Get("expand_dims");
  static const auto& reshape = Op::Get("reshape");
  static const auto& transpose = Op::Get("transpose");
  static const auto& squeeze = Op::Get("squeeze");

  if (const auto* constant = expr.as<ConstantNode>()) {
    const auto& tensor = constant->data;
    const auto& dtype = tensor->dtype;
    if (dtype.lanes != 1) {
      return false;
    } else if (dtype.code == kDLFloat && dtype.bits == 32) {
      return IsNDArrayAllGreaterEqual<float>(tensor, 0);
    } else if (dtype.code == kDLFloat && dtype.bits == 64) {
      return IsNDArrayAllGreaterEqual<double>(tensor, 0);
    } else if (dtype.code == kDLInt && dtype.bits == 8) {
      return IsNDArrayAllGreaterEqual<int8_t>(tensor, 0);
    } else if (dtype.code == kDLInt && dtype.bits == 32) {
      return IsNDArrayAllGreaterEqual<int32_t>(tensor, 0);
    } else if (dtype.code == kDLUInt && dtype.bits == 8) {
      return IsNDArrayAllGreaterEqual<uint8_t>(tensor, 0);
    } else if (dtype.code == kDLUInt && dtype.bits == 32) {
      return IsNDArrayAllGreaterEqual<uint32_t>(tensor, 0);
    } else {
      return false;
    }
  } else if (const auto* op = expr.as<CallNode>()) {
    // tail recursion.
    if (op->op.same_as(expand_dims) ||
        op->op.same_as(reshape) ||
        op->op.same_as(transpose) ||
        op->op.same_as(squeeze)) {
      return IsAllPositiveConstant(op->args[0]);
    } else {
      return false;
    }
  } else {
    return false;
  }
}

Type TypeSubst(const Type& type, const TypeVar& tvar, const Type& subst) {
  return TypeSubst(type, tvm::Map<TypeVar, Type>({{tvar, subst}}));
}

Expr TypeSubst(const Expr& expr, const TypeVar& tvar, const Type& subst) {
  return TypeSubst(expr, tvm::Map<TypeVar, Type>({{tvar, subst}}));
}

Type TypeSubst(const Type& type, const tvm::Map<TypeVar, Type>& subst_map) {
  return Bind(type, subst_map);
}

Expr TypeSubst(const Expr& expr, const tvm::Map<TypeVar, Type>& subst_map) {
  class TypeSubstMutator : public ExprMutator, public PatternMutator {
   public:
    explicit TypeSubstMutator(const tvm::Map<TypeVar, Type>& subst_map) : subst_map_(subst_map) { }
    Type VisitType(const Type& t) final {
      return TypeSubst(t, subst_map_);
    }
    Var VisitVar(const Var& v) final {
      return Downcast<Var>(VisitExpr(v));
    }

    Pattern VisitPattern(const Pattern& p) final {
      return PatternMutator::VisitPattern(p);
    }

    Clause VisitClause(const Clause& c) final {
      Pattern pat = VisitPattern(c->lhs);
      return ClauseNode::make(pat, VisitExpr(c->rhs));
    }

   private:
    const tvm::Map<TypeVar, Type>& subst_map_;
  };
  return TypeSubstMutator(subst_map).VisitExpr(expr);
}

}  // namespace relay
}  // namespace tvm
