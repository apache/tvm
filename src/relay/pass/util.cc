/*!
 * Copyright (c) 2018 by Contributors
 *
 * \file util.cc
 *
 * \brief Utility functions for Relay.
 */
#include <tvm/relay/pass.h>
#include <tvm/relay/expr_functor.h>
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
    for (const auto& v : bound_type_vars_.data) {
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

  void VisitType(const Type& t) final {
    TypeVarTVisitor(&type_vars_, &bound_type_vars_)
        .VisitType(t);
  }

 private:
  InsertionSet<TypeVar> type_vars_;
  InsertionSet<TypeVar> bound_type_vars_;
};

class VarVisitor : protected ExprVisitor {
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

  Array<Var> Bound(const Expr& expr) {
    this->VisitExpr(expr);
    Array<Var> ret;
    for (const auto& v : bound_vars_.data) {
      ret.push_back(v);
    }
    return ret;
  }

  Array<Var> All(const Expr& expr) {
    this->VisitExpr(expr);
    Array<Var> ret;
    for (const auto& v : vars_.data) {
      ret.push_back(v);
    }
    return ret;
  }

  void Bounded(const Var& v) {
    bound_vars_.Insert(v);
    vars_.Insert(v);
  }

  void VisitExpr_(const VarNode* var) final {
    vars_.Insert(GetRef<Var>(var));
  }

  void VisitExpr_(const FunctionNode* op) final {
    for (const auto& param : op->params) {
      Bounded(param);
    }
    VisitExpr(op->body);
  }

  void VisitExpr_(const LetNode* op) final {
    Bounded(op->var);
    VisitExpr(op->value);
    VisitExpr(op->body);
  }

 private:
  InsertionSet<Var> vars_;
  InsertionSet<Var> bound_vars_;
};

tvm::Array<TypeVar> FreeTypeVars(const Expr& expr) {
  return TypeVarEVisitor().Free(expr);
}

tvm::Array<TypeVar> FreeTypeVars(const Type& type) {
  return TypeVarEVisitor().Free(type);
}

tvm::Array<TypeVar> BoundTypeVars(const Expr& expr) {
  return TypeVarEVisitor().Bound(expr);
}

tvm::Array<TypeVar> BoundTypeVars(const Type& type) {
  return TypeVarEVisitor().Bound(type);
}

tvm::Array<TypeVar> AllTypeVars(const Expr& expr) {
  return TypeVarEVisitor().All(expr);
}

tvm::Array<TypeVar> AllTypeVars(const Type& type) {
  return TypeVarEVisitor().All(type);
}

tvm::Array<Var> FreeVars(const Expr& expr) {
  return VarVisitor().Free(expr);
}

tvm::Array<Var> BoundVars(const Expr& expr) {
  return VarVisitor().Bound(expr);
}

tvm::Array<Var> AllVars(const Expr& expr) {
  return VarVisitor().All(expr);
}

TVM_REGISTER_API("relay._ir_pass.free_vars")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = FreeVars(args[0]);
  });

TVM_REGISTER_API("relay._ir_pass.bound_vars")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
      *ret = BoundVars(args[0]);
    });

TVM_REGISTER_API("relay._ir_pass.all_vars")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
      *ret = AllVars(args[0]);
    });

TVM_REGISTER_API("relay._ir_pass.free_type_vars")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    NodeRef x = args[0];
    if (x.as<TypeNode>()) {
      *ret = FreeTypeVars(Downcast<Type>(x));
    } else {
      *ret = FreeTypeVars(Downcast<Expr>(x));
    }
  });

TVM_REGISTER_API("relay._ir_pass.bound_type_vars")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
      NodeRef x = args[0];
      if (x.as<TypeNode>()) {
        *ret = BoundTypeVars(Downcast<Type>(x));
      } else {
        *ret = BoundTypeVars(Downcast<Expr>(x));
      }
    });

TVM_REGISTER_API("relay._ir_pass.all_type_vars")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
      NodeRef x = args[0];
      if (x.as<TypeNode>()) {
        *ret = AllTypeVars(Downcast<Type>(x));
      } else {
        *ret = AllTypeVars(Downcast<Expr>(x));
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

}  // namespace relay
}  // namespace tvm
