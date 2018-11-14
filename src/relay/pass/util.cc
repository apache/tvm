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

// FreeTypeVar
class FreeTypeVarTVisitor : public TypeVisitor {
 public:
  FreeTypeVarTVisitor(
      Array<TypeVar>* free_vars,
      std::unordered_set<TypeVar, NodeHash, NodeEqual>* bound_vars)
      : free_vars_(free_vars), bound_vars_(bound_vars) { }

  void VisitType_(const TypeVarNode* tp) final {
    TypeVar var = GetRef<TypeVar>(tp);
    if (bound_vars_->count(var) == 0) {
      free_vars_->push_back(var);
    }
  }

  void VisitType_(const FuncTypeNode* f) final {
    for (auto type_param : f->type_params) {
      bound_vars_->insert(type_param);
    }
    TypeVisitor::VisitType_(f);
  }

 private:
  Array<TypeVar>* free_vars_;
  std::unordered_set<TypeVar, NodeHash, NodeEqual>* bound_vars_;
};

class FreeTypeVarEVisitor : private ExprVisitor {
 public:
  Array<TypeVar> Find(const Expr& expr) {
    this->VisitExpr(expr);
    return free_vars_;
  }

  Array<TypeVar> Find(const Type& type) {
    this->VisitType(type);
    return free_vars_;
  }

  void VisitExpr_(const FunctionNode* f) final {
    for (const auto& tp : f->type_params) {
      bound_vars_.insert(tp);
    }
    ExprVisitor::VisitExpr_(f);
  }

  void VisitType(const Type& t) final {
    FreeTypeVarTVisitor(&free_vars_, &bound_vars_)
        .VisitType(t);
  }

 private:
  // The result list
  Array<TypeVar> free_vars_;
  std::unordered_set<TypeVar, NodeHash, NodeEqual> bound_vars_;
};

class FreeVarVisitor : protected ExprVisitor {
 public:
  Array<Var> Find(const Expr& expr) {
    this->VisitExpr(expr);
    return free_vars_;
  }

  void VisitExpr_(const VarNode* var) final {
    if (bound_vars_.count(var) == 0) {
      free_vars_.push_back(GetRef<Var>(var));
    }
  }

  void VisitExpr_(const FunctionNode* op) final {
    for (const auto& param : op->params) {
      bound_vars_.insert(param.operator->());
    }
    VisitExpr(op->body);
  }

  void VisitExpr_(const LetNode* op) final {
    bound_vars_.insert(op->var.operator->());
    VisitExpr(op->value);
    VisitExpr(op->body);
  }

 private:
  // The result list
  Array<Var> free_vars_;
  std::unordered_set<const VarNode*> bound_vars_;
};

tvm::Array<TypeVar> FreeTypeVars(const Expr& expr) {
  return FreeTypeVarEVisitor().Find(expr);
}

tvm::Array<TypeVar> FreeTypeVars(const Type& type) {
  return FreeTypeVarEVisitor().Find(type);
}

tvm::Array<Var> FreeVars(const Expr& expr) {
  return FreeVarVisitor().Find(expr);
}

TVM_REGISTER_API("relay._ir_pass.free_vars")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = FreeVars(args[0]);
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
