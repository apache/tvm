/*!
 * Copyright (c) 2020 by Contributors
 * \file auto_scheduler/expr_hasher.h
 * \brief Hash function for a tvm::Expr
 */

#ifndef TVM_ANSOR_EXPR_HASHER_H_
#define TVM_ANSOR_EXPR_HASHER_H_

#include <tvm/tir/expr.h>
#include <tvm/tir/expr_functor.h>
#include <tvm/tir/op.h>
#include <dmlc/common.h>

namespace tvm {

/*! \brief Assign a hash value for a tvm::Expr */
class ExprHasher: public tir::ExprFunctor<size_t(const PrimExpr& n)> {
 public:
  size_t VisitExpr_(const tir::AddNode* op) final {
    return VisitExpr(op->a) + VisitExpr(op->b);
  }

  size_t VisitExpr_(const tir::SubNode* op) final {
    return VisitExpr(op->a) - VisitExpr(op->b);
  }

  size_t VisitExpr_(const tir::MulNode* op) final {
    return VisitExpr(op->a) * VisitExpr(op->b);
  }

  size_t VisitExpr_(const tir::DivNode* op) final {
    size_t t = VisitExpr(op->b);
    if (t != 0) {
      return VisitExpr(op->a) / t;
    } else {
      return dmlc::HashCombine(VisitExpr(op->a), 0x5A);
    }
  }

  size_t VisitExpr_(const tir::FloorDivNode* op) final {
    size_t t = VisitExpr(op->b);
    if (t != 0) {
      return VisitExpr(op->a) / t;
    } else {
      return dmlc::HashCombine(VisitExpr(op->a), 0x5B);
    }
  }

  size_t VisitExpr_(const tir::ModNode* op) final {
    size_t t = VisitExpr(op->b);
    if (t != 0) {
      return VisitExpr(op->a) % t;
    } else {
      return dmlc::HashCombine(VisitExpr(op->a), 0x5C);
    }
  }

  size_t VisitExpr_(const tir::FloorModNode* op) final {
    size_t t = VisitExpr(op->b);
    if (t != 0) {
      return VisitExpr(op->a) % t;
    } else {
      return dmlc::HashCombine(VisitExpr(op->a), 0x5D);
    }
  }

  size_t VisitExpr_(const tir::CallNode* op) final {
    size_t ret = ObjectHash()(op->func);
    for (size_t i = 0; i < op->args.size(); ++i) {
      ret = dmlc::HashCombine(ret, VisitExpr(op->args[i]));
    }
    return ret;
  }

  size_t VisitExpr_(const tir::VarNode* op) final {
    return std::hash<const tir::VarNode*>()(op);
  }

  size_t VisitExpr_(const tir::FloatImmNode* op) final {
    return std::hash<double>()(op->value);
  }

  size_t VisitExpr_(const tir::IntImmNode* op) final {
    return std::hash<int64_t>()(op->value);
  }

  size_t VisitExprDefault_(const Object* op) final {
    LOG(WARNING) << "Encounter undefined node in ExprHasher: "
                 << Object::_type_key;
    return std::hash<const Object*>()(op);
  }
};

}  // namespace tvm

#endif  // TVM_ANSOR_EXPR_HASHER_H_
