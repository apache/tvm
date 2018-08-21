/*!
 *  Copyright (c) 2018 by Contributors
 * \file alpha_eq.cc
 * \brief Compute the set of variables not bound in the expression.
 */
#include "tvm/relay/compiler/alpha_eq.h"
#include "tvm/relay/expr_visitor.h"
#include "./type_visitor.h"

namespace tvm {
namespace relay {

using namespace tvm::runtime;

struct TypeAlphaEq : TypeVisitor<const Type &> {
  tvm::Map<TypeParam, TypeParam> eq_map;
  bool equal;

  TypeAlphaEq() : eq_map(), equal(true) {}

  void DataTypeEqual(const DataType & dt1, const DataType & dt2) {
      equal = equal && dt1 == dt2;
  }
  void ShapeEqual(Array<ShapeExpr> s1, Array<ShapeExpr> s2) {
  }

  void VisitType_(const TensorTypeNode *tt1, const Type &t2) override {
    if (const TensorTypeNode *tt2 = t2.as<TensorTypeNode>()) {
      DataTypeEqual(tt1->dtype, tt2->dtype);
      ShapeEqual(tt1->shape, tt2->shape);
    } else {
      equal = false;
    }
  }

//   void VisitType_(const TypeVarNode *bt1, const Type &t2) override {
//     if (const TypeVarNode *bt2 = t2.as<TypeVarNode>()) {
//       equal = equal && bt1 == bt2;
//       return;
//     } else {
//       equal = false;
//     }
//   }

  void VisitType_(const TypeParamNode *ti1, const Type &t2) override {
    if (const TypeParamNode *ti2 = t2.as<TypeParamNode>()) {
      auto tid1 = GetRef<TypeParam>(ti1);
      auto tid2 = GetRef<TypeParam>(ti2);

      // We handle open terms with this rule assuming variables are identical.
      //
      // Not sure if we should do this.
      if (tid1 == tid2) {
        return;
      }

      // Check that they are same kind
      if (tid1->kind != tid2->kind) {
        equal = false;
        return;
      }

      // Next we see if there is mapping for local1 into the rhs term.
      // If there is we check to see if those are equal.
      if (eq_map.find(tid1) != eq_map.end()) {
        equal = equal && eq_map[tid1] == tid2;
      } else {
        equal = false;
      }
    } else {
      equal = false;
    }
  }

  void VisitType_(const FuncTypeNode *op, const Type &t2) override {
    if (const FuncTypeNode *ta2 = t2.as<FuncTypeNode>()) {
      if (op->arg_types.size() != ta2->arg_types.size()) {
        equal = false;
        return;
      }

      for (size_t i = 0; i < op->arg_types.size(); i++) {
        this->VisitType(op->arg_types[i], ta2->arg_types[i]);
        if (!equal) {
          return;
        }
      }

      this->VisitType(op->ret_type, ta2->ret_type);
    } else {
      equal = false;
    }
  }

  void VisitType_(const TypeFunctionNode *op, const Type &t2) override {
  }
//   void VisitType_(const TupleTypeNode *op, const Type &t2) override {
//     if (const TupleTypeNode *pt = t2.as<TupleTypeNode>()) {
//       if (op->fields.size() != pt->fields.size()) {
//         equal = false;
//         return;
//       }

//       for (size_t i = 0U; i < op->fields.size(); i++) {
//         if (!equal) {
//           return;
//         }
//         this->VisitType(op->fields[i], pt->fields[i]);
//       }
//     } else {
//       equal = false;
//     }
//   }

//   void VisitType_(const TypeCallNode *tyn1, const Type &t2) override {
//     TypeCall tycall = GetRef<TypeCall>(tyn1);
//     if (const TypeCallNode *tyn2 = t2.as<TypeCallNode>()) {
//       if (tycall->func != tyn2->func) {
//         equal = false;
//         return;
//       }

//       if (tycall->args.size() != tyn2->args.size()) {
//         equal = false;
//         return;
//       }

//       for (size_t i = 0U; i < tycall->args.size(); i++) {
//         this->VisitType(tycall->args[i], tyn2->args[i]);
//       }
//     } else {
//       equal = false;
//     }
//   }
};

bool alpha_eq(const Type &t1, const Type &t2) {
  TypeAlphaEq aeq;
  aeq.VisitType(t1, t2);
  return aeq.equal;
}

// struct AlphaEq : ExprVisitor<const Expr &> {
//  public:
//   tvm::Map<LocalId, LocalId> eq_map;
//   bool equal;
//   AlphaEq() : eq_map(), equal(true) {}

//   void VisitExpr_(const LocalIdNode *e1, const Expr &e2) override {
//     if (const LocalIdNode *id2 = e2.as<LocalIdNode>()) {
//       auto local1 = GetRef<LocalId>(e1);
//       auto local2 = GetRef<LocalId>(id2);
//       //
//       // We handle open terms with this rule assuming variables are identical.
//       //
//       // Not sure if we should do this.
//       if (local1 == local2) {
//         equal = true;
//         return;
//       }

//       // Next we see if there is mapping for local1 into the rhs term.
//       // If there is we check to see if those are equal.
//       if (eq_map.find(local1) != eq_map.end()) {
//         equal = equal && eq_map[local1] == local2;
//       } else {
//         equal = false;
//       }
//     } else {
//       equal = false;
//     }
//   }

//   void VisitExpr_(const GlobalIdNode *g1, const Expr &e2) override {
//     if (const GlobalIdNode *g2 = e2.as<GlobalIdNode>()) {
//       equal = equal && g1 == g2;
//     } else {
//       equal = false;
//     }
//   }

//   void VisitExpr_(const OperatorIdNode *i1, const Expr &e2) override {
//     if (const OperatorIdNode *i2 = e2.as<OperatorIdNode>()) {
//       equal = equal && i1 == i2;
//     } else {
//       equal = false;
//     }
//   }

//   void VisitExpr_(const TupleNode *pl1, const Expr &e2) override {
//     Tuple prod1 = GetRef<Tuple>(pl1);
//     if (const TupleNode *pl2 = e2.as<TupleNode>()) {
//       Tuple prod2 = GetRef<Tuple>(pl2);
//       if (prod1->fields.size() != prod2->fields.size()) {
//         equal = false;
//         return;
//       }

//       for (size_t i = 0U; i < prod1->fields.size(); i++) {
//         this->VisitExpr(prod1->fields[i], prod2->fields[i]);
//       }
//     } else {
//       equal = false;
//     }
//   }

//   void VisitExpr_(const ParamNode *p1, const Expr &e2) override {
//     if (const ParamNode *p2 = e2.as<ParamNode>()) {
//       eq_map.Set(p1->id, p2->id);
//       equal = equal && alpha_eq(p1->type, p2->type);
//     } else {
//       equal = false;
//     }
//   }

//   void VisitExpr_(const FunctionNode *func1, const Expr &e2) override {
//     if (const FunctionNode *func2 = e2.as<FunctionNode>()) {
//       if (func1->params.size() != func2->params.size()) {
//         equal = false;
//         return;
//       }

//       for (size_t i = 0U; i < func1->params.size(); i++) {
//         this->VisitExpr(func1->params[i], func2->params[i]);
//       }

//       this->VisitExpr(func1->body, func2->body);
//     } else {
//       equal = false;
//     }
//   }

//   void VisitExpr_(const CallNode *op, const Expr &e2) override {
//     if (const CallNode *call = e2.as<CallNode>()) {
//       this->VisitExpr(op->fn, call->fn);

//       if (op->args.size() != call->args.size()) {
//         equal = false;
//         return;
//       }

//       for (size_t i = 0U; i < op->args.size(); i++) {
//         this->VisitExpr(op->args[i], call->args[i]);
//       }

//     } else {
//       equal = false;
//     }
//   }

//   void VisitExpr_(const LetNode *op, const Expr &e2) override {
//     if (const LetNode *let = e2.as<LetNode>()) {
//       eq_map.Set(op->id, let->id);
//       this->VisitExpr(op->value, let->value);
//       this->VisitExpr(op->body, let->body);
//     } else {
//       equal = false;
//     }
//   }
// };

// bool alpha_eq(const Expr &e1, const Expr &e2) {
//   AlphaEq eq;
//   eq.VisitExpr(e1, e2);
//   return eq.equal;
// }

// // TODO(@jroesch): move to correct namespace?
// TVM_REGISTER_API("relay._make._alpha_eq")
//     .set_body([](TVMArgs args, TVMRetValue *ret) {
//       Expr e1 = args[0];
//       Expr e2 = args[1];
//       *ret = alpha_eq(e1, e2);
//     });

TVM_REGISTER_API("relay._make._type_alpha_eq")
    .set_body([](TVMArgs args, TVMRetValue *ret) {
      Type t1 = args[0];
      Type t2 = args[1];
      *ret = alpha_eq(t1, t2);
    });

}  // namespace relay
}  // namespace tvm
