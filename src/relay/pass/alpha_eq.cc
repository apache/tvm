/*!
 *  Copyright (c) 2018 by Contributors
 * \file src/tvm/relay/pass/alpha_eq.cc
 * \brief Check that two type are syntactically equal up to alpha equivalence.
 */
#include <tvm/ir_pass.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/runtime/ndarray.h>
#include "./type_visitor.h"
#include "tvm/relay/pass.h"

namespace tvm {
namespace relay {

using namespace tvm::runtime;

bool SameNDArray(const NDArray& lhs, const NDArray& rhs) {
  if (lhs.defined() != rhs.defined()) {
    return false;
  } else if (lhs.same_as(rhs)) {
    return true;
  } else {
    auto ldt = lhs->dtype;
    auto rdt = rhs->dtype;
    CHECK_EQ(lhs->ctx.device_type, kDLCPU) << "can only compare CPU tensor";
    CHECK_EQ(rhs->ctx.device_type, kDLCPU) << "can only compare CPU tensor";
    if (ldt.code == rdt.code && ldt.lanes == rdt.lanes && ldt.bits == rdt.bits) {
      size_t s = GetDataSize(*lhs.operator->());
      return memcmp(lhs->data, rhs->data, s) == 0;
    } else {
      return false;
    }
  }
}

struct TypeAlphaEq : TypeVisitor<const Type&> {
  tvm::Map<TypeParam, TypeParam> eq_map;
  bool equal;

  TypeAlphaEq() : eq_map(), equal(true) {}

  void DataTypeEqual(const DataType& dt1, const DataType& dt2) {
    if (dt1 != dt2) {
      equal = false;
    }
  }

  void ShapeEqual(const Array<IndexExpr>& s1, const Array<IndexExpr>& s2) {
    if (s1.size() != s2.size()) {
      equal = false;
      return;
    }
    for (size_t i = 0; i < s1.size(); ++i) {
      if (!tvm::ir::Equal(s1[i], s2[i])) {
        equal = false;
        return;
      }
    }
  }

  void VisitType_(const TensorTypeNode* tt1, const Type& t2) final {
    if (const TensorTypeNode* tt2 = t2.as<TensorTypeNode>()) {
      DataTypeEqual(tt1->dtype, tt2->dtype);
      ShapeEqual(tt1->shape, tt2->shape);
    } else {
      equal = false;
    }
  }

  void VisitType_(const IncompleteTypeNode* bt1, const Type& t2) final {
    if (const IncompleteTypeNode* bt2 = t2.as<IncompleteTypeNode>()) {
      equal = equal && bt1 == bt2;
      return;
    } else {
      equal = false;
    }
  }

  void VisitType_(const TypeParamNode* ti1, const Type& t2) final {
    if (const TypeParamNode* ti2 = t2.as<TypeParamNode>()) {
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

  void VisitType_(const FuncTypeNode* op, const Type& t2) final {
    if (const FuncTypeNode* ta2 = t2.as<FuncTypeNode>()) {
      if (op->arg_types.size() != ta2->arg_types.size()
          || op->type_params.size() != ta2->type_params.size()
          || op->type_constraints.size() != ta2->type_constraints.size()) {
        equal = false;
        return;
      }

      // must visit params first so they are appropriate entered
      // into equality map
      for (size_t i = 0; i < op->type_params.size(); i++) {
        eq_map.Set(op->type_params[i], ta2->type_params[i]);
        this->VisitType(op->type_params[i], ta2->type_params[i]);
        if (!equal) {
          return;
        }
      }

      for (size_t i = 0; i < op->arg_types.size(); i++) {
        this->VisitType(op->arg_types[i], ta2->arg_types[i]);
        if (!equal) {
          return;
        }
      }

      this->VisitType(op->ret_type, ta2->ret_type);
      if (!equal) {
        return;
      }

      for (size_t i = 0; i < op->type_constraints.size(); i++) {
        this->VisitType(op->type_constraints[i], ta2->type_constraints[i]);
        if (!equal) {
          return;
        }
      }
    } else {
      equal = false;
    }
  }

  void VisitType_(const TypeRelationNode* tr1, const Type& t2) final {
    if (const TypeRelationNode* tr2 = t2.as<TypeRelationNode>()) {
      if (tr1->func != tr2->func
          || tr1->num_inputs != tr2->num_inputs
          || tr1->attrs != tr2->attrs) {
        equal = false;
        return;
      }

      if (tr1->args.size() != tr2->args.size()) {
        equal = false;
        return;
      }

      for (size_t i = 0; i < tr1->args.size(); i++) {
        this->VisitType(tr1->args[i], tr2->args[i]);
        if (!equal) {
          return;
        }
      }
    } else {
      equal = false;
    }
  }

  void VisitType_(const TupleTypeNode* op, const Type& t2) final {
    if (const TupleTypeNode* pt = t2.as<TupleTypeNode>()) {
      if (op->fields.size() != pt->fields.size()) {
        equal = false;
        return;
      }

      for (size_t i = 0U; i < op->fields.size(); i++) {
        if (!equal) {
          return;
        }
        this->VisitType(op->fields[i], pt->fields[i]);
      }
    } else {
      equal = false;
    }
  }
};

bool AlphaEqual(const Type& t1, const Type& t2) {
  TypeAlphaEq aeq;
  aeq.VisitType(t1, t2);
  return aeq.equal;
}

struct AlphaEq : ExprFunctor<void(const Expr&, const Expr&)> {
 public:
  tvm::Map<Var, Var> eq_map;

  bool equal;
  AlphaEq() : eq_map(), equal(true) {}

  void VisitExpr_(const VarNode* e1, const Expr& e2) final {
    if (const VarNode* id2 = e2.as<VarNode>()) {
      auto local1 = GetRef<Var>(e1);
      auto local2 = GetRef<Var>(id2);
      // We handle open terms with this rule assuming variables are identical.
      if (local1 == local2) {
        equal = true;
        return;
      }

      // Next we see if there is mapping for local1 into the rhs term.
      // If there is we check to see if those are equal.
      if (eq_map.find(local1) != eq_map.end()) {
        equal = equal && eq_map[local1] == local2;
      } else {
        equal = false;
      }
    } else {
      equal = false;
    }
  }

  void VisitExpr_(const GlobalVarNode* g1, const Expr& e2) final {
    if (const GlobalVarNode* g2 = e2.as<GlobalVarNode>()) {
      equal = equal && g1 == g2;
    } else {
      equal = false;
    }
  }

  void VisitExpr_(const TupleNode* pl1, const Expr& e2) final {
    Tuple prod1 = GetRef<Tuple>(pl1);
    if (const TupleNode* pl2 = e2.as<TupleNode>()) {
      Tuple prod2 = GetRef<Tuple>(pl2);
      if (prod1->fields.size() != prod2->fields.size()) {
        equal = false;
        return;
      }

      for (size_t i = 0U; i < prod1->fields.size(); i++) {
        this->VisitExpr(prod1->fields[i], prod2->fields[i]);
      }
    } else {
      equal = false;
    }
  }

  void VisitExpr_(const ParamNode* p1, const Expr& e2) final {
    if (const ParamNode* p2 = e2.as<ParamNode>()) {
      eq_map.Set(p1->var, p2->var);
      equal = equal && AlphaEqual(p1->type, p2->type);
    } else {
      equal = false;
    }
  }

  void VisitExpr_(const FunctionNode* func1, const Expr& e2) final {
    if (const FunctionNode* func2 = e2.as<FunctionNode>()) {
      if (func1->params.size() != func2->params.size()) {
        equal = false;
        return;
      }

      for (size_t i = 0U; i < func1->params.size(); i++) {
        this->VisitExpr(func1->params[i], func2->params[i]);
      }

      this->VisitExpr(func1->body, func2->body);
    } else {
      equal = false;
    }
  }

  void VisitExpr_(const CallNode* op, const Expr& e2) final {
    if (const CallNode* call = e2.as<CallNode>()) {
      this->VisitExpr(op->op, call->op);

      if (op->args.size() != call->args.size()) {
        equal = false;
        return;
      }

      for (size_t i = 0U; i < op->args.size(); i++) {
        this->VisitExpr(op->args[i], call->args[i]);
      }

    } else {
      equal = false;
    }
  }

  void VisitExpr_(const LetNode* op, const Expr& e2) final {
    if (const LetNode* let = e2.as<LetNode>()) {
      eq_map.Set(op->var, let->var);
      this->VisitExpr(op->value, let->value);
      this->VisitExpr(op->body, let->body);
    } else {
      equal = false;
    }
  }

  void VisitExpr_(const IfNode* op, const Expr& e2) final {
    if (const IfNode* i = e2.as<IfNode>()) {
      VisitExpr(op->cond, i->cond);
      VisitExpr(op->true_branch, i->true_branch);
      VisitExpr(op->false_branch, i->false_branch);
    } else {
      equal = false;
    }
  }

  void VisitExpr_(const OpNode* op, const Expr& e2) final {
    if (const OpNode* o = e2.as<OpNode>()) {
      equal = equal && op->name == o->name;
    } else {
      equal = false;
    }
  }

  void VisitExpr_(const ConstantNode* op, const Expr& e2) final {
    if (const ConstantNode* c = e2.as<ConstantNode>()) {
      if (AlphaEqual(op->tensor_type(), c->tensor_type())) {
        equal = equal && SameNDArray(op->data, c->data);
      } else {
        equal = false;
      }
    } else {
      equal = false;
    }
  }
};

bool AlphaEqual(const Expr& e1, const Expr& e2) {
  AlphaEq eq;
  eq.VisitExpr(e1, e2);
  return eq.equal;
}

// TODO(@jroesch): move to correct namespace?
TVM_REGISTER_API("relay._make._alpha_equal")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      Expr e1 = args[0];
      Expr e2 = args[1];
      *ret = AlphaEqual(e1, e2);
    });

TVM_REGISTER_API("relay._make._type_alpha_equal")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      Type t1 = args[0];
      Type t2 = args[1];
      *ret = AlphaEqual(t1, t2);
    });

}  // namespace relay
}  // namespace tvm
