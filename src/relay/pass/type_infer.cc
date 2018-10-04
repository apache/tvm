/*!
 *  Copyright (c) 2018 by Contributors
 * \file type_infer.cc
 * \brief Relay type inference and checking.
 *
 * This file implements one of the most important passes to the
 * Relay IR. In order to do many transformations and generate the
 * most efficient code we need to obtain type information for the
 * IR.
 *
 * Like computation graphs the IR leaves most type information
 * implicit and relies performing analysis of the program to
 * generate this information.
 *
 * This pass given an expression `e` will infer a type `t` for
 * the expression simultaneous checking the property `e : t`
 * (i.e we can show e has type t).
 *
 * If we can not infer a type or there are conflicting typing
 * constraints we will trigger an error.
 */

#include <tvm/relay/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pass.h>
#include "type_solver.h"
#include "type_subst.h"

namespace tvm {
namespace relay {
//
// The inference algorithm can roughly be devided into three stages:
// - Populate the constraints by visiting the expression (TypeInferencer.GetType)
//   - solver.AddConstraint and solver.Unify are called to populate the necessary constraints
// - Solve the constraints (solver_.Solve)
// - Recreate expression with the resolved checked_type (Resolver.VisitExpr)
//
class TypeInferencer : private ExprFunctor<Type(const Expr&)> {
 public:
  // constructors
  TypeInferencer()
      : env_(EnvironmentNode::make({})) {
  }
  explicit TypeInferencer(Environment env)
      : env_(env) {
  }

  // inference the type of expr.
  Expr Infer(Expr expr);

 private:
  // type resolver that maps back to type
  class Resolver;
  // internal environment
  Environment env_;
  // map from expression to checked type
  // type inferencer will populate it up
  std::unordered_map<Expr, Type, NodeHash, NodeEqual> type_map_;
  // The solver used by the inferencer.
  TypeSolver solver_;
  // Unify two types
  Type Unify(const Type& t1, const Type& t2, const Span& span) {
    // TODO(tqchen, jroesch): propagate span to solver
    try {
      return solver_.Unify(t1, t2);
    } catch (const dmlc::Error &e) {
      LOG(FATAL)
          << "Error unifying `"
          << t1
          << "` and `"
          << t2
          << "`: " << e.what();
      return Type();
    }
  }
  // Lazily get type for expr
  // will call visit to deduce it if it is not in the type_map_
  Type GetType(const Expr &expr) {
    auto it = type_map_.find(expr);
    if (it != type_map_.end()) {
      return it->second;
    }
    Type ret = this->VisitExpr(expr);
    type_map_[expr] = ret;
    return ret;
  }

  // Visitor logics
  Type VisitExpr_(const VarNode* op) final {
    // The type of Var can already been lookedup in type_map_;
    LOG(FATAL) << "Cannot find binding for var " << GetRef<Var>(op);
    return Type();
  }

  Type VisitExpr_(const ParamNode* op) final {
    // directly handled by Funtion
    LOG(FATAL) << "not reached";
    return Type();
  }

  Type VisitExpr_(const GlobalVarNode* op) final {
    GlobalVar var = GetRef<GlobalVar>(op);
    Expr e = env_->Lookup(var);
    return e->checked_type();
  }

  Type VisitExpr_(const ConstantNode* op) final {
    return op->tensor_type();
  }

  Type VisitExpr_(const TupleNode* op) final {
    // TODO(tqchen, jroesch)
    // tuple should be a constraint in the type solver
    // to handle cases where the field type is not known.
    Array<Type> fields;
    for (Expr field : op->fields) {
      fields.push_back(GetType(field));
    }
    return TupleTypeNode::make(fields);
  }

  Type VisitExpr_(const OpNode* op) final {
    return op->op_type;
  }

  Type VisitExpr_(const LetNode* op) final {
    Type vtype = GetType(op->value);
    if (op->value_type.defined()) {
      vtype = Unify(vtype, op->value_type, op->span);
    }
    CHECK(!type_map_.count(op->var));
    // NOTE: no scoping is necessary becase var are unique in program
    type_map_[op->var] = vtype;
    return GetType(op->body);
  }

  Type VisitExpr_(const IfNode* op) final {
    // Ensure the type of the guard is of Tensor[Bool, ()],
    // that is a rank-0 boolean tensor.
    Type cond_type = this->GetType(op->cond);
    this->Unify(cond_type,
                TensorTypeNode::Scalar(tvm::Bool()),
                op->cond->span);
    Type checked_true = this->GetType(op->true_branch);
    Type checked_false = this->GetType(op->false_branch);
    return this->Unify(checked_true, checked_false, op->span);
  }

  // Handle special case basic primitive operator,
  // if successful return the return type
  Type PrimitiveCall(const FuncTypeNode* op,
                     Array<Type> arg_types,
                     const Attrs& attrs) {
    if (op->type_params.size() != arg_types.size() + 1) return Type();
    if (op->type_constraints.size() != 1) return Type();
    const TypeRelationNode* rel = op->type_constraints[0].as<TypeRelationNode>();
    if (rel == nullptr) return Type();
    // validate if the type parameter matches up
    for (size_t i = 0; i < op->type_params.size(); ++i) {
      if (!op->type_params[i].same_as(rel->args[i])) return Type();
    }
    Type rtype = IncompleteTypeNode::make(TypeParamNode::Kind::kType);
    arg_types.push_back(rtype);
    // we can do simple replacement here
    solver_.AddConstraint(TypeRelationNode::make(
        rel->func, arg_types, arg_types.size() - 1, attrs));
    return rtype;
  }

  // instantiate the function type with fresh
  FuncType Instantiate(const FuncTypeNode* fn_ty, Array<Type>* ty_args) {
    tvm::Map<TypeParam, Type> subst_map;

    // Build a subsitituion map up from the function type and type arguments.
    // Eventually allow the type vars to be passed in.
    for (auto ty_param : fn_ty->type_params) {
      IncompleteType fresh = IncompleteTypeNode::make(ty_param->kind);
      subst_map.Set(ty_param, fresh);
      ty_args->push_back(fresh);
    }
    Type ret_type = fn_ty->ret_type;

    // If the function type is incomplete, place a new IncompleteType
    // This relax the fn_ty to inputs -> Any
    // The type checking can still pass when there are additional constraints on the type
    // This is a temporary work around to check recursive functions whose
    // return type is not yet known.
    if (!ret_type.defined()) {
      ret_type = IncompleteTypeNode::make(TypeParamNode::Kind::kType);
    }
    Type inst_ty = FuncTypeNode::make(fn_ty->arg_types,
                                      ret_type, {},
                                      fn_ty->type_constraints);
    inst_ty = TypeSubst(inst_ty, subst_map);
    return Downcast<FuncType>(inst_ty);
  }

  // Handle general call node.
  Type GeneralCall(const CallNode* op, Array<Type> arg_types) {
    Type ftype = GetType(op->op);
    auto* fn_ty_node = ftype.as<FuncTypeNode>();
    CHECK(fn_ty_node != nullptr)
        << "only expressions with function types can be called, at "
        << op->span;

    Array<Type> type_args;
    FuncType fn_ty = Instantiate(fn_ty_node, &type_args);
    size_t type_arity = fn_ty->arg_types.size();
    size_t number_of_args = arg_types.size();

    if (type_arity != number_of_args) {
      if (type_arity < number_of_args) {
        LOG(FATAL) << "the function is provided too many arguments " << op->span;
      } else {
        LOG(FATAL) << "the function is provided too few arguments" << op->span;
      }
    }
    for (size_t i = 0; i < fn_ty->arg_types.size(); i++) {
      this->Unify(fn_ty->arg_types[i], arg_types[i], op->args[i]->span);
    }

    for (auto cs : fn_ty->type_constraints) {
      solver_.AddConstraint(cs);
    }
    return fn_ty->ret_type;
  }

  Type VisitExpr_(const CallNode* op) final {
    // Fast path: well-formed primitive op
    Array<Type> arg_types;
    for (Expr arg : op->args) {
      arg_types.push_back(GetType(arg));
    }
    if (const OpNode* opnode = op->op.as<OpNode>()) {
      Type rtype = PrimitiveCall(opnode->op_type.as<FuncTypeNode>(),
                                 arg_types,
                                 op->attrs);
      if (rtype.defined()) return rtype;
    }
    return GeneralCall(op, arg_types);
  }

  Type VisitExpr_(const FunctionNode* f) final {
    for (auto param : f->params) {
      type_map_[param->var] = param->type;
      type_map_[param] = param->type;
    }
    Type rtype = GetType(f->body);
    // Run solver using the currently known information
    solver_.Solve();
    // Trying to resolve
    Array<Type> arg_types;
    for (size_t i = 0; i < f->params.size(); ++i) {
      Param param = f->params[i];
      Type atype = solver_.Resolve(param->type);
      CHECK(atype.as<IncompleteTypeNode>() == nullptr)
          << "Cannot resolve type of " << i
          << "-th parameter of function at" << f->span;
      arg_types.push_back(atype);
    }
    rtype = solver_.Resolve(rtype);
    CHECK(rtype.as<IncompleteTypeNode>() == nullptr)
        << "Cannot resolve return type of function at" << f->span;
    // do not support constraint lifting for now.
    return FuncTypeNode::make(arg_types, rtype, f->type_params, {});
  }
};

class TypeInferencer::Resolver : public ExprMutator {
 public:
  Resolver(const std::unordered_map<Expr, Type, NodeHash, NodeEqual>& tmap,
           TypeSolver* solver)
      : tmap_(tmap), solver_(solver) {
  }

  Expr VisitExpr_(const VarNode* op) final {
    return AttachCheckedType(op);
  }

  Expr VisitExpr_(const ConstantNode* op) final {
    return AttachCheckedType(op);
  }

  Expr VisitExpr_(const GlobalVarNode* op) final {
    return AttachCheckedType(op);
  }

  Expr VisitExpr_(const OpNode* op) final {
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const TupleNode* op) final {
    return AttachCheckedType(op);
  }

  Expr VisitExpr_(const ParamNode* op) final {
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    return AttachCheckedType(op);
  }

  Expr VisitExpr_(const CallNode* op) final {
    return AttachCheckedType(op);
  }

  Expr VisitExpr_(const LetNode* op) final {
    return AttachCheckedType(op);
  }

  Expr VisitExpr_(const IfNode* op) final {
    return AttachCheckedType(op);
  }

  // attach checked type to the mutated node.
  template<typename T>
  Expr AttachCheckedType(const T* op) {
    auto it = tmap_.find(GetRef<Expr>(op));
    CHECK(it != tmap_.end());
    Type checked_type = solver_->Resolve(it->second);
    CHECK(checked_type.as<IncompleteTypeNode>() == nullptr)
        << "Cannot resolve type of " << GetRef<Expr>(op)
        << " at " << op->span;
    Expr new_e = ExprMutator::VisitExpr_(op);
    if (!checked_type.same_as(new_e->checked_type_)) {
      // Copy on write optimization
      // If new_e is an old expression,
      // we make a copy mutating an existing reference.
      if (!new_e.node_.unique()) {
        new_e = Expr(make_node<T>(*new_e.as<T>()));
      }
      new_e->checked_type_ = checked_type;
    }
    return new_e;
  }

  Type VisitType(const Type &t) final {
    return solver_->Resolve(t);
  }

 private:
  const std::unordered_map<Expr, Type, NodeHash, NodeEqual>& tmap_;
  TypeSolver* solver_;
};


Expr TypeInferencer::Infer(Expr expr) {
  // step 0: populate the constraints
  GetType(expr);
  // step 1: solve the constraints
  solver_.Solve();
  // step 2: attach resolved types to checked_type field
  return Resolver(type_map_, &solver_).VisitExpr(expr);
}

Expr InferType(const Environment& env, const Expr& expr) {
  return TypeInferencer(env).Infer(expr);
}

Expr InferType(const Environment& env,
               const GlobalVar& var,
               const Function& func) {
  Function func_copy = Function(make_node<FunctionNode>(*func.operator->()));
  func_copy->checked_type_ = func_copy->fn_type();
  env->functions.Set(var, func_copy);
  Expr func_ret = TypeInferencer(env).Infer(func_copy);
  auto map_node = env->functions.CopyOnWrite();
  map_node->data.erase(var.node_);
  return func_ret;
}

TVM_REGISTER_API("relay._ir_pass.infer_type")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = InferType(args[0], args[1]);
  });

}  // namespace relay
}  // namespace tvm
