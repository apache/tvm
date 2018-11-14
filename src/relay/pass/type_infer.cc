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
#include "../ir/type_functor.h"

namespace tvm {
namespace relay {

// Necessary deferred relation for TupleGetItem
struct TupleGetItemAttrs : public tvm::AttrsNode<TupleGetItemAttrs> {
  int index;

  TVM_DECLARE_ATTRS(TupleGetItemAttrs, "relay.attrs.TupleGetItemAttrs") {
    TVM_ATTR_FIELD(index);
  }
};

bool TupleGetItemRel(const Array<Type>& types,
                     int num_inputs,
                     const Attrs& attrs,
                     const TypeReporter& reporter) {
  CHECK_EQ(types.size(), 2);
  if (types[0].as<IncompleteTypeNode>()) return false;
  const auto* data = types[0].as<TupleTypeNode>();
  CHECK(data != nullptr)
      << "TupleGetItem expect input type to be TupleType "
      << " get " << types[0] << " instead";
  const auto* param = attrs.as<TupleGetItemAttrs>();
  CHECK(param != nullptr);
  CHECK_GE(param->index, 0);
  CHECK_LT(param->index,  data->fields.size());
  reporter->Assign(types[1], data->fields[param->index]);
  return true;
}

bool MakeTupleRel(const Array<Type>& types,
                  int num_inputs,
                  const Attrs& attrs,
                  const TypeReporter& reporter) {
  CHECK_EQ(static_cast<size_t>(num_inputs + 1), types.size());
  for (int i = 0; i < num_inputs; ++i) {
    if (types[i].as<IncompleteTypeNode>()) return false;
  }
  Array<Type> fields;
  for (int i = 0; i < num_inputs; ++i) {
    fields.push_back(types[i]);
  }
  reporter->Assign(types[num_inputs], TupleTypeNode::make(fields));
  return true;
}

TVM_REGISTER_NODE_TYPE(TupleGetItemAttrs);
TVM_REGISTER_API("tvm.relay.type_relation.TupleGetItem")
.set_body_typed<bool(const Array<Type>&, int, const Attrs&, const TypeReporter&)>(
    TupleGetItemRel);

TVM_REGISTER_API("tvm.relay.type_relation.MakeTuple")
.set_body_typed<bool(const Array<Type>&, int, const Attrs&, const TypeReporter&)>(
    MakeTupleRel);

struct ResolvedTypeInfo {
  explicit ResolvedTypeInfo(Type checked_type, Array<Type> type_args)
      : checked_type(checked_type), type_args(type_args) {}
  ResolvedTypeInfo() {}

  Type checked_type;
  // Only allocated when the expression is a call.

  Array<Type> type_args = Array<Type>(NodePtr<Node>(nullptr));
};

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
  TypeInferencer() {
  }
  explicit TypeInferencer(Module mod)
      : mod_(mod) {
  }

  // inference the type of expr.
  Expr Infer(Expr expr);

 private:
  // type resolver that maps back to type
  class Resolver;
  // internal environment
  Module mod_;
  // map from expression to checked type
  // type inferencer will populate it up
  std::unordered_map<Expr, ResolvedTypeInfo, NodeHash, NodeEqual> type_map_;

  // The solver used by the inferencer.
  TypeSolver solver_;
  // relation function
  TypeRelationFn tuple_getitem_rel_;
  TypeRelationFn make_tuple_rel_;
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
    if (it != type_map_.end() && it->second.checked_type.defined()) {
      return it->second.checked_type;
    }
    Type ret = this->VisitExpr(expr);
    ResolvedTypeInfo& rti = type_map_[expr];
    rti.checked_type = ret;
    return ret;
  }

  // Visitor logics
  Type VisitExpr_(const VarNode* op) final {
    if (op->type_annotation.defined()) {
      return op->type_annotation;
    } else {
      return IncompleteTypeNode::make(TypeVarNode::kType);
    }
  }

  Type VisitExpr_(const GlobalVarNode* op) final {
    GlobalVar var = GetRef<GlobalVar>(op);
    CHECK(mod_.defined())
        << "Cannot do type inference without a global variable";
    Expr e = mod_->Lookup(var);
    return e->checked_type();
  }

  Type VisitExpr_(const ConstantNode* op) final {
    return op->tensor_type();
  }

  Type VisitExpr_(const TupleNode* op) final {
    if (!make_tuple_rel_.defined())  {
      make_tuple_rel_ = TypeRelationFn(
          EnvFunc::Get("tvm.relay.type_relation.MakeTuple").node_);
    }
    Array<Type> types;
    for (Expr field : op->fields) {
      types.push_back(GetType(field));
    }
    Type rtype = IncompleteTypeNode::make(TypeVarNode::Kind::kType);
    types.push_back(rtype);
    solver_.AddConstraint(TypeRelationNode::make(
        make_tuple_rel_, types, op->fields.size(), Attrs()));
    return rtype;
  }

  Type VisitExpr_(const TupleGetItemNode* op) final {
    if (!tuple_getitem_rel_.defined())  {
      tuple_getitem_rel_ = TypeRelationFn(
          EnvFunc::Get("tvm.relay.type_relation.TupleGetItem").node_);
    }
    Type tuple_type = GetType(op->tuple);
    Type rtype = IncompleteTypeNode::make(TypeVarNode::Kind::kType);
    auto attrs = make_node<TupleGetItemAttrs>();
    attrs->index = op->index;
    solver_.AddConstraint(TypeRelationNode::make(
        tuple_getitem_rel_, {tuple_type, rtype}, 1, Attrs(attrs)));
    return rtype;
  }

  Type VisitExpr_(const OpNode* op) final {
    return op->op_type;
  }

  Type VisitExpr_(const LetNode* op) final {
    Type vtype = GetType(op->value);
    if (op->var->type_annotation.defined()) {
      vtype = Unify(vtype, op->var->type_annotation, op->span);
    }
    CHECK(!type_map_.count(op->var));
    // NOTE: no scoping is necessary because var are unique in program
    type_map_[op->var].checked_type = vtype;
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
    Type rtype = IncompleteTypeNode::make(TypeVarNode::Kind::kType);
    arg_types.push_back(rtype);
    // we can do simple replacement here
    solver_.AddConstraint(TypeRelationNode::make(
        rel->func, arg_types, arg_types.size() - 1, attrs));
    return rtype;
  }

  // instantiate the function type with fresh
  FuncType Instantiate(const FuncTypeNode* fn_ty, Array<Type>* ty_args) {
    tvm::Map<TypeVar, Type> subst_map;

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
      ret_type = IncompleteTypeNode::make(TypeVarNode::Kind::kType);
    }

    Type inst_ty = FuncTypeNode::make(fn_ty->arg_types,
                                      ret_type, {},
                                      fn_ty->type_constraints);
    inst_ty = Bind(inst_ty, subst_map);
    return Downcast<FuncType>(inst_ty);
  }

  void AddTypeArgs(const Expr& expr, Array<Type> type_args) {
    auto type_info = type_map_.find(expr);
    if (type_info == type_map_.end()) {
      type_map_.insert({expr, ResolvedTypeInfo(Type(), type_args)});
    } else {
      CHECK(!type_info->second.type_args.defined());
      type_info->second.type_args = type_args;
    }
  }

  // Handle general call node.
  Type GeneralCall(const CallNode* call, Array<Type> arg_types) {
    Type ftype = GetType(call->op);
    auto* fn_ty_node = ftype.as<FuncTypeNode>();

    CHECK(fn_ty_node != nullptr)
        << "only expressions with function types can be called, found "
        << ftype << " at " << call->span;

    Array<Type> type_args;
    FuncType fn_ty = Instantiate(fn_ty_node, &type_args);

    AddTypeArgs(GetRef<Call>(call), type_args);

    size_t type_arity = fn_ty->arg_types.size();
    size_t number_of_args = arg_types.size();

    if (type_arity != number_of_args) {
      if (type_arity < number_of_args) {
        LOG(FATAL) << "the function is provided too many arguments " << call->span;
      } else {
        LOG(FATAL) << "the function is provided too few arguments" << call->span;
      }
    }

    for (size_t i = 0; i < fn_ty->arg_types.size(); i++) {
      this->Unify(fn_ty->arg_types[i], arg_types[i], call->args[i]->span);
    }

    for (auto cs : fn_ty->type_constraints) {
      if (auto tr = cs.as<TypeRelationNode>()) {
        solver_.AddConstraint(
          TypeRelationNode::make(tr->func, tr->args, tr->num_inputs, call->attrs));
      } else {
        solver_.AddConstraint(cs);
      }
    }

    return fn_ty->ret_type;
  }

  Type VisitExpr_(const CallNode* call) final {
    Array<Type> arg_types;
    for (Expr arg : call->args) {
      arg_types.push_back(GetType(arg));
    }

    if (const OpNode* opnode = call->op.as<OpNode>()) {
      Type rtype = PrimitiveCall(opnode->op_type.as<FuncTypeNode>(),
                                 arg_types,
                                 call->attrs);
      if (rtype.defined()) {
        AddTypeArgs(GetRef<Call>(call), arg_types);
        return rtype;
      }
    }

    return GeneralCall(call, arg_types);
  }

  Type VisitExpr_(const FunctionNode* f) final {
    for (auto param : f->params) {
      GetType(param);
    }
    Type rtype = GetType(f->body);
    // Run solver using the currently known information
    solver_.Solve();
    // Trying to resolve
    Array<Type> arg_types;
    for (size_t i = 0; i < f->params.size(); ++i) {
      Type atype = solver_.Resolve(GetType(f->params[i]));
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
  Resolver(const std::unordered_map<Expr, ResolvedTypeInfo, NodeHash, NodeEqual>& tmap,
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
    return GetRef<GlobalVar>(op);
  }

  Expr VisitExpr_(const OpNode* op) final {
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const TupleNode* op) final {
    return AttachCheckedType(op);
  }

  Expr VisitExpr_(const TupleGetItemNode* op) final {
    return AttachCheckedType(op);
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
    Type checked_type = solver_->Resolve(it->second.checked_type);
    CHECK(checked_type.as<IncompleteTypeNode>() == nullptr)
        << "Cannot resolve type of " << GetRef<Expr>(op)
        << " at " << op->span;

    Expr new_e = ExprMutator::VisitExpr_(op);
    // new_call and new_var's code is only going to be valid for VarNode/CallNode.
    // Compiler optimization will likely fold these away for other nodes.
    CallNode* new_call =(
        std::is_base_of<CallNode, T>::value ?
        static_cast<CallNode*>(new_e.node_.get()) : nullptr);
    VarNode* new_var =(
        std::is_base_of<VarNode, T>::value ?
        static_cast<VarNode*>(new_e.node_.get()) : nullptr);
    FunctionNode* new_fn =(
        std::is_base_of<FunctionNode, T>::value ?
        static_cast<FunctionNode*>(new_e.node_.get()) : nullptr);

    // check if we need update the new_e
    bool need_update_type = !checked_type.same_as(new_e->checked_type_);
    bool need_update_call = (
        std::is_base_of<CallNode, T>::value &&
        it->second.type_args.defined() &&
        !it->second.type_args.same_as(new_call->type_args));
    bool need_update_var = (
        std::is_base_of<VarNode, T>::value &&
        update_missing_type_annotation_ &&
        !new_var->type_annotation.defined());

    bool need_update_fn = (
        std::is_base_of<FunctionNode, T>::value &&
        update_missing_type_annotation_ &&
        !new_fn->ret_type.defined());

    if (!need_update_type &&
        !need_update_var &&
        !need_update_call &&
        !need_update_fn) {
      return new_e;
    }

    if (!new_e.node_.unique()) {
      // Copy on write optimization
      // If new_e is an old expression,
      // we make a copy mutating an existing reference.
      new_e = Expr(make_node<T>(*new_e.as<T>()));
      new_call = (
          std::is_base_of<CallNode, T>::value ?
          static_cast<CallNode*>(new_e.node_.get()) : nullptr);
      new_var = (
          std::is_base_of<VarNode, T>::value ?
          static_cast<VarNode*>(new_e.node_.get()) : nullptr);
      new_fn = (
          std::is_base_of<FunctionNode, T>::value ?
          static_cast<FunctionNode*>(new_e.node_.get()) : nullptr);
    }

    // attach the information.
    if (need_update_type) {
      new_e->checked_type_ = checked_type;
    }

    if (need_update_call) {
      new_call->type_args = it->second.type_args;
      for (size_t i = 0; i < new_call->type_args.size(); i++) {
        new_call->type_args.Set(i, solver_->Resolve(new_call->type_args[i]));
      }
    }
    if (need_update_var) {
      new_var->type_annotation = checked_type;
    }
    if (need_update_fn) {
      auto* fn_type = checked_type.as<FuncTypeNode>();
      CHECK(fn_type != nullptr);
      new_fn->ret_type = fn_type->ret_type;
    }
    return new_e;
  }

  Type VisitType(const Type &t) final {
    return solver_->Resolve(t);
  }

 private:
  const std::unordered_map<Expr, ResolvedTypeInfo, NodeHash, NodeEqual>& tmap_;
  TypeSolver* solver_;
  // whether attach the checked type as type_annotation
  // if original type anntation is missing.
  bool update_missing_type_annotation_{true};
};


Expr TypeInferencer::Infer(Expr expr) {
  // Step 0: Populate the constraints.
  GetType(expr);
  // Step 1: Solve the constraints.
  solver_.Solve();
  // Step 2: Attach resolved types to checked_type field.
  auto resolved_expr = Resolver(type_map_, &solver_).VisitExpr(expr);
  CHECK(WellFormed(resolved_expr));
  return resolved_expr;
}


Expr InferType(const Expr& expr, const Module& mod) {
  auto e = TypeInferencer(mod).Infer(expr);
  CHECK(WellFormed(e));
  return e;
}

Function InferType(const Function& func,
                   const Module& mod,
                   const GlobalVar& var) {
  Function func_copy = Function(make_node<FunctionNode>(*func.operator->()));
  func_copy->checked_type_ = func_copy->func_type_annotation();
  mod->functions.Set(var, func_copy);
  Expr func_ret = TypeInferencer(mod).Infer(func_copy);
  auto map_node = mod->functions.CopyOnWrite();
  map_node->data.erase(var.node_);
  CHECK(WellFormed(func_ret));
  return Downcast<Function>(func_ret);
}

TVM_REGISTER_API("relay._ir_pass.infer_type")
.set_body([](TVMArgs args, TVMRetValue* ret) {
    *ret = InferType(args[0], args[1]);
  });
}  // namespace relay
}  // namespace tvm
