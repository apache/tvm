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
 * \file type_infer.cc
 * \brief Relay type inference and checking.
 *
 * This file implements one of the most important passes to the
 * Relay IR. In order to do many transformations and generate the
 * most efficient code we need to obtain type information for the
 * IR.
 *
 * Similar to previous computation graph based IRs, the Relay IR leaves
 * type information implicit and computes types by performing program
 * analysis.
 *
 * Given an expression `e` this pass infers a type `t` for
 * the expression as well as simultaneously checking the property `e : t`
 * (i.e., we can show e has type t).
 *
 * If we can not infer a type or there is a conflicting
 * constraint it will emit errors.
 */

#include <tvm/ir/transform.h>
#include <tvm/ir/type_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relay/transform.h>

#include "../analysis/type_solver.h"
#include "pass_utils.h"

namespace tvm {
namespace relay {

// Necessary deferred relation for TupleGetItem
struct TupleGetItemAttrs : public tvm::AttrsNode<TupleGetItemAttrs> {
  int index;

  TVM_DECLARE_ATTRS(TupleGetItemAttrs, "relay.attrs.TupleGetItemAttrs") { TVM_ATTR_FIELD(index); }
};

bool TupleGetItemRel(const Array<Type>& types, int num_inputs, const Attrs& attrs,
                     const TypeReporter& reporter) {
  ICHECK_EQ(types.size(), 2);
  if (types[0].as<IncompleteTypeNode>()) return false;
  const auto* data = types[0].as<TupleTypeNode>();
  ICHECK(data != nullptr) << "TupleGetItem expect input type to be TupleType "
                          << " get " << types[0] << " instead";
  const auto* param = attrs.as<TupleGetItemAttrs>();
  ICHECK(param != nullptr);
  ICHECK_GE(param->index, 0);
  ICHECK_LT(param->index, data->fields.size());
  reporter->Assign(types[1], data->fields[param->index]);
  return true;
}

TVM_REGISTER_NODE_TYPE(TupleGetItemAttrs);
TVM_REGISTER_GLOBAL("tvm.relay.type_relation.TupleGetItem").set_body_typed(TupleGetItemRel);

struct ResolvedTypeInfo {
  explicit ResolvedTypeInfo(Type checked_type, Array<Type> type_args)
      : checked_type(checked_type), type_args(type_args) {}
  ResolvedTypeInfo() {}

  Type checked_type;
  // Only allocated when the expression is a call.

  Array<Type> type_args = Array<Type>(ObjectPtr<Object>(nullptr));
};

//
// The inference algorithm can roughly be divided into three stages:
// - Populate the constraints by visiting the expression (TypeInferencer.GetType)
//   - solver.AddConstraint and solver.Unify are called to populate the necessary constraints
// - Solve the constraints (solver_.Solve)
// - Recreate expression with the resolved checked_type (Resolver.VisitExpr)
//
class TypeInferencer : private ExprFunctor<Type(const Expr&)>,
                       private PatternFunctor<void(const Pattern&, const Type&)> {
 public:
  // constructors

  explicit TypeInferencer(IRModule mod, DiagnosticContext diag_ctx)
      : mod_(mod), diag_ctx(diag_ctx), solver_(GlobalVar(), diag_ctx) {
    ICHECK(mod.defined()) << "Module must not be null in the type inferencer.";
  }

  // Infer the types inside of a function.
  Expr Infer(GlobalVar var, Function expr);

 private:
  // type resolver that maps back to type
  class Resolver;
  // internal environment
  IRModule mod_;

  // The current function being type checked.
  GlobalVar current_func_;

  /*! \brief The diagnostic context. */
  DiagnosticContext diag_ctx;

  // map from expression to checked type
  // type inferencer will populate it up
  std::unordered_map<Expr, ResolvedTypeInfo, ObjectPtrHash, ObjectPtrEqual> type_map_;

  // The solver used by the inferencer.
  TypeSolver solver_;
  // relation function
  TypeRelationFn tuple_getitem_rel_;
  TypeRelationFn make_tuple_rel_;

  /*! \brief Internal map used for memoization. */
  std::unordered_map<Expr, Type, ObjectPtrHash, ObjectPtrEqual> memo_;

  void VisitLeaf(const Expr& expr) {
    if (!memo_.count(expr)) {
      Type ret = this->DispatchVisitExpr(expr);
      memo_[expr] = ret;
    }
  }

  bool CheckVisited(const Expr& expr) {
    if (memo_.count(expr)) {
      return true;
    } else {
      return false;
    }
  }

  Type DispatchVisitExpr(const Expr& expr) { return ExprFunctor::VisitExpr(expr); }

  Type VisitExpr(const Expr& expr) final {
    auto fcheck_visited = [this](const Expr& expr) { return this->CheckVisited(expr); };
    auto fvisit_leaf = [this](const Expr& expr) { return this->VisitLeaf(expr); };
    if (memo_.count(expr)) {
      return memo_[expr];
    } else {
      ExpandDataflow(expr, fcheck_visited, fvisit_leaf);
      return memo_[expr];
    }
  }

  // Perform unification on two types and report the error at the expression
  // or the span of the expression.
  Type Unify(const Type& t1, const Type& t2, const Span& span, bool assign_lhs = true,
             bool assign_rhs = true) {
    try {
      return solver_.Unify(t1, t2, span, assign_lhs, assign_rhs);
    } catch (const Error& e) {
      this->EmitFatal(Diagnostic::Error(span)
                      << "Error unifying `" << t1 << "` and `" << t2 << "`: " << e.what());
      return Type();
    }
  }

  // Lazily get type for expr
  // expression, we will populate it now, and return the result.
  Type GetType(const Expr& expr) {
    auto it = type_map_.find(expr);
    if (it != type_map_.end() && it->second.checked_type.defined()) {
      return it->second.checked_type;
    }
    Type ret = this->VisitExpr(expr);
    ICHECK(ret.defined()) << "expression:" << std::endl << PrettyPrint(expr);
    KindCheck(ret, mod_, this->diag_ctx);
    ResolvedTypeInfo& rti = type_map_[expr];
    rti.checked_type = ret;
    return ret;
  }

  void EmitFatal(const Diagnostic& diag) { this->diag_ctx.EmitFatal(diag); }

  // Visitor Logic
  Type VisitExpr_(const VarNode* op) final {
    if (op->type_annotation.defined()) {
      return op->type_annotation;
    } else {
      return IncompleteType(Kind::kType);
    }
  }

  Type VisitExpr_(const GlobalVarNode* op) final {
    GlobalVar var = GetRef<GlobalVar>(op);
    if (!mod_.defined()) {
      this->EmitFatal(Diagnostic::Error(op->span) << "Cannot do type inference on global variables "
                                                  << "without a module");
    }
    if (mod_->ContainGlobalVar(var->name_hint)) {
      BaseFunc func = mod_->Lookup(var->name_hint);

      if (const auto* function_node = func.as<FunctionNode>()) {
        VLOG(1) << "global var '" << op->name_hint << "' bound to Function";
        return function_node->checked_type();
      } else {
        VLOG(1) << "global var '" << op->name_hint << "' bound to PrimFunc";
        return op->checked_type_;
      }
    } else {
      // TODO(mbs): extern function cleanup
      // Assume the function is extern thus no longer in the IRModule.
      VLOG(1) << "global var '" << op->name_hint << "' not in module";
      return op->checked_type_;
    }
  }

  Type VisitExpr_(const ConstantNode* op) final { return op->tensor_type(); }

  Type VisitExpr_(const TupleNode* op) final {
    Array<Type> types;
    for (Expr field : op->fields) {
      types.push_back(GetType(field));
    }
    return TupleType(types);
  }

  Type VisitExpr_(const TupleGetItemNode* op) final {
    if (!tuple_getitem_rel_.defined()) {
      tuple_getitem_rel_ =
          Downcast<TypeRelationFn>(EnvFunc::Get("tvm.relay.type_relation.TupleGetItem"));
    }
    Type tuple_type = GetType(op->tuple);
    Type rtype = IncompleteType(Kind::kType);
    auto attrs = make_object<TupleGetItemAttrs>();
    attrs->index = op->index;
    solver_.AddConstraint(TypeRelation(tuple_getitem_rel_, {tuple_type, rtype}, 1, Attrs(attrs)),
                          op->span);
    return rtype;
  }

  void VisitPattern_(const PatternConstructorNode* con, const Type& t) {
    ICHECK(mod_.defined()) << "Cannot do type inference without a environment:"
                           << con->constructor->name_hint;
    TypeData td = mod_->type_definitions.at(con->constructor->belong_to);
    auto pc = GetRef<PatternConstructor>(con);

    // we can expect a certain number of arguments
    Array<Type> unknown_args;
    for (size_t i = 0; i < td->type_vars.size(); i++) {
      unknown_args.push_back(IncompleteType(Kind::kType));
    }

    Type expected = TypeCall(con->constructor->belong_to, unknown_args);
    Type unified = Unify(t, expected, pc->span);

    auto* tc = unified.as<TypeCallNode>();
    if (!tc) {
      this->EmitFatal(Diagnostic::Error(pc->span) << "Expected a type call, got " << unified);
    }

    if (td->header != tc->func) {
      this->EmitFatal(Diagnostic::Error(pc->span) << "ADT headers must match, but we have "
                                                  << td->header << " and " << tc->func);
    }

    if (td->type_vars.size() != tc->args.size()) {
      this->EmitFatal(Diagnostic::Error(pc->span)
                      << "The number of type args must match"
                      << "the number of type vars in the type data: " << td->type_vars.size()
                      << " != " << tc->args.size());
    }
    std::unordered_map<TypeVar, Type, ObjectPtrHash, ObjectPtrEqual> type_var_map_;
    for (size_t i = 0; i < td->type_vars.size(); ++i) {
      type_var_map_[td->type_vars[i]] = tc->args[i];
    }

    if (con->constructor->inputs.size() != con->patterns.size()) {
      this->EmitFatal(Diagnostic::Error(pc->span) << "Not enough inputs for the constructor; "
                                                  << "expected " << con->constructor->inputs.size()
                                                  << ", got " << con->patterns.size());
    }

    for (size_t i = 0; i < con->constructor->inputs.size(); ++i) {
      VisitPattern(con->patterns[i], Bind(con->constructor->inputs[i], type_var_map_));
    }
  }

  void VisitPattern_(const PatternTupleNode* tup, const Type& t) {
    auto pt = GetRef<PatternTuple>(tup);

    // we can expect a certain number of arguments
    Array<Type> unknown_args;
    for (size_t i = 0; i < tup->patterns.size(); i++) {
      unknown_args.push_back(IncompleteType(Kind::kType));
    }

    Type expected = TupleType(unknown_args);
    Type unified = Unify(t, expected, tup->span);

    auto* tt = unified.as<TupleTypeNode>();
    if (!tt) {
      this->EmitFatal(Diagnostic::Error(pt->span) << "Expected a tuple type, got " << unified);
    }
    ICHECK(tup->patterns.size() == tt->fields.size()) << "not enough pattern";
    for (size_t i = 0; i < tup->patterns.size(); ++i) {
      VisitPattern(tup->patterns[i], tt->fields[i]);
    }
  }

  void VisitPattern_(const PatternVarNode* pv, const Type& t) {
    Type vt = GetType(pv->var);
    Unify(vt, t, pv->span);
  }

  void VisitPattern_(const PatternWildcardNode* wc, const Type& t) {}

  Type VisitExpr_(const MatchNode* op) final {
    Type dtype = GetType(op->data);
    for (const auto& c : op->clauses) {
      VisitPattern(c->lhs, dtype);
    }
    Type rtype = IncompleteType(Kind::kType);
    for (const auto& c : op->clauses) {
      rtype = this->Unify(rtype, GetType(c->rhs), op->span);
    }

    if (op->complete) {
      // check completness
      Match match = GetRef<Match>(op);
      Array<Pattern> unmatched_cases = UnmatchedCases(match, this->mod_);
      if (unmatched_cases.size() != 0) {
        ErrorBuilder ss;
        auto err = Diagnostic::Error(match->span);
        err << "match expression does not handle the following cases: ";
        int i = 0;
        for (auto cs : unmatched_cases) {
          err << "case " << i++ << ": \n" << PrettyPrint(cs);
        }
        this->EmitFatal(err);
      }
    }

    return rtype;
  }

  Type VisitExpr_(const OpNode* op) final { return op->op_type; }

  Type VisitExpr_(const LetNode* let) final {
    auto pre_visit = [this](const LetNode* op) {
      // if the definition is a function literal, permit recursion
      bool is_functional_literal = op->value.as<FunctionNode>() != nullptr;
      Type let_type = IncompleteType(Kind::kType);

      if (is_functional_literal) {
        let_type = this->GetType(op->var);
        this->type_map_[op->var].checked_type = let_type;
      }

      if (op->var->type_annotation.defined()) {
        let_type = this->Unify(let_type, op->var->type_annotation, op->span);
      }

      Type vtype = this->GetType(op->value);
      let_type = this->Unify(let_type, vtype, op->span);

      ICHECK(is_functional_literal || !this->type_map_.count(op->var));
      // NOTE: no scoping is necessary because var are unique in program
      this->type_map_[op->var].checked_type = let_type;
    };
    auto post_visit = [this](const LetNode* op) {
      Expr expr = GetRef<Expr>(op);
      this->memo_[expr] = this->GetType(op->body);
      this->type_map_[expr].checked_type = this->memo_[expr];
    };
    ExpandANormalForm(let, pre_visit, post_visit);
    return memo_[GetRef<Expr>(let)];
  }

  Type VisitExpr_(const IfNode* ite) final {
    // Ensure the type of the guard is of Tensor[Bool, ()],
    // that is a rank-0 boolean tensor.
    Type cond_type = this->GetType(ite->cond);
    this->Unify(cond_type, TensorType::Scalar(tvm::DataType::Bool()), ite->cond->span);
    Type checked_true = this->GetType(ite->true_branch);
    Type checked_false = this->GetType(ite->false_branch);
    return this->Unify(checked_true, checked_false, ite->span);
  }

  // This code is special-cased for primitive operators,
  // which are registered in the style defined in src/relay/op/*.
  //
  // The result will be the return type of the operator.
  Type PrimitiveCall(const FuncTypeNode* op, Array<Type> arg_types, const Attrs& attrs,
                     const Span& span) {
    if (op->type_params.size() != arg_types.size() + 1) return Type();
    if (op->type_constraints.size() != 1) return Type();
    const TypeRelationNode* rel = op->type_constraints[0].as<TypeRelationNode>();
    if (rel == nullptr) return Type();
    // validate if the type parameter matches up
    for (size_t i = 0; i < op->type_params.size(); ++i) {
      if (!op->type_params[i].same_as(rel->args[i])) return Type();
    }
    Type rtype = IncompleteType(Kind::kType);
    arg_types.push_back(rtype);
    // we can do simple replacement here
    solver_.AddConstraint(TypeRelation(rel->func, arg_types, arg_types.size() - 1, attrs), span);
    return rtype;
  }

  // substitute the type args in the function type
  FuncType InstantiateFuncType(const FuncTypeNode* fn_ty, const Array<Type>& ty_args) {
    tvm::Map<TypeVar, Type> subst_map;

    // Build a subsitituion map up from the function type and type arguments.
    // Eventually allow the type vars to be passed in.
    ICHECK(fn_ty->type_params.size() == ty_args.size())
        << "number of type parameters does not match expected";
    for (size_t i = 0; i < ty_args.size(); ++i) {
      subst_map.Set(fn_ty->type_params[i], ty_args[i]);
    }

    Type ret_type = fn_ty->ret_type;

    // If the function type is incomplete, place a new IncompleteType
    // This relax the fn_ty to inputs -> Any
    // The type checking can still pass when there are additional constraints on the type
    // This is a temporary work around to check recursive functions whose
    // return type is not yet known.
    if (!ret_type.defined()) {
      ret_type = IncompleteType(Kind::kType);
    }

    Type inst_ty = FuncType(fn_ty->arg_types, ret_type, {}, fn_ty->type_constraints);
    inst_ty = Bind(inst_ty, subst_map);
    return Downcast<FuncType>(inst_ty);
  }

  // instantiates starting from incompletes
  FuncType InstantiateFuncType(const FuncTypeNode* fn_ty) {
    if (fn_ty->type_params.size() == 0) {
      return GetRef<FuncType>(fn_ty);
    }

    Array<Type> type_args;
    for (size_t i = 0; i < fn_ty->type_params.size(); i++) {
      type_args.push_back(IncompleteType(Kind::kType));
    }
    return InstantiateFuncType(fn_ty, type_args);
  }

  void AddTypeArgs(const Expr& expr, Array<Type> type_args) {
    auto type_info = type_map_.find(expr);
    if (type_info == type_map_.end()) {
      type_map_.insert({expr, ResolvedTypeInfo(Type(), type_args)});
    } else {
      ICHECK(!type_info->second.type_args.defined());
      type_info->second.type_args = type_args;
    }
  }

  // Handle general call node.
  Type GeneralCall(const CallNode* call, Array<Type> arg_types) {
    Type ftype = GetType(call->op);
    auto* fn_ty_node = ftype.as<FuncTypeNode>();
    auto* inc_ty_node = ftype.as<IncompleteTypeNode>();

    if (fn_ty_node == nullptr && inc_ty_node == nullptr) {
      this->EmitFatal(Diagnostic::Error(call->span)
                      << "only expressions with function types can be called, found " << ftype);
    }

    // incomplete type => it must be a function taking the arg types
    // with an unknown return type
    if (inc_ty_node != nullptr) {
      Type ret_type = IncompleteType(Kind::kType);
      Type func_type = FuncType(arg_types, ret_type, {}, {});
      Type unified = this->Unify(ftype, func_type, call->op->span);
      fn_ty_node = unified.as<FuncTypeNode>();
    }

    Array<Type> type_args = call->type_args;
    if (type_args.size() > fn_ty_node->type_params.size()) {
      this->EmitFatal(Diagnostic::Error(call->span)
                      << "Incorrect number of type args in " << call->span << ": "
                      << "Expected " << fn_ty_node->type_params.size() << " but got "
                      << type_args.size() << " for call:\n"
                      << PrettyPrint(GetRef<Call>(call)));
    }
    for (size_t i = type_args.size(); i < fn_ty_node->type_params.size(); i++) {
      type_args.push_back(IncompleteType(TypeKind::kType));
    }

    FuncType fn_ty = InstantiateFuncType(fn_ty_node, type_args);

    AddTypeArgs(GetRef<Call>(call), type_args);

    size_t type_arity = fn_ty->arg_types.size();
    size_t number_of_args = arg_types.size();
    bool is_variable = false;

    if (const OpNode* opnode = call->op.as<OpNode>()) {
      if (opnode->num_inputs == -1) {
        is_variable = true;
      }
    }

    if ((type_arity < number_of_args) && !is_variable) {
      this->EmitFatal(Diagnostic::Error(call->span)
                      << "the function is provided too many arguments "
                      << "expected " << type_arity << ", found " << number_of_args);
    } else if (type_arity > number_of_args) {
      this->EmitFatal(Diagnostic::Error(call->span)
                      << "the function is provided too few arguments "
                      << "expected " << type_arity << ", found " << number_of_args);
    }

    Array<Type> unified_arg_types;
    if (!is_variable) {
      for (size_t i = 0; i < fn_ty->arg_types.size(); i++) {
        this->Unify(fn_ty->arg_types[i], arg_types[i], call->span, true, false);
      }
    } else {
      for (size_t i = 0; i < number_of_args; i++) {
        if (i < fn_ty->arg_types.size()) {
          unified_arg_types.push_back(
              this->Unify(fn_ty->arg_types[i], arg_types[i], call->span, false, false));
        } else {
          unified_arg_types.push_back(arg_types[i]);
        }
      }
      unified_arg_types.push_back(fn_ty->ret_type);
    }
    for (auto cs : fn_ty->type_constraints) {
      if (const auto* tr = cs.as<TypeRelationNode>()) {
        if (!is_variable) {
          solver_.AddConstraint(TypeRelation(tr->func, tr->args, tr->num_inputs, call->attrs),
                                call->span);
        } else {
          solver_.AddConstraint(
              TypeRelation(tr->func, unified_arg_types, number_of_args, call->attrs), call->span);
        }
      } else {
        solver_.AddConstraint(cs, call->span);
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
      Type rtype =
          PrimitiveCall(opnode->op_type.as<FuncTypeNode>(), arg_types, call->attrs, call->span);

      if (rtype.defined()) {
        AddTypeArgs(GetRef<Call>(call), arg_types);
        return rtype;
      }
    }

    solver_.Solve();
    return GeneralCall(call, arg_types);
  }

  Type VisitExpr_(const FunctionNode* f) final {
    solver_.Solve();
    Array<Type> arg_types;
    for (auto param : f->params) {
      arg_types.push_back(GetType(param));
    }
    Type rtype = GetType(f->body);
    if (auto* ft = rtype.as<FuncTypeNode>()) {
      rtype = InstantiateFuncType(ft);
    }
    if (f->ret_type.defined()) {
      rtype = this->Unify(f->ret_type, rtype, GetRef<Function>(f)->span);
    }
    ICHECK(rtype.defined());
    auto ret = FuncType(arg_types, rtype, f->type_params, {});
    return solver_.Resolve(ret);
  }

  Type VisitExpr_(const RefCreateNode* op) final { return RelayRefType(GetType(op->value)); }

  Type VisitExpr_(const RefReadNode* op) final {
    Type it = IncompleteType(Kind::kType);
    this->Unify(GetType(op->ref), RelayRefType(it), op->span);
    return it;
  }

  Type VisitExpr_(const RefWriteNode* op) final {
    Type it = IncompleteType(Kind::kType);
    this->Unify(GetType(op->ref), RelayRefType(it), op->span);
    this->Unify(GetType(op->value), it, op->span);
    return TupleType::Empty();
  }

  Type VisitExpr_(const ConstructorNode* c) final {
    ICHECK(mod_.defined()) << "Cannot do type inference without a environment:" << c->name_hint;
    TypeData td = mod_->LookupTypeDef(c->belong_to);
    std::vector<Type> types;
    for (const auto& t : td->type_vars) {
      types.push_back(t);
    }
    return FuncType(c->inputs, TypeCall(c->belong_to, types), td->type_vars, {});
  }

  void Solve() { solver_.Solve(); }
};

class TypeInferencer::Resolver : public MixedModeMutator, PatternMutator {
 public:
  Resolver(const std::unordered_map<Expr, ResolvedTypeInfo, ObjectPtrHash, ObjectPtrEqual>& tmap,
           TypeSolver* solver)
      : tmap_(tmap), solver_(solver) {}

  using MixedModeMutator::VisitExpr_;

  Expr VisitExpr_(const VarNode* op) final { return VisitVar(GetRef<Var>(op)); }

  Expr VisitExpr_(const ConstantNode* op) final { return AttachCheckedType(op); }

  Expr VisitExpr_(const GlobalVarNode* op) final { return GetRef<GlobalVar>(op); }

  Expr VisitExpr_(const OpNode* op) final { return ExprMutator::VisitExpr_(op); }

  Expr Rewrite_(const TupleNode* op, const Expr& post) final { return AttachCheckedType(op, post); }

  Expr Rewrite_(const TupleGetItemNode* op, const Expr& post) final {
    return AttachCheckedType(op, post);
  }

  Expr VisitExpr_(const FunctionNode* op) final { return AttachCheckedType(op); }

  Expr Rewrite_(const CallNode* op, const Expr& post) final { return AttachCheckedType(op, post); }

  Expr VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* op) {
      this->VisitExpr(op->var);
      this->VisitExpr(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      Expr expr = GetRef<Expr>(op);
      Var var = Downcast<Var>(this->VisitExpr(op->var));
      Expr value = this->VisitExpr(op->value);
      Expr body = this->VisitExpr(op->body);
      this->memo_[expr] = this->AttachCheckedType(op, Let(var, value, body));
    };
    ExpandANormalForm(op, pre_visit, post_visit);
    return memo_[GetRef<Expr>(op)];
  }

  Expr VisitExpr_(const IfNode* op) final { return AttachCheckedType(op); }

  Expr VisitExpr_(const RefCreateNode* op) final { return AttachCheckedType(op); }

  Expr VisitExpr_(const RefReadNode* op) final { return AttachCheckedType(op); }

  Expr VisitExpr_(const RefWriteNode* op) final { return AttachCheckedType(op); }

  Expr VisitExpr_(const ConstructorNode* op) final { return AttachCheckedType(op); }

  Expr VisitExpr_(const MatchNode* op) final { return AttachCheckedType(op); }

  Pattern VisitPattern(const Pattern& p) final { return PatternMutator::VisitPattern(p); }

  Var VisitVar(const Var& v) final {
    if (vmap_.count(v) == 0) {
      vmap_[v] = Downcast<Var>(AttachCheckedType(v.as<VarNode>()));
    }
    return vmap_.at(v);
  }

  // attach checked type to the mutated node.
  template <typename T>
  Expr AttachCheckedType(const T* op, const Expr& post = Expr()) {
    auto it = tmap_.find(GetRef<Expr>(op));
    ICHECK(it != tmap_.end());
    Type checked_type = solver_->Resolve(it->second.checked_type);

    if (checked_type.as<IncompleteTypeNode>() != nullptr) {
      this->solver_->Emit(
          Diagnostic::Error(op->span)
          << "The type inference pass was unable to infer a type for this expression.\n"
          << "This usually occurs when an operator call is under constrained in some way,"
          << " check other reported errors for hints of what may of happened.");
    }

    Expr new_e = post.defined() ? post : ExprMutator::VisitExpr_(op);
    // new_call and new_var's code is only going to be valid for VarNode/CallNode.
    // Compiler optimization will likely fold these away for other nodes.
    CallNode* new_call = (std::is_base_of<CallNode, T>::value
                              ? const_cast<CallNode*>(static_cast<const CallNode*>(new_e.get()))
                              : nullptr);
    VarNode* new_var = (std::is_base_of<VarNode, T>::value
                            ? const_cast<VarNode*>(static_cast<const VarNode*>(new_e.get()))
                            : nullptr);
    FunctionNode* new_fn =
        (std::is_base_of<FunctionNode, T>::value
             ? const_cast<FunctionNode*>(static_cast<const FunctionNode*>(new_e.get()))
             : nullptr);

    // check if we need update the new_e
    bool need_update_type = !checked_type.same_as(new_e->checked_type_);
    bool need_update_call =
        (std::is_base_of<CallNode, T>::value && it->second.type_args.defined() &&
         !it->second.type_args.same_as(new_call->type_args));
    bool need_update_var = (std::is_base_of<VarNode, T>::value && update_missing_type_annotation_ &&
                            !new_var->type_annotation.defined());

    bool need_update_fn = (std::is_base_of<FunctionNode, T>::value &&
                           update_missing_type_annotation_ && !new_fn->ret_type.defined());

    if (!need_update_type && !need_update_var && !need_update_call && !need_update_fn) {
      return new_e;
    }

    if (!new_e.unique()) {
      // Copy on write optimization
      // If new_e is an old expression,
      // we make a copy mutating an existing reference.
      ObjectPtr<ExprNode> ptr = make_object<T>(*new_e.as<T>());
      new_e = Expr(ptr);
      new_call =
          (std::is_base_of<CallNode, T>::value ? static_cast<CallNode*>(ptr.get()) : nullptr);
      new_var = (std::is_base_of<VarNode, T>::value ? static_cast<VarNode*>(ptr.get()) : nullptr);
      new_fn = (std::is_base_of<FunctionNode, T>::value ? static_cast<FunctionNode*>(ptr.get())
                                                        : nullptr);
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
      ICHECK(fn_type != nullptr);
      new_fn->ret_type = fn_type->ret_type;
    }
    return new_e;
  }

  Type VisitType(const Type& t) final { return solver_->Resolve(t); }

 private:
  std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual> vmap_;
  const std::unordered_map<Expr, ResolvedTypeInfo, ObjectPtrHash, ObjectPtrEqual>& tmap_;
  TypeSolver* solver_;
  // whether attach the checked type as type_annotation
  // if original type anntation is missing.
  bool update_missing_type_annotation_{true};
};

Expr TypeInferencer::Infer(GlobalVar var, Function function) {
  // Set the current function being type checked.
  this->current_func_ = var;

  // Step 1: Populate the constraints.
  GetType(function);

  // Step 2: Solve the constraints.
  Solve();

  // Step 3: Attach resolved types to checked_type field.
  auto resolved_expr = Resolver(type_map_, &solver_).VisitExpr(function);

  if (!WellFormed(resolved_expr, this->diag_ctx)) {
    this->diag_ctx.Emit(Diagnostic::Bug(function->span)
                        << "the type checked function is malformed, please report this");
  }

  return resolved_expr;
}

struct AllCheckTypePopulated : MixedModeVisitor {
  using MixedModeVisitor::VisitExpr_;
  void DispatchExprVisit(const Expr& e) {
    if (e.as<OpNode>()) {
      return;
    }
    if (e.as<GlobalVarNode>()) {
      return;
    }
    if (e.as<ConstructorNode>()) {
      return;
    }
    ICHECK(e->checked_type_.defined()) << "Expression: " << e;
    return ExprVisitor::VisitExpr(e);
  }
  void VisitExpr_(const LetNode* op) final {
    auto pre_visit = [this](const LetNode* op) {
      this->VisitExpr(op->var);
      this->VisitExpr(op->value);
    };
    auto post_visit = [this](const LetNode* op) {
      this->VisitExpr(op->body);
      this->visit_counter_[op] += 1;
    };
    ExpandANormalForm(op, pre_visit, post_visit);
  }
};

void EnsureCheckedType(const Expr& e) { AllCheckTypePopulated().VisitExpr(e); }

// TODO(@jroesch): Can we optimize this?
void AddGlobalTypes(IRModule mod) {
  std::vector<std::pair<GlobalVar, Function>> updates;
  for (const auto& it : mod->functions) {
    // Currently we don't type check TIR.
    // The inferencer will only check Relay functions
    // the future plan is to have a unified type checker
    // that works on TIR and Relay at the same time.
    if (auto* func_node = it.second.as<FunctionNode>()) {
      Function func = Function(make_object<FunctionNode>(*func_node));
      func->checked_type_ = func->func_type_annotation();
      updates.push_back({it.first, Downcast<Function>(func)});
    }
  }

  for (const auto& pair : updates) {
    mod->Add(pair.first, pair.second, true);
  }
}

/*!
 * \brief Returns a possibly much smaller subgraph whose inner nodes have the same type.
 *
 * Returns the largest sub-graph who's inner nodes need types and leaves are vars standing in
 * for already typed sub-expressions. This creates a graph whose inner nodes have the same
 * type as the original graph and when running type inference, we can avoid copying and
 * recursing through most of the expression graph when running type inference. Note, this assumes
 * that current populated type information is correct!
 *
 * ExprMutator is sufficient over MixedModemutator since we will not recurse much.
 */
class SameTypedSubgraphExtractor : public ExprMutator {
  Expr VisitExpr_(const VarNode* op) { return Var(op->vid, op->type_annotation, op->span); }
  Expr VisitExpr_(const ConstantNode* op) { return Constant(op->data, op->span); }
  Expr VisitExpr_(const GlobalVarNode* op) { return GlobalVar(op->name_hint); }
  Expr VisitExpr_(const OpNode* op) { return Op(GetRef<Op>(op)); }
  Expr VisitExpr_(const TupleNode* op) {
    return Tuple(GetAnalogousExpression(op->fields), op->span);
  }
  Expr VisitExpr_(const FunctionNode* op) {
    // Unfortunately our strategy of inserting variables as dummies would change the signature of
    // existing function nodes so we have to copy all used functions always :/
    return Function(op->params, op->body, op->ret_type, op->type_params, op->attrs, op->span);
  }
  Expr VisitExpr_(const CallNode* op) {
    return Call(op->op, GetAnalogousExpression(op->args), op->attrs, op->type_args, op->span);
  }
  Expr VisitExpr_(const LetNode* op) {
    return Let(op->var, GetAnalogousExpression(op->value), GetAnalogousExpression(op->body),
               op->span);
  }
  Expr VisitExpr_(const IfNode* op) {
    return If(GetAnalogousExpression(op->cond), GetAnalogousExpression(op->true_branch),
              GetAnalogousExpression(op->false_branch), op->span);
  }
  Expr VisitExpr_(const TupleGetItemNode* op) {
    return TupleGetItem(GetAnalogousExpression(op->tuple), op->index, op->span);
  }
  Expr VisitExpr_(const RefCreateNode* op) {
    return RefCreate(GetAnalogousExpression(op->value), op->span);
  }
  Expr VisitExpr_(const RefReadNode* op) {
    return RefRead(GetAnalogousExpression(op->ref), op->span);
  }
  Expr VisitExpr_(const RefWriteNode* op) {
    return RefWrite(GetAnalogousExpression(op->ref), GetAnalogousExpression(op->value), op->span);
  }
  Expr VisitExpr_(const ConstructorNode* op) {
    return Constructor(op->name_hint, op->inputs, op->belong_to);
  }
  Expr VisitExpr_(const MatchNode* op) {
    return Match(GetAnalogousExpression(op->data), op->clauses, op->complete, op->span);
  }

 private:
  Expr GetAnalogousExpression(const Expr& expr) {
    // Replace the expression with a potentially simpler expression of the same type
    if (expr->checked_type_.defined()) {
      // Since the expression already has a checked_type which we assume is correct we don't need
      // full type inference to enter it. So stub it out with a dummy var of the same type.
      return Var("dummy_var", expr->checked_type(), expr->span);
    }

    return VisitExpr(expr);
  }
  Array<Expr> GetAnalogousExpression(const Array<Expr>& fields) {
    Array<Expr> new_fields;
    for (Expr expr : fields) {
      new_fields.push_back(GetAnalogousExpression(expr));
    }
    return new_fields;
  }
};

namespace transform {

Type InferTypeLocal(const Expr& expr) {
  /*
  This type inference differs from InferType in that it uses existing type information
  to avoid recursing over much of the graph, and it only examines the type of the input
  node. This makes it faster if you need to run type inference iteratively throughout
  a pass for example.

  However, it assumes any existing populated type inference is correct! If some populated
  type inference is incorrect, an incorrect type may be returned or a type error will be
  raised. If you know not all populated type fields are correct with the current graph,
  you should use InferType() instead.
  */
  SameTypedSubgraphExtractor subgraph_extractor;
  Expr sub_graph = subgraph_extractor(expr);

  Type result_type;
  result_type = relay::InferType(sub_graph)->checked_type();

  expr->checked_type_ = result_type;
  return result_type;
}

TVM_REGISTER_GLOBAL("relay._transform.InferTypeLocal").set_body_typed([](const Expr& expr) {
  return InferTypeLocal(expr);
});

Pass InferType() {
  auto pass_info = PassInfo(0, "InferType", {}, /* trace */ false);
  return tvm::transform::CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        // Execute the pass function and return a new module.
        IRModule updated_mod = mod->ShallowCopy();

        pass_ctx->diag_ctx = DiagnosticContext::Default(updated_mod);

        // Add all the type annotations to the functions in the model.
        AddGlobalTypes(mod);

        std::vector<std::pair<GlobalVar, Function>> updates;
        for (const auto& it : updated_mod->functions) {
          // Currently we don't type check TIR.
          //
          // The inferencer will only check Relay functions.

          // In the future we plan a unified type checker
          // that works on TIR and Relay at the same time.
          if (auto func = it.second.as<Function>()) {
            // // If a function already has type information we can skip checking it.
            // if (func->checked_type_.defined()) {
            //   continue;
            // }

            // TODO(@jroesch): we should be able to move the type inferencer outside
            // of this function but it seems to be more stateful then I expect.
            auto inferencer = TypeInferencer(mod, pass_ctx->diag_ctx.value());
            auto updated_func = inferencer.Infer(it.first, func.value());

            pass_ctx->diag_ctx.value().Render();

            // After we are done checking write the global type back
            // into the global var.
            it.first->checked_type_ = updated_func->checked_type();

            if (!WellFormed(updated_func, pass_ctx->diag_ctx)) {
              LOG(FATAL) << "The type checked intermediate representation is malformed";
            }

            auto free_tvars = FreeTypeVars(updated_func, mod);
            ICHECK(free_tvars.size() == 0)
                << "Found unbound type variables in " << updated_func << ": " << free_tvars;
            EnsureCheckedType(updated_func);
            updates.push_back({it.first, Downcast<Function>(updated_func)});
          }
        }

        for (const auto& pair : updates) {
          updated_mod->Add(pair.first, pair.second, true);
        }

        return updated_mod;
      },
      0, "InferType", {});
}

TVM_REGISTER_GLOBAL("relay._transform.InferType").set_body_typed([]() { return InferType(); });

}  // namespace transform

}  // namespace relay
}  // namespace tvm
