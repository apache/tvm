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
#include <tvm/ir/type_functor.h>
#include <tvm/ir/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include "./pass_util.h"
#include "type_solver.h"

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
  CHECK_LT(param->index, data->fields.size());
  reporter->Assign(types[1], data->fields[param->index]);
  return true;
}

TVM_REGISTER_NODE_TYPE(TupleGetItemAttrs);
TVM_REGISTER_GLOBAL("tvm.relay.type_relation.TupleGetItem")
.set_body_typed(
    TupleGetItemRel);

struct ResolvedTypeInfo {
  explicit ResolvedTypeInfo(Type checked_type, Array<Type> type_args)
      : checked_type(checked_type), type_args(type_args) {}
  ResolvedTypeInfo() {}

  Type checked_type;
  // Only allocated when the expression is a call.

  Array<Type> type_args = Array<Type>(ObjectPtr<Object>(nullptr));
};

//
// The inference algorithm can roughly be devided into three stages:
// - Populate the constraints by visiting the expression (TypeInferencer.GetType)
//   - solver.AddConstraint and solver.Unify are called to populate the necessary constraints
// - Solve the constraints (solver_.Solve)
// - Recreate expression with the resolved checked_type (Resolver.VisitExpr)
//
class TypeInferencer : private ExprFunctor<Type(const Expr&)>,
                       private PatternFunctor<void(const Pattern&, const Type&)> {
 public:
  // constructors

  explicit TypeInferencer(IRModule mod, GlobalVar current_func)
      : mod_(mod), current_func_(current_func),
        err_reporter(), solver_(current_func, mod, &this->err_reporter) {
    CHECK(mod.defined()) << "internal error: Module must be set in the type inferencer";
  }

  // inference the type of expr.
  Expr Infer(Expr expr);

 private:
  // type resolver that maps back to type
  class Resolver;
  // internal environment
  IRModule mod_;

  // The current function being type checked.
  GlobalVar current_func_;

  // The error reporter.
  ErrorReporter err_reporter;

  // map from expression to checked type
  // type inferencer will populate it up
  std::unordered_map<Expr, ResolvedTypeInfo, ObjectHash, ObjectEqual> type_map_;

  // The solver used by the inferencer.
  TypeSolver solver_;
  // relation function
  TypeRelationFn tuple_getitem_rel_;
  TypeRelationFn make_tuple_rel_;

  // Perform unification on two types and report the error at the expression
  // or the span of the expression.
  Type Unify(const Type& t1, const Type& t2, const ObjectRef& expr) {
    try {
      return solver_.Unify(t1, t2, expr);
    } catch (const dmlc::Error &e) {
      this->ReportFatalError(
        expr,
        ErrorBuilder()
          << "Error unifying `"
          << t1
          << "` and `"
          << t2
          << "`: " << e.what());
      return Type();
    }
  }

  // Lazily get type for expr
  // expression, we will populate it now, and return the result.
  Type GetType(const Expr &expr) {
    auto it = type_map_.find(expr);
    if (it != type_map_.end() && it->second.checked_type.defined()) {
      return it->second.checked_type;
    }
    Type ret = this->VisitExpr(expr);
    CHECK(ret.defined());
    KindCheck(ret, mod_);
    ResolvedTypeInfo& rti = type_map_[expr];
    rti.checked_type = ret;
    return ret;
  }

  void ReportFatalError(const ObjectRef& expr, const Error& err) {
    CHECK(this->current_func_.defined());
    this->err_reporter.ReportAt(this->current_func_, expr, err);
    this->err_reporter.RenderErrors(this->mod_);
  }

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
      this->ReportFatalError(
        GetRef<GlobalVar>(op),
        ErrorBuilder() <<
          "Cannot do type inference on global variables " \
          "without a module");
    }
    Expr e = mod_->Lookup(var);
    return e->checked_type();
  }

  Type VisitExpr_(const ConstantNode* op) final {
    return op->tensor_type();
  }

  Type VisitExpr_(const TupleNode* op) final {
    Array<Type> types;
    for (Expr field : op->fields) {
      types.push_back(GetType(field));
    }
    return TupleType(types);
  }

  Type VisitExpr_(const TupleGetItemNode* op) final {
    if (!tuple_getitem_rel_.defined())  {
      tuple_getitem_rel_ = Downcast<TypeRelationFn>(
          EnvFunc::Get("tvm.relay.type_relation.TupleGetItem"));
    }
    Type tuple_type = GetType(op->tuple);
    Type rtype = IncompleteType(Kind::kType);
    auto attrs = make_object<TupleGetItemAttrs>();
    attrs->index = op->index;
    solver_.AddConstraint(TypeRelation(
        tuple_getitem_rel_, {tuple_type, rtype}, 1, Attrs(attrs)), GetRef<TupleGetItem>(op));
    return rtype;
  }

  void VisitPattern_(const PatternConstructorNode* con, const Type& t) {
    CHECK(mod_.defined())
      << "Cannot do type inference without a environment:"
      << con->constructor->name_hint;
    TypeData td = mod_->type_definitions.at(con->constructor->belong_to);
    auto pc = GetRef<PatternConstructor>(con);

    // we can expect a certain number of arguments
    Array<Type> unknown_args;
    for (size_t i = 0; i < td->type_vars.size(); i++) {
      unknown_args.push_back(IncompleteType(Kind::kType));
    }
    Type expected = TypeCall(con->constructor->belong_to, unknown_args);
    Type unified = Unify(t, expected, GetRef<ObjectRef>(con));

    auto* tc = unified.as<TypeCallNode>();
    if (!tc) {
      this->ReportFatalError(pc, ErrorBuilder() << "Expected a type call, got " << unified);
    }
    if (td->header != tc->func) {
      this->ReportFatalError(pc,
        ErrorBuilder() << "ADT headers must match, but we have "
                        << td->header << " and " << tc->func);
    }
    if (td->type_vars.size() != tc->args.size()) {
      this->ReportFatalError(pc,
        ErrorBuilder() << "The number of type args must match"
                       << "the number of type vars in the type data: "
                       << td->type_vars.size() << " != " << tc->args.size());
    }
    std::unordered_map<TypeVar, Type, ObjectHash, ObjectEqual> type_var_map_;
    for (size_t i = 0; i < td->type_vars.size(); ++i) {
      type_var_map_[td->type_vars[i]] = tc->args[i];
    }
    CHECK(con->constructor->inputs.size() == con->patterns.size()) << "not enough pattern";
    if (con->constructor->inputs.size() != con->patterns.size()) {
      this->ReportFatalError(pc,
        ErrorBuilder() << "Not enough inputs for the constructor; "
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
    Type unified = Unify(t, expected, GetRef<ObjectRef>(tup));

    auto* tt = unified.as<TupleTypeNode>();
    if (!tt) {
      this->ReportFatalError(pt, ErrorBuilder() << "Expected a tuple type, got " << unified);
    }
    CHECK(tup->patterns.size() == tt->fields.size()) << "not enough pattern";
    for (size_t i = 0; i < tup->patterns.size(); ++i) {
      VisitPattern(tup->patterns[i], tt->fields[i]);
    }
  }

  void VisitPattern_(const PatternVarNode* pv, const Type& t) {
    Type vt = GetType(pv->var);
    Unify(vt, t, pv->span);
  }

  void VisitPattern_(const PatternWildcardNode* wc, const Type& t) { }

  Type VisitExpr_(const MatchNode* op) final {
    Type dtype = GetType(op->data);
    for (const auto& c : op->clauses) {
      VisitPattern(c->lhs, dtype);
    }
    Type rtype = IncompleteType(Kind::kType);
    for (const auto& c : op->clauses) {
      rtype = this->Unify(rtype,
                          GetType(c->rhs),
                          op->span);
    }

    if (op->complete) {
      // check completness
      Match match = GetRef<Match>(op);
      Array<Pattern> unmatched_cases = UnmatchedCases(match, this->mod_);
      if (unmatched_cases.size() != 0) {
        ErrorBuilder ss;
        ss << "match expression does not handle the following cases: ";
        int i = 0;
        for (auto cs : unmatched_cases) {
          ss << "case " << i++ << ": \n" << PrettyPrint(cs);
        }
        this->ReportFatalError(
          match,
          ss);
      }
    }

    return rtype;
  }

  Type VisitExpr_(const OpNode* op) final {
    return op->op_type;
  }

  Type VisitExpr_(const LetNode* let) final {
    // if the definition is a function literal, permit recursion
    bool is_functional_literal = let->value.as<FunctionNode>() != nullptr;
    Type let_type = IncompleteType(Kind::kType);

    if (is_functional_literal) {
      let_type = GetType(let->var);
      type_map_[let->var].checked_type = let_type;
    }


    if (let->var->type_annotation.defined()) {
      let_type = Unify(let_type, let->var->type_annotation, GetRef<Let>(let));
    }

    Type vtype = GetType(let->value);
    let_type = Unify(let_type, vtype, GetRef<Let>(let));

    CHECK(is_functional_literal || !type_map_.count(let->var));
    // NOTE: no scoping is necessary because var are unique in program
    type_map_[let->var].checked_type = let_type;
    return GetType(let->body);
  }

  Type VisitExpr_(const IfNode* ite) final {
    // Ensure the type of the guard is of Tensor[Bool, ()],
    // that is a rank-0 boolean tensor.
    Type cond_type = this->GetType(ite->cond);
    this->Unify(cond_type,
                TensorType::Scalar(tvm::DataType::Bool()),
                ite->cond);
    Type checked_true = this->GetType(ite->true_branch);
    Type checked_false = this->GetType(ite->false_branch);
    return this->Unify(checked_true, checked_false, GetRef<If>(ite));
  }

  // This code is special-cased for primitive operators,
  // which are registered in the style defined in src/relay/op/*.
  //
  // The result will be the return type of the operator.
  Type PrimitiveCall(const FuncTypeNode* op,
                     Array<Type> arg_types,
                     const Attrs& attrs,
                     const ObjectRef& loc) {
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
    solver_.AddConstraint(TypeRelation(
        rel->func, arg_types, arg_types.size() - 1, attrs), loc);
    return rtype;
  }

  // substitute the type args in the function type
  FuncType InstantiateFuncType(const FuncTypeNode* fn_ty, const Array<Type>& ty_args) {
    tvm::Map<TypeVar, Type> subst_map;

    // Build a subsitituion map up from the function type and type arguments.
    // Eventually allow the type vars to be passed in.
    for (size_t i = 0; i < ty_args.size(); ++i) {
      subst_map.Set(fn_ty->type_params[i], ty_args[i]);
    }

    for (size_t i = ty_args.size(); i < fn_ty->type_params.size(); ++i) {
      subst_map.Set(fn_ty->type_params[i], IncompleteType(Kind::kType));
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

    Type inst_ty = FuncType(fn_ty->arg_types,
                                      ret_type, {},
                                      fn_ty->type_constraints);
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
      CHECK(!type_info->second.type_args.defined());
      type_info->second.type_args = type_args;
    }
  }

  // Handle general call node.
  Type GeneralCall(const CallNode* call, Array<Type> arg_types) {
    Type ftype = GetType(call->op);
    auto* fn_ty_node = ftype.as<FuncTypeNode>();
    auto* inc_ty_node = ftype.as<IncompleteTypeNode>();

    if (fn_ty_node == nullptr && inc_ty_node == nullptr) {
      this->ReportFatalError(
        GetRef<Call>(call),
        ErrorBuilder()
          << "only expressions with function types can be called, found "
          << ftype);
    }

    // incomplete type => it must be a function taking the arg types
    // with an unknown return type
    if (inc_ty_node != nullptr) {
      Type ret_type = IncompleteType(Kind::kType);
      Type func_type = FuncType(arg_types, ret_type, {}, {});
      Type unified = this->Unify(ftype, func_type, GetRef<Call>(call));
      fn_ty_node = unified.as<FuncTypeNode>();
    }

    Array<Type> type_args = call->type_args;
    if (type_args.size() > fn_ty_node->type_params.size()) {
      this->ReportFatalError(GetRef<Call>(call),
        ErrorBuilder()
          << "Incorrect number of type args in "
          << call->span << ": "
          << "Expected "
          << fn_ty_node->type_params.size()
          << "but got " << type_args.size());
    }

    FuncType fn_ty = InstantiateFuncType(fn_ty_node, type_args);

    AddTypeArgs(GetRef<Call>(call), type_args);

    size_t type_arity = fn_ty->arg_types.size();
    size_t number_of_args = arg_types.size();

    if (type_arity != number_of_args) {
      if (type_arity < number_of_args) {
        this->ReportFatalError(
          GetRef<Call>(call),
          ErrorBuilder()
            << "the function is provided too many arguments "
            << "expected " << type_arity << ", found " << number_of_args);
      } else {
        this->ReportFatalError(
          GetRef<Call>(call),
          ErrorBuilder()
            << "the function is provided too few arguments "
            << "expected " << type_arity << ", found " << number_of_args);
      }
    }

    for (size_t i = 0; i < fn_ty->arg_types.size(); i++) {
      this->Unify(fn_ty->arg_types[i], arg_types[i], GetRef<Call>(call));
    }

    for (auto cs : fn_ty->type_constraints) {
      if (const auto* tr = cs.as<TypeRelationNode>()) {
        solver_.AddConstraint(
          TypeRelation(tr->func, tr->args, tr->num_inputs, call->attrs),
          GetRef<Call>(call));
      } else {
        solver_.AddConstraint(cs, GetRef<Call>(call));
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
                                 call->attrs,
                                 GetRef<Call>(call));
      if (rtype.defined()) {
        AddTypeArgs(GetRef<Call>(call), arg_types);
        return rtype;
      }
    }

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
      rtype = this->Unify(f->ret_type, rtype, GetRef<Function>(f));
    }
    CHECK(rtype.defined());
    auto ret = FuncType(arg_types, rtype, f->type_params, {});
    return solver_.Resolve(ret);
  }

  Type VisitExpr_(const RefCreateNode* op) final {
    return RelayRefType(GetType(op->value));
  }

  Type VisitExpr_(const RefReadNode* op) final {
    Type it = IncompleteType(Kind::kType);
    this->Unify(GetType(op->ref), RelayRefType(it), GetRef<RefRead>(op));
    return it;
  }

  Type VisitExpr_(const RefWriteNode* op) final {
    Type it = IncompleteType(Kind::kType);
    this->Unify(GetType(op->ref), RelayRefType(it), GetRef<RefWrite>(op));
    this->Unify(GetType(op->value), it, GetRef<RefWrite>(op));
    return TupleType::Empty();
  }

  Type VisitExpr_(const ConstructorNode* c) final {
    CHECK(mod_.defined())
      << "Cannot do type inference without a environment:"
      << c->name_hint;
    TypeData td = mod_->LookupTypeDef(c->belong_to);
    std::vector<Type> types;
    for (const auto & t : td->type_vars) {
      types.push_back(t);
    }
    return FuncType(c->inputs, TypeCall(c->belong_to, types),
                              td->type_vars, {});
  }

  void Solve() {
    solver_.Solve();

    if (err_reporter.AnyErrors()) {
      err_reporter.RenderErrors(mod_);
    }
  }
};

class TypeInferencer::Resolver : public ExprMutator, PatternMutator {
 public:
  Resolver(const std::unordered_map<Expr, ResolvedTypeInfo, ObjectHash, ObjectEqual>& tmap,
           TypeSolver* solver)
    : tmap_(tmap), solver_(solver) {
  }

  Expr VisitExpr_(const VarNode* op) final {
    return VisitVar(GetRef<Var>(op));
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

  Expr VisitExpr_(const RefCreateNode* op) final {
    return AttachCheckedType(op);
  }

  Expr VisitExpr_(const RefReadNode* op) final {
    return AttachCheckedType(op);
  }

  Expr VisitExpr_(const RefWriteNode* op) final {
    return AttachCheckedType(op);
  }

  Expr VisitExpr_(const ConstructorNode* op) final {
    return AttachCheckedType(op);
  }

  Expr VisitExpr_(const MatchNode* op) final {
    return AttachCheckedType(op);
  }

  Pattern VisitPattern(const Pattern& p) final {
    return PatternMutator::VisitPattern(p);
  }

  Var VisitVar(const Var& v) final {
    if (vmap_.count(v) == 0) {
      vmap_[v] = GetRef<Var>(AttachCheckedType(v.as<VarNode>()).as<VarNode>());
    }
    return vmap_.at(v);
  }

  // attach checked type to the mutated node.
  template<typename T>
  Expr AttachCheckedType(const T* op) {
    auto it = tmap_.find(GetRef<Expr>(op));
    CHECK(it != tmap_.end());
    Type checked_type = solver_->Resolve(it->second.checked_type);

    // TODO(@jroesch): it would be nice if we would report resolution
    // errors directly on the program.
    CHECK(checked_type.as<IncompleteTypeNode>() == nullptr)
        << "Cannot resolve type of " << GetRef<Expr>(op)
        << " at " << op->span;

    Expr new_e = ExprMutator::VisitExpr_(op);
    // new_call and new_var's code is only going to be valid for VarNode/CallNode.
    // Compiler optimization will likely fold these away for other nodes.
    CallNode* new_call =(
        std::is_base_of<CallNode, T>::value ?
        const_cast<CallNode*>(static_cast<const CallNode*>(new_e.get())) : nullptr);
    VarNode* new_var =(
        std::is_base_of<VarNode, T>::value ?
        const_cast<VarNode*>(static_cast<const VarNode*>(new_e.get())) : nullptr);
    FunctionNode* new_fn =(
        std::is_base_of<FunctionNode, T>::value ?
        const_cast<FunctionNode*>(static_cast<const FunctionNode*>(new_e.get())) : nullptr);

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

    bool need_update_fn =(
        std::is_base_of<FunctionNode, T>::value &&
        update_missing_type_annotation_ &&
        !new_fn->ret_type.defined());

    if (!need_update_type &&
        !need_update_var &&
        !need_update_call &&
        !need_update_fn) {
      return new_e;
    }

    if (!new_e.unique()) {
      // Copy on write optimization
      // If new_e is an old expression,
      // we make a copy mutating an existing reference.
      ObjectPtr<ExprNode> ptr = make_object<T>(*new_e.as<T>());
      new_e = Expr(ptr);
      new_call = (
          std::is_base_of<CallNode, T>::value ?
          static_cast<CallNode*>(ptr.get()) : nullptr);
      new_var = (
          std::is_base_of<VarNode, T>::value ?
          static_cast<VarNode*>(ptr.get()) : nullptr);
      new_fn = (
          std::is_base_of<FunctionNode, T>::value ?
          static_cast<FunctionNode*>(ptr.get()) : nullptr);
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
  std::unordered_map<Var, Var, ObjectHash, ObjectEqual> vmap_;
  const std::unordered_map<Expr, ResolvedTypeInfo, ObjectHash, ObjectEqual>& tmap_;
  TypeSolver* solver_;
  // whether attach the checked type as type_annotation
  // if original type anntation is missing.
  bool update_missing_type_annotation_{true};
};

Expr TypeInferencer::Infer(Expr expr) {
  // Step 1: Populate the constraints.
  GetType(expr);

  // Step 2: Solve the constraints.
  Solve();

  // Step 3: Attach resolved types to checked_type field.
  auto resolved_expr = Resolver(type_map_, &solver_).VisitExpr(expr);
  CHECK(WellFormed(resolved_expr));
  return resolved_expr;
}

struct AllCheckTypePopulated : ExprVisitor {
  void VisitExpr(const Expr& e) {
    if (e.as<OpNode>()) { return; }
    if (e.as<GlobalVarNode>()) { return; }
    if (e.as<ConstructorNode>()) { return; }
    CHECK(e->checked_type_.defined()) << "Expression: " << e;
    return ExprVisitor::VisitExpr(e);
  }
};

void EnsureCheckedType(const Expr& e) {
  AllCheckTypePopulated().VisitExpr(e);
}

Expr InferType(const Expr& expr, const IRModule& mod) {
  auto main = mod->GetGlobalVar("main");
  auto inferencer = TypeInferencer(mod, main);
  auto e = inferencer.Infer(expr);
  CHECK(WellFormed(e));
  auto free_tvars = FreeTypeVars(e, mod);
  CHECK(free_tvars.size() == 0)
    << "Found unbound type variables in " << e << ": " << free_tvars;
  EnsureCheckedType(e);
  return e;
}

Function InferType(const Function& func,
                   const IRModule& mod,
                   const GlobalVar& var) {
  CHECK(mod.defined()) << "internal error: module must be set for type inference";
  Function func_copy = Function(make_object<FunctionNode>(*func.operator->()));
  func_copy->checked_type_ = func_copy->func_type_annotation();
  mod->AddUnchecked(var, func_copy);
  Expr func_ret = TypeInferencer(mod, var).Infer(func_copy);
  mod->Remove(var);
  CHECK(WellFormed(func_ret));
  auto free_tvars = FreeTypeVars(func_ret, mod);
  CHECK(free_tvars.size() == 0)
    << "Found unbound type variables in: "
    << std::endl
    << AsText(func, true)
    << std::endl << free_tvars;
  return Downcast<Function>(func_ret);
}

namespace transform {

Pass InferType() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
    [=](Function f, IRModule m, PassContext pc) {
      return Downcast<Function>(InferType(f, m));
  };
  return CreateFunctionPass(pass_func, 0, "InferType", {});
}

TVM_REGISTER_GLOBAL("relay._transform.InferType")
.set_body_typed([]() {
  return InferType();
});

}  // namespace transform

}  // namespace relay
}  // namespace tvm
