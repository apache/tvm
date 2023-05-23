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
 *
 * \file defunctionalization.cc
 *
 * \brief Defunctionalization for Relay IR
 *
 * This pass transforms a higher-order program into a first-order program with defunctionalization.
 * This means that all higher order functions (i.e functions that take function arguments or return
 * functions) should be transformed into a semantically equivalent first order one.
 *
 * This pass implements a basic typed defunctionalization method.
 * All higher order functions are cloned and specialized (so that there are no type params).
 * Function type arguments are encoded as datatypes and a helper `apply` function is used
 * to "call" them.
 *
 * For example, take the following higher order program:
 * fun map F y = case y of
 *          Nil => Nil
 *          | Cons(x, XS) => Cons(F z, map F XS)
 * fun addone 1 = map (\x -> \x + 1) 1
 *
 * where `addone` is our program.
 * When we call the `map` function, we see that it is a higher-order function,
 * but we can clone `map ` function and specialize it with the type_params of the call.
 * In addition, our function argument `(\x -> \x + 1)` will be encoded as a datatype constructor,
 * which we will call `incr`, and all calls to `F` in our specialized map function will use the
 * helper `apply` function.
 *
 * After defunctionalization, we get:
 * fun apply encoding arg =  case encoding of
 *     “incr” => incr arg
 * fun map’ F y = case y of
 *           Nil => Nil
 *           | Cons(x, xs) => Cons(apply F x, map’ F xs)
 * fun addone 1 = map’ “incr” 1
 *
 * Currently, defunctionalization makes the following assumptions:
 * - functions cannot return function values
 * - function arguments are in two forms: identifier or a lambda abstraction
 * - no functions stored in datatype
 * - functions are not let binded
 */

#include <tvm/ir/type_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/feature.h>
#include <tvm/relay/transform.h>
#include <tvm/te/operation.h>

#include "../analysis/type_solver.h"
#include "../transforms/pass_utils.h"
namespace tvm {
namespace relay {

// determine if type contains a FuncType
bool HasFuncType(const Type& t) {
  struct FuncTypeVisitor : TypeVisitor {
    bool has_func_type;
    FuncTypeVisitor() : has_func_type(false) {}

    void VisitType_(const FuncTypeNode* op) { this->has_func_type = true; }
  };

  auto visitor = FuncTypeVisitor();
  visitor.VisitType(t);
  return visitor.has_func_type;
}
// determine if FuncType is a higher order type
bool IsHigherOrderFunc(const FuncType& t) {
  bool higher_order = false;
  for (auto arg : t->arg_types) {
    higher_order |= HasFuncType(arg);
  }
  return higher_order |= HasFuncType(t->ret_type);
}

/*!
 * \brief mutator for driving the Defunctionalization transformation
 */
class DefuncMutator : public ExprMutator {
 public:
  explicit DefuncMutator(const IRModule& mod) : mod(mod), constructor_counter(0) {}

  Expr VisitExpr_(const CallNode* call) {
    if (auto op = call->op.as<GlobalVarNode>()) {
      ICHECK_EQ(call->type_args.size(), op->checked_type().as<FuncTypeNode>()->type_params.size())
          << "all type args must be explicit";

      auto op_type = InstFuncType(op->checked_type().as<FuncTypeNode>(), call->type_args);
      ICHECK_EQ(FreeTypeVars(op_type, mod).size(), 0) << "free type vars in instantiated";
      ICHECK(!HasFuncType(op_type->ret_type)) << "returning functions not supported";

      if (!IsHigherOrderFunc(op_type)) {
        // not higher order function
        return ExprMutator::VisitExpr_(call);
      }

      // first we encode function arguments
      Array<Expr> args;
      for (size_t i = 0; i < call->args.size(); i++) {
        auto arg = call->args[i];
        auto type = op_type->arg_types[i];
        if (!HasFuncType(type)) {
          args.push_back(arg);
        } else {
          args.push_back(EncodeArg(arg, type));
        }
      }
      auto name = op->name_hint + TypeToString(op_type);
      auto gv = GlobalVar(name);
      if (specialized_gv_map.count(name)) {
        gv = specialized_gv_map[name];
      } else {
        specialized_gv_map[name] = gv;
        // clone and specialize with specific type
        auto clone = Downcast<Function>(DeDup(mod->Lookup(GetRef<GlobalVar>(op))));
        auto specialized_function = Specialize(clone, call->type_args);
        // change var types and change all applications to use `apply` method
        auto f = Downcast<Function>(FirstifyVars(specialized_function));
        mod->Add(gv, f);
      }
      return Call(gv, args);
    } else if (auto op = call->op.as<FunctionNode>()) {
      // reduction by applying vars
      std::unordered_map<Var, Expr, ObjectHash, ObjectEqual> var_binding_map;
      for (size_t i = 0; i < op->params.size(); i++) {
        var_binding_map[op->params[i]] = call->args[i];
      }
      auto e = Bind(op->body, var_binding_map);
      return this->VisitExpr(e);
    } else if (auto op = call->op.as<VarNode>()) {
      // var node will be encoded as datatype
      // so we need to use the `apply` helper method
      auto var_original_type = GetUnencodedType(op->type_annotation).as<FuncTypeNode>();
      ICHECK(var_original_type) << "var original type not saved in var_save_type map";
      auto op_type = InstFuncType(var_original_type, call->type_args);

      Array<Expr> args = {GetRef<Var>(op)};
      for (auto arg : call->args) {
        args.push_back(this->VisitExpr(arg));
      }

      return Call(GetApplyFunction(op_type), args);
    }
    return ExprMutator::VisitExpr_(call);
  }

 private:
  // module
  IRModule mod;
  // gv + str(type) to specialized clone gv
  std::unordered_map<std::string, GlobalVar> specialized_gv_map;
  // str(func_type) to ADT
  std::unordered_map<std::string, GlobalTypeVar> func_encoding;
  // str(func_tyoe) to apply gv
  std::unordered_map<std::string, GlobalVar> apply_map;
  // encoded ADT handle to FuncType
  std::unordered_map<GlobalTypeVar, Type, ObjectHash, StructuralEqual> original_func_type_map;
  // gv to (str(func_type) to constructor encoding)
  std::unordered_map<GlobalVar, std::unordered_map<std::string, Constructor>, ObjectHash,
                     ObjectEqual>
      gv_datatype_map;
  // use monotonically increasing integer to represent new constructor_name
  uint64_t constructor_counter;

  /*!
   * \brief add a constructor to the GlobalTypeVar, creating a new TypeDef if GlobalTypeVar does not
   * exist
   */
  void AddConstructor(GlobalTypeVar gtv, Constructor c) {
    if (!mod->ContainGlobalTypeVar(gtv->name_hint)) {
      mod->AddTypeDef(gtv, TypeData(gtv, {}, {c}));
    } else {
      auto typedata = mod->LookupTypeDef(gtv);
      auto constructors = typedata->constructors;
      constructors.push_back(c);
      mod->UpdateTypeDef(gtv, TypeData(typedata->header, typedata->type_vars, constructors));
    }
  }
  /*!
   * \brief add a case to the apply function, creating the function if it does not exist
   *
   * \param apply_gv GlobalVar of the apply function
   * \param ft is the type functions the apply function handles
   * \param c constructor to add a case for
   * \param expr calls this expr with the args to the apply_gv
   * \param patterns PatterVars to match with the constructor, used for handling free vars in
   * functions
   */
  void AddApplyCase(GlobalVar apply_gv, FuncType ft, Constructor c, const Expr& expr,
                    const Array<Pattern> patterns) {
    ICHECK(c->inputs.size() == patterns.size())
        << "constructor function and pattern vars have different sizes";
    if (!mod->ContainGlobalVar(apply_gv->name_hint)) {
      auto x = Var("x", TypeCall(c->belong_to, {}));
      auto vars = Array<Var>({x});
      auto args = Array<Expr>();
      for (auto t : ft->arg_types) {
        auto y = Var("y", t);
        vars.push_back(y);
        args.push_back(y);
      }

      auto clauses = Array<Clause>({Clause(PatternConstructor(c, patterns), Call(expr, args))});
      auto body = Match(x, clauses);
      auto f = Function(vars, body, ft->ret_type, {});

      mod->Add(apply_gv, f);
    } else {
      auto f = Downcast<Function>(mod->Lookup(apply_gv));
      auto body = f->body.as<MatchNode>();
      ICHECK(body) << "internal invariant broken; apply function body should be a match node";

      auto clauses = body->clauses;
      auto x = f->params[0];
      auto args = Array<Expr>();
      for (size_t i = 1; i < f->params.size(); i++) {
        args.push_back(f->params[i]);
      }
      clauses.push_back(Clause(PatternConstructor(c, patterns), Call(expr, args)));

      mod->Add(apply_gv, Function(f->params, Match(x, clauses), f->ret_type, f->type_params), true);
    }
  }

  Expr EncodeArg(const Expr& arg, const Type& type) {
    // we assume arg is either an identifier (var or globalvar) or a function
    ICHECK(type.as<FuncTypeNode>()) << "assume no nested functions";
    ICHECK(arg.as<VarNode>() || arg.as<GlobalVarNode>() || arg.as<FunctionNode>())
        << "assume all first-order-parameters are identifiers or functions";

    if (arg.as<VarNode>()) {
      // variable with functype will be encoded as datatype in surrounding function
      return arg;
    } else if (arg.as<GlobalVarNode>()) {
      return EncodeGlobalVar(Downcast<GlobalVar>(arg), Downcast<FuncType>(type));
    } else if (auto fn = arg.as<FunctionNode>()) {
      // we handle free vars in anonymous functions by adding arguments to
      // the constructor function
      auto free_vars = FreeVars(arg);
      auto ft = Downcast<FuncType>(type);

      auto arg_types = Array<Type>();
      auto pattern_vars = Array<Pattern>();
      auto call_args = Array<Expr>();
      Map<Var, Expr> free_var_bind_map;
      for (auto free_var : free_vars) {
        // free vars are already encoded, can only exist within
        // specialized functions
        if (free_var->type_annotation.defined()) {
          arg_types.push_back(free_var->type_annotation);
        } else {
          arg_types.push_back(free_var->checked_type());
        }
        auto new_var = Var(free_var->name_hint(), free_var->type_annotation);
        free_var_bind_map.Set(free_var, new_var);
        pattern_vars.push_back(PatternVar(new_var));
        call_args.push_back(free_var);
      }
      auto gtv = GetFuncEncode(ft);
      auto c = Constructor(std::to_string(++constructor_counter), arg_types, gtv);
      AddConstructor(gtv, c);

      auto apply_gv = GetApplyFunction(ft);
      auto body = this->VisitExpr(Bind(fn->body, free_var_bind_map));
      AddApplyCase(apply_gv, ft, c, WithFields(GetRef<Function>(fn), fn->params, body),
                   pattern_vars);

      return Call(c, call_args);
    }
    LOG(FATAL) << "EncodeArg failed to cast arg into identifier node or function node";
  }

  /*!
   * \brief encode a global var with a specialized type with a datatype
   */
  Expr EncodeGlobalVar(const GlobalVar& gv, const FuncType& ft) {
    auto map = gv_datatype_map[gv];
    auto type_key = TypeToString(ft);
    if (map.count(type_key) == 0) {
      auto gtv = GetFuncEncode(ft);
      auto c = Constructor(std::to_string(constructor_counter++), {}, gtv);
      map[type_key] = c;
      AddConstructor(gtv, c);
      AddApplyCase(GetApplyFunction(ft), ft, c, gv, {});
    }
    return Call(map[type_key], {});
  }

  /*!
   * \brief type to string
   */
  std::string TypeToString(const Type& t) {
    std::ostringstream s;
    s << t->GetTypeKey();
    return s.str();
  }

  /*!
   * \brief get ADT handle for encoding type t
   */
  GlobalTypeVar GetFuncEncode(const Type& t) {
    auto adt_name = "Defunc" + TypeToString(t);
    if (func_encoding.count(adt_name) == 0) {
      func_encoding[adt_name] = GlobalTypeVar(adt_name, TypeKind::kAdtHandle);
    }
    original_func_type_map[func_encoding[adt_name]] = t;
    return func_encoding[adt_name];
  }

  /*!
   * \brief get original function type represented by type t
   */
  FuncType GetUnencodedType(const Type& t) {
    auto tc = t.as<TypeCallNode>();
    ICHECK(tc) << "expected type call when getting original type from encoded type";
    auto gv = tc->func.as<GlobalTypeVarNode>();
    ICHECK(gv) << "expected global type var in encoded type";
    auto type = original_func_type_map[GetRef<GlobalTypeVar>(gv)];
    ICHECK(type.defined()) << "reverse mapping from encoded type to original type not found";
    return Downcast<FuncType>(type);
  }

  /*!
   * \brief get the apply function for calling datatypes encoding functions of type t
   */
  GlobalVar GetApplyFunction(const Type& t) {
    auto f_name = "apply" + TypeToString(t);
    if (apply_map.count(f_name) == 0) {
      apply_map[f_name] = GlobalVar("apply" + TypeToString(t));
    }
    return apply_map[f_name];
  }

  /*!
   * \brief specialize a function type
   */
  FuncType InstFuncType(const FuncTypeNode* fty, const Array<Type> type_args) {
    ICHECK(fty) << "InstFuncType functype is null";
    ICHECK_EQ(fty->type_params.size(), type_args.size())
        << "size mismatch between function type params and type args";
    auto map = tvm::Map<TypeVar, Type>();
    for (size_t i = 0; i < type_args.size(); i++) {
      map.Set(fty->type_params[i], type_args[i]);
    }
    // copy with typevars removed
    return Downcast<FuncType>(TypeSubst(FuncType(fty->arg_types, fty->ret_type, {}, {}), map));
  }

  /*!
   * \brief specialize a function expression
   */
  Function Specialize(const Function& f, const Array<Type> type_args) {
    ICHECK_EQ(f->type_params.size(), type_args.size())
        << "cannot specialize function with size mismatch between function type params and type "
           "args";
    auto map = tvm::Map<TypeVar, Type>();
    for (size_t i = 0; i < type_args.size(); i++) {
      map.Set(f->type_params[i], type_args[i]);
    }
    // copy with typevars removed
    auto copy = TypeSubst(WithFields(f, {}, {}, {}, /* erase type params */ Array<TypeVar>()), map);
    return Downcast<Function>(copy);
  }

  /*!
   * \brief transform a function to be first order by transforming arg_types and
   * using the `apply` function for applications
   */
  Function FirstifyVars(const Function& f) {
    ICHECK(f->type_params.size() == 0) << "firstify function has type params";

    tvm::Map<Var, Expr> var_bind_map;
    Array<Var> params;
    for (auto var : f->params) {
      if (auto var_type = var->type_annotation.as<FuncTypeNode>()) {
        // first order parameter
        auto fop_type = GetRef<FuncType>(var_type);
        auto adt = GetFuncEncode(fop_type);
        auto new_var = Var(var->name_hint(), TypeCall(adt, {}));
        mod->LookupTypeDef(adt);
        var_bind_map.Set(var, new_var);
        params.push_back(new_var);
      } else {
        ICHECK(!HasFuncType(var->type_annotation))
            << "nested function type in parameter not supported yet";
        params.push_back(var);
      }
    }

    auto bind = Downcast<Function>(Bind(f, var_bind_map));
    return WithFields(bind, params, this->VisitExpr(bind->body), bind->ret_type,
                      /* erase type params */ Array<TypeVar>());
  }
};

Expr Defunctionalization(const Function& f, const IRModule& mod) {
  // f is the starting point of the program, all types MUST be known
  ICHECK(f->type_params.size() == 0) << "no polymorphism supported for defunctionalization";
  for (const auto& p : f->params) {
    ICHECK(!HasFuncType(p->checked_type())) << "program cannot have func type parameters";
  }
  ICHECK(!HasFuncType(f->ret_type)) << "return type cannot contain function";

  return Downcast<Function>(DefuncMutator(mod).VisitExpr(f));
}

TVM_REGISTER_GLOBAL("relay._transform.Defunctionalization").set_body_typed(Defunctionalization);

}  // namespace relay
}  // namespace tvm
