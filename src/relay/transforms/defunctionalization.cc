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
 * \brief
 */

#include <tvm/ir/type_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/feature.h>
#include <tvm/relay/transform.h>
#include <tvm/te/operation.h>

#include "../analysis/type_solver.h"
#include "../transforms/pass_util.h"
namespace tvm {
namespace relay {

struct FuncTypeVisitor : TypeVisitor {
  bool has_func_type;
  FuncTypeVisitor() : has_func_type(false) {}

  void VisitType_(const FuncTypeNode* op) { this->has_func_type = true; }
};

bool HasFuncType(const Expr& e) {
  auto visitor = FuncTypeVisitor();
  visitor.VisitType(e->checked_type());
  return visitor.has_func_type;
}

bool HasFuncType(const Type& t) {
  auto visitor = FuncTypeVisitor();
  visitor.VisitType(t);
  return visitor.has_func_type;
}

bool IsHigherOrderFunc(const FuncType& t) {
  bool higher_order = false;
  for (auto arg: t->arg_types) {
    higher_order |= HasFuncType(arg);
  }
  return higher_order |= HasFuncType(t->ret_type);
}

Array<Type> InferTypeArgs(const CallNode* call, const IRModule& mod) {
  ErrorReporter err;
  TypeSolver solver(mod->GetGlobalVar("main"), mod, &err);
  const FuncTypeNode* fn_ty = call->op->checked_type().as<FuncTypeNode>();

  tvm::Map<TypeVar, Type> subst_map;
  for (auto& tv: fn_ty->type_params) {
    subst_map.Set(tv, IncompleteType(Kind::kType));
  }

  auto inst_fnty = FuncType(fn_ty->arg_types, fn_ty->ret_type, {}, {});
  auto f_incomplete = Downcast<FuncType>(Bind(inst_fnty, subst_map));

  CHECK(call->args.size() == f_incomplete->arg_types.size()) << "num of arguments does not match expected";
  size_t num_args = f_incomplete->arg_types.size();
  for (size_t i = 0; i < num_args; i++) {
    auto t1 = f_incomplete->arg_types[i];
    auto t2 = call->args[i]->checked_type();
    auto t = solver.Unify(t1, t2, GetRef<Call>(call));
  }
  Array<Type> ret;
  for (auto& tv: fn_ty->type_params) {
    std::cout << "Resolved Type: " << solver.Resolve(subst_map[tv]) << std::endl;
    ret.push_back(solver.Resolve(subst_map[tv])); 
  }
  return ret;
}

class DefuncMutator : public ExprMutator {
 public:
  DefuncMutator(const IRModule& mod) : mod(mod), constructor_name(0), anon_name(0) {}

  Expr VisitExpr_(const CallNode* op) {
    auto op_func = op->op;
    auto f = DeGlobal(mod, op_func).as<FunctionNode>();
    CHECK(f) << "only calls to functions or globalvars are supported so far";
    // CHECK(op->type_args.size() == f->type_params.size()) << "all type args must be explicit";

    // clone function and specialize if there are higher order functions
    if (IsHigherOrderFunc(Downcast<FuncType>(f->func_type_annotation()))) {
      std::cout << "Call Function: " << GetRef<Function>(f) << std::endl;

      std::string name;
      if (auto gv = op->op.as<GlobalVarNode>()) {
        name = gv->name_hint;
      } else {
        name = "anon" + std::to_string(anon_name++);
      }
      auto clone_gv = Clone(name, f, InferTypeArgs(op, mod));
      auto f_clone = Downcast<Function>(DeGlobal(mod, clone_gv));
      std::cout << f_clone << std::endl;
      auto f_clone_type = f_clone->func_type_annotation();
      CHECK(FreeTypeVars(f_clone_type, mod).size() == 0)
          << "free type vars in specialized function";
      CHECK(FreeVars(f_clone).size() == FreeVars(GetRef<Function>(f)).size())
          << "local closures not supported yet";
      CHECK(!HasFuncType(f_clone_type->ret_type)) << "returning function not supported yet";

      Array<Expr> args;
      std::unordered_map<Var, GlobalVar, ObjectHash, ObjectEqual> applyVars;
      for (size_t i = 0; i < f_clone_type->arg_types.size(); i++) {
        if (f_clone_type->arg_types[i].as<FuncTypeNode>()) {
          auto arg = EncodeFunctionArg(op->args[i], f_clone_type->arg_types[i].as<FuncTypeNode>());
          args.push_back(arg);
          applyVars[f_clone->params[i]] = apply_map[f_clone->params[i]->checked_type()];
        } else {
          CHECK(!HasFuncType(f_clone_type->arg_types[i])) << "nested function type in parameter not supported yet";
          args.push_back(op->args[i]);
        }
      }
      auto new_func = ApplyVars(clone_gv, applyVars);

      return Call(new_func, args);
    }
    return ExprMutator::VisitExpr_(op);
  }

  // Expr VisitExpr_(const LetNode* op) {
  //   var_map[op->var] = this->VisitExpr(op->value);
  //   return this->VisitExpr(op->body);
  // }

  // Expr VisitExpr_(const VarNode* op) {
  //   if (var_map.count(GetRef<Var>(op)) != 0) {
  //     return var_map[GetRef<Var>(op)];
  //   }
  //   return GetRef<Var>(op);
  // }

 private:
  IRModule mod;
  // encode func type to ADT
  std::unordered_map<Type, GlobalTypeVar, ObjectHash, StructuralEqual> func_encoding;
  std::unordered_map<Type, GlobalVar, ObjectHash, StructuralEqual> apply_map;
  // use monotonically increasing integer to represent new constructor_name
  unsigned int constructor_name;
  unsigned int anon_name;

  Expr ApplyVars(GlobalVar gv, const std::unordered_map<Var, GlobalVar, ObjectHash, ObjectEqual>& vars) {
    struct ApplyVarMutator: public ExprMutator {
      std::unordered_map<Var, GlobalVar, ObjectHash, ObjectEqual> vars;
      std::unordered_map<Var, Var, ObjectHash, ObjectEqual> var_map;
      ApplyVarMutator(const std::unordered_map<Var, GlobalVar, ObjectHash, ObjectEqual>& vars, const std::unordered_map<Var, Var, ObjectHash, ObjectEqual>& var_map) : vars(vars), var_map(var_map) {}
      Expr VisitExpr_(const CallNode* op) {
        if (auto var_op = op->op.as<VarNode>()) {
          if (vars.count(GetRef<Var>(var_op)) != 0) {
            auto gv = vars[GetRef<Var>(var_op)];
            Array<Expr> args = {GetRef<Var>(var_op)};
            for (auto arg: op->args) {
              args.push_back(arg);
            }
            return ExprMutator::VisitExpr_(Call(gv, args).as<CallNode>());
          }
        } else if (IsHigherOrderFunc(Downcast<FuncType>(op->op->checked_type()))) 

        return ExprMutator::VisitExpr_(op);
      }

      Expr VisitExpr_(const VarNode* op) {
        auto var = GetRef<Var>(op);
        if (var_map.count(var) != 0) {
          return var_map[var];
        }
        return ExprMutator::VisitExpr_(op);
      }
    };
    auto e = Downcast<Function>(mod->Lookup(gv));

    std::unordered_map<Var, Var, ObjectHash, ObjectEqual> var_map;
    for (auto v : e->params) {
      if (v->type_annotation.as<FuncTypeNode>()) {
        var_map[v] = Var(v->name_hint(), IncompleteType(TypeKind::kType));
      }
    }
    auto applied = Downcast<Function>(ApplyVarMutator(vars, var_map).Mutate(e));
    auto typed = this->VisitExpr(InferType(applied, mod, gv));
    std::cout << "TYPED: " << typed << std::endl;
    mod->Add(gv, Downcast<Function>(typed), true);
    
    return gv;
  }

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

  void AddApplyCase(GlobalVar gv, FuncType ft, Constructor c, const Expr& expr) {
    if (!mod->ContainGlobalVar(gv->name_hint)) {
      auto x = Var("x", TypeCall(c->belong_to, {}));
      auto vars = Array<Var>({x});
      auto args = Array<Expr>();
      for (auto t: ft->arg_types) {
        auto y = Var("y", t);
        vars.push_back(y);
        args.push_back(y);
      }

      auto clauses = Array<Clause>({Clause(PatternConstructor(c, {}), Call(expr, args))});
      auto body = Match(x, clauses);
      auto f = Function(vars, body, ft->ret_type, {});

      mod->Add(gv, f);
    } else {
      auto f = Downcast<Function>(mod->Lookup(gv));
      auto body = f->body.as<MatchNode>();
      CHECK(body) << "internal invariant broken; apply function body should be a match node";

      auto clauses = body->clauses;
      auto x = f->params[0];
      auto args = Array<Expr>();
      for (size_t i = 1; i < f->params.size(); i++) {
        args.push_back(f->params[i]);
      }
      clauses.push_back(Clause(PatternConstructor(c, {}), Call(expr, args)));

      mod->Add(gv, Function(f->params, Match(x, clauses), f->ret_type, f->type_params), true);
    }
  }

  Expr EncodeFunctionArg(const Expr& f, const FuncTypeNode* ft) {
    auto adt_name = "T" + TypeToString(ft);
    if (func_encoding.count(GetRef<FuncType>(ft)) == 0) {
      func_encoding[GetRef<FuncType>(ft)] = GlobalTypeVar(adt_name, TypeKind::kAdtHandle);
    }

    auto gtv = func_encoding[GetRef<FuncType>(ft)];
    auto c = Constructor(std::to_string(constructor_name++), {}, gtv);
    AddConstructor(gtv, c);

    if (apply_map.count(GetRef<FuncType>(ft)) == 0) {
      apply_map[GetRef<FuncType>(ft)] = GlobalVar("apply" + TypeToString(ft));
    }

    auto gv = apply_map[GetRef<FuncType>(ft)];
    AddApplyCase(gv, GetRef<FuncType>(ft), c, f);

    return Call(c, {});
  }
  
  std::string TypeToString(const TypeNode* t) {
    std::ostringstream s;
    s << GetRef<Type>(t);
    return s.str();
  }

  GlobalVar Clone(std::string name_prefix, const FunctionNode* f, const Array<Type> type_args) {
    auto spec = Specialize(f, type_args);
    auto gv_name = name_prefix + TypeToString(spec->func_type_annotation().as<FuncTypeNode>());
    std::cout << gv_name << std::endl;
    if (mod->ContainGlobalVar(gv_name)) {
      return mod->GetGlobalVar(gv_name);
    }
    auto gv = GlobalVar(gv_name);
    mod->Add(gv, Downcast<Function>(DeDup(spec)));
    return gv;
  }

  Function Specialize(const FunctionNode* f, const Array<Type> type_args) {
    auto map = tvm::Map<TypeVar, Type>();
    for (size_t i = 0; i < type_args.size(); i++) {
      map.Set(f->type_params[i], type_args[i]);
    }
    // copy with typevars removed
    auto copy = TypeSubst(Function(f->params, f->body, f->ret_type, {}), map);
    return Downcast<Function>(copy);
  }
};

Expr Defunctionalization(const Expr& e, const IRModule& mod) {
  auto f = e.as<FunctionNode>();
  CHECK(f) << "input need to be a function";
  CHECK(f->type_params.size() == 0) << "no polymorphism supported for defunctionalization";
  for (const auto& p : f->params) {
    CHECK(!HasFuncType(p)) << "input parameters cannot have func type";
  }

  return DefuncMutator(mod).VisitExpr(e);
}

TVM_REGISTER_GLOBAL("relay._transform.Defunctionalization").set_body_typed(Defunctionalization);

}  // namespace relay
}  // namespace tvm
