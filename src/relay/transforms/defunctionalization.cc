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

// Array<Type> InferTypeArgs(const CallNode* call, const IRModule& mod) {
//   ErrorReporter err;
//   TypeSolver solver(mod->GetGlobalVar("main"), mod, &err);
//   const FuncTypeNode* fn_ty = call->op->checked_type().as<FuncTypeNode>();

//   tvm::Map<TypeVar, Type> subst_map;
//   for (auto& tv: fn_ty->type_params) {
//     subst_map.Set(tv, IncompleteType(Kind::kType));
//   }

//   auto inst_fnty = FuncType(fn_ty->arg_types, fn_ty->ret_type, {}, {});
//   auto f_incomplete = Downcast<FuncType>(Bind(inst_fnty, subst_map));

//   CHECK(call->args.size() == f_incomplete->arg_types.size()) << "num of arguments does not match expected";
//   size_t num_args = f_incomplete->arg_types.size();
//   for (size_t i = 0; i < num_args; i++) {
//     auto t1 = f_incomplete->arg_types[i];
//     auto t2 = call->args[i]->checked_type();
//     auto t = solver.Unify(t1, t2, GetRef<Call>(call));
//   }
//   Array<Type> ret;
//   for (auto& tv: fn_ty->type_params) {
//     std::cout << "Resolved Type: " << solver.Resolve(subst_map[tv]) << std::endl;
//     ret.push_back(solver.Resolve(subst_map[tv])); 
//   }
//   return ret;
// }

class DefuncMutator : public ExprMutator {
 public:
  DefuncMutator(const IRModule& mod) : mod(mod), constructor_name(0), anon_name(0) {}

  Expr VisitExpr_(const CallNode* call) {
    if (auto op = call->op.as<GlobalVarNode>()) {
      CHECK(call->type_args.size() == op->checked_type().as<FuncTypeNode>()->type_params.size()) << "all type args must be explicit";

      auto op_type = InstFuncType(op->checked_type().as<FuncTypeNode>(), call->type_args);

      CHECK(!HasFuncType(op_type->ret_type)) << "returning functions not supported";
      if (!IsHigherOrderFunc(op_type)) {
        return ExprMutator::VisitExpr_(call);
      }
      auto name = op->name_hint + TypeToString(op_type);
      auto gv = GlobalVar(name);
      if (mod->ContainGlobalVar(name)) {
        gv = mod->GetGlobalVar(name);
      } else {
        // clone and specialize with specific type
        auto clone = DeDup(DeGlobal(mod, GetRef<GlobalVar>(op))).as<FunctionNode>();
        auto specialized_function = Specialize(clone, call->type_args);
        auto f = Downcast<Function>(this->VisitExpr(FirstifyVars(specialized_function)));
        mod->Add(gv, f);
      }

      Array<Expr> args;
      for (size_t i = 0; i < call->args.size(); i++) {
        auto arg = call->args[i];
        auto type = op_type->arg_types[i];
        // we assume arg is either an identifier or a function
        if (!HasFuncType(type)) {
          args.push_back(arg);
          continue;
        }

        CHECK(type.as<FuncTypeNode>()) << "assume no nested functions";

        if (arg.as<VarNode>()) {
          args.push_back(arg);
        }
        if (arg.as<GlobalVarNode>()) {
          args.push_back(EncodeGlobalVar(Downcast<GlobalVar>(arg), Downcast<FuncType>(type)));
        }
        if (arg.as<FunctionNode>()) {

        }
        CHECK(false) << "assume all first-order-parameters are identifiers or functions";
      }

    } else if (auto op = call->op.as<FunctionNode>()) {
      std::unordered_map<Var, Expr, ObjectHash, ObjectEqual> var_binding_map;
      for (size_t i = 0; i < op->params.size(); i++) {
        var_binding_map[op->params[i]] = call->args[i];
      } 
      auto e = Bind(op->body, var_binding_map);
      return this->VisitExpr(e);
    } else if (auto op = call->op.as<VarNode>()) {
      auto op_type = InstFuncType(var_save_type[GetRef<Var>(op)].as<FuncTypeNode>(), call->type_args);
      
      Array<Expr> args = {GetRef<Var>(op)};
      for (auto arg: call->args) {
        args.push_back(this->VisitExpr(arg));
      }

      auto e = Call(apply_map[op_type], args);
      return e;
    }
    return ExprMutator::VisitExpr_(call);
  }

 private:
  IRModule mod;
  // encode func type to ADT
  std::unordered_map<Type, GlobalTypeVar, ObjectHash, StructuralEqual> func_encoding;
  std::unordered_map<Type, GlobalVar, ObjectHash, StructuralEqual> apply_map;
  std::unordered_map<Var, Type, ObjectHash, StructuralEqual> var_save_type;
  std::unordered_map<GlobalVar, std::unordered_map<Type, Constructor, ObjectHash, StructuralEqual>, ObjectHash, ObjectEqual> gv_datatype_map;
  // use monotonically increasing integer to represent new constructor_name
  unsigned int constructor_name;
  unsigned int anon_name;

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

  Expr EncodeGlobalVar(const GlobalVar& gv, const FuncType& ft) {
    auto map = gv_datatype_map[gv];
    if (map.count(ft) == 0) {
      auto adt_name = "T" + TypeToString(ft);

      if (func_encoding.count(ft) == 0) {
        func_encoding[ft] = GlobalTypeVar(adt_name, TypeKind::kAdtHandle);
      }

      auto gtv = func_encoding[ft];
      auto c = Constructor(std::to_string(constructor_name++), {}, gtv);
      AddConstructor(gtv, c);

      if (apply_map.count(ft) == 0) {
        apply_map[ft] = GlobalVar("apply" + TypeToString(ft));
      }

      auto gv = apply_map[ft];
      AddApplyCase(gv, ft, c, gv);
    }
    
    auto c = map[ft];
    return Call(c, {});
  }
  
  std::string TypeToString(const Type& t) {
    std::ostringstream s;
    s << t;
    return s.str();
  }

  FuncType InstFuncType(const FuncTypeNode* fty, const Array<Type> type_args) {
    auto map = tvm::Map<TypeVar, Type>();
    for (size_t i = 0; i < type_args.size(); i++) {
      map.Set(fty->type_params[i], type_args[i]);
    }
    // copy with typevars removed
    return Downcast<FuncType>(TypeSubst(FuncType(fty->arg_types, fty->ret_type, {}, {}), map));
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

  Function FirstifyVars(const Function& f) {
    CHECK(f->type_params.size() == 0) << "firstify function has type params";

    std::unordered_map<Var, Expr, ObjectHash, ObjectEqual> var_bind_map;
    for (auto var: f->params) {
      if (auto var_type = var->checked_type().as<FuncTypeNode>()) {
        // first order parameter
        auto fop_type = GetRef<FuncType>(var_type);
        if (func_encoding.count(fop_type) == 0) {
          auto name = "T" + TypeToString(fop_type);
          func_encoding[fop_type] = GlobalTypeVar(name, TypeKind::kAdtHandle);
        }
        auto adt = func_encoding[fop_type];
        var_bind_map[var] = Var(var->name_hint(), TypeCall(adt, {}));
      } else {
        CHECK(!HasFuncType(var->checked_type())) << "nested function type in parameter not supported yet";
      }
    }

    return Downcast<Function>(Bind(f, var_bind_map));
  }
};

Expr Defunctionalization(const Expr& e, const IRModule& mod) {
  auto f = e.as<FunctionNode>();
  CHECK(f) << "input need to be a function";
  CHECK(f->type_params.size() == 0) << "no polymorphism supported for defunctionalization";
  for (const auto& p : f->params) {
    CHECK(!HasFuncType(p)) << "input parameters cannot have func type";
  }
  CHECK(!HasFuncType(f->ret_type)) << "return type cannot contain function";

  return DefuncMutator(mod).VisitExpr(e);
}

TVM_REGISTER_GLOBAL("relay._transform.Defunctionalization").set_body_typed(Defunctionalization);

}  // namespace relay
}  // namespace tvm
