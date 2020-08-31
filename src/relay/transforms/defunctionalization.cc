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

class DefuncMutator : public ExprMutator {
 public:
  DefuncMutator(const IRModule& mod) : mod(mod), constructor_name(0) {}

  Expr VisitExpr_(const CallNode* op) {
    auto op_func = op->op;
    auto f = op_func.as<FunctionNode>();
    std::cout << op_func << std::endl;
    CHECK(f) << "only calls to functions are supported so far";
    
    // clone function and specialize if there are higher order functions
    if (IsHigherOrderFunc(Downcast<FuncType>(f->checked_type()))) {
      auto f_clone = Downcast<Function>(Clone(f, op->type_args));
      std::cout << f_clone << std::endl;
      auto f_clone_type = Downcast<FuncType>(f_clone->checked_type());
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
        }

        CHECK(!HasFuncType(f_clone_type->arg_types[i])) << "nested function type in parameter not supported yet";
        args.push_back(op->args[i]);
      }

      auto new_func = ApplyVars(f_clone, applyVars);

      return Call(ExprMutator::VisitExpr(new_func), args);
    }
    return ExprMutator::VisitExpr(GetRef<Call>(op));
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

  Expr VisitExpr_(const GlobalVarNode* op) { CHECK(false) << "global var not supported yet"; 
    throw std::runtime_error("GlobalVar not supported");
  }

 private:
  IRModule mod;
  // encode func type to ADT
  std::unordered_map<Type, GlobalTypeVar, ObjectHash, StructuralEqual> func_encoding;
  std::unordered_map<Type, GlobalVar, ObjectHash, StructuralEqual> apply_map;
  // use monotonically increasing integer to represent new constructor_name
  unsigned int constructor_name;

  Expr ApplyVars(Expr body, const std::unordered_map<Var, GlobalVar, ObjectHash, ObjectEqual>& vars) {
    struct ApplyVarMutator: public ExprMutator {
      std::unordered_map<Var, GlobalVar, ObjectHash, ObjectEqual> vars;
      ApplyVarMutator(const std::unordered_map<Var, GlobalVar, ObjectHash, ObjectEqual>& vars) : vars(vars) {}
      Expr VisitExpr_(const CallNode* op) {
        if (auto var_op = op->op.as<VarNode>()) {
          if (vars.count(GetRef<Var>(var_op)) != 0) {
            auto gv = vars[GetRef<Var>(var_op)];
            Array<Expr> args = {GetRef<Var>(var_op)};
            for (auto arg: op->args) {
              args.push_back(arg);
            }
            return Call(gv, args);
          }
        }

        return ExprMutator::VisitExpr_(op);
      }
    };

    return ApplyVarMutator(vars).Mutate(body);
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

  void AddApplyCase(GlobalVar gv, FuncType ft, Constructor c) {
    if (!mod->ContainGlobalVar(gv->name_hint)) {
      auto x = Var("x", func_encoding[ft]);
      auto vars = Array<Var>({x});
      auto args = Array<Expr>();
      for (auto t: ft->arg_types) {
        auto y = Var("y", t);
        vars.push_back(y);
        args.push_back(y);
      }


      auto clauses = Array<Clause>({Clause(PatternConstructor(c, {}), Call(x, args))});
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
      clauses.push_back(Clause(PatternConstructor(c, {}), Call(x, args)));

      mod->Add(gv, Function(f->params, Match(x, clauses), f->ret_type, f->type_params), true);
    }
  }

  Expr EncodeFunctionArg(const Expr& f, const FuncTypeNode* ft) {
    if (func_encoding.count(GetRef<FuncType>(ft)) == 0) {
      func_encoding[GetRef<FuncType>(ft)] = GlobalTypeVar("T" + TypeToString(ft), TypeKind::kAdtHandle);
    }

    auto gtv = func_encoding[GetRef<FuncType>(ft)];
    auto c = Constructor(std::to_string(constructor_name++), {}, gtv);
    AddConstructor(gtv, c);

    if (apply_map.count(GetRef<FuncType>(ft)) == 0) {
      apply_map[GetRef<FuncType>(ft)] = GlobalVar("apply" + TypeToString(ft));
    }

    auto gv = apply_map[GetRef<FuncType>(ft)];
    AddApplyCase(gv, GetRef<FuncType>(ft), c);

    return Call(c, {});
  }
  
  std::string TypeToString(const TypeNode* t) {
    std::ostringstream s;
    s << t;
    return s.str();
  }

  Expr Clone(const FunctionNode* f, const Array<Type> type_args) {
    return DeDup(Specialize(f, type_args));
  }

  Expr Specialize(const FunctionNode* f, const Array<Type> type_args) {
    auto map = tvm::Map<TypeVar, Type>();
    for (size_t i = 0; i < type_args.size(); i++) {
      map.Set(f->type_params[i], type_args[i]);
    }
    return TypeSubst(GetRef<Function>(f), map);
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
