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
 * \file to_cps.cc
 *
 * \brief Turn a program to continuation passing style.
 *
 * Given a fresh type variable 'answer',
 * continuation passing style(CPS) convert every function of a -> b to a -> (b -> anwer) -> answer.
 *
 * That is, instead of returning the result directly,
 * function will now call another function (called the continuation)
 * and return that value as a result instead.
 *
 * Continuation passing style turn all function call into tail call,
 * which bound the stack size, prevent stack from overflowing during recursion,
 * and allow tail call optimization.
 *
 * In relay, as tensor operation is the bottleneck,
 * CPS is currently intended to transform the program before partial eval (PE),
 * as it reify the control flow and enable PE to handle control flow join more agressively.
 *
 * For example, given 'let a = if b then c else d in e', it will transform the code into
 * 'let f a = e in if b then f c else f d'.
 * This allow f to be optimized individually in both branch.
 *
 * We implement CPS conversion by higher order transform
 * (see http://matt.might.net/articles/cps-conversion/).
 * The basic idea is that we will recursively traverse the AST.
 * During the traversal, there is an extra parameter, mcont, of expr -> expr.
 * It is basically a continuation at the metalevel.
 * All cases in the transform must return via the mcont,
 * wheter directly invoking it, or indirectly by recursion.
 */
#include <tvm/relay/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include "../ir/type_functor.h"
#include "let_list.h"
#include "pass_util.h"

namespace tvm {
namespace relay {

// we assume the data type has no closure - no idea how to look into datatype right now.

Type Arrow(const Type& l, const Type& r) {
  return FuncTypeNode::make({l}, r, {}, {});
}

Type CPSType(const Type& t, const TypeVar& answer);

FuncType CPSFuncType(const FuncType& f, const TypeVar& answer) {
  tvm::Array<Type> new_arg_types;
  for (const Type& t : f->arg_types) {
    new_arg_types.push_back(CPSType(t, answer));
  }
  new_arg_types.push_back(Arrow(CPSType(f->ret_type, answer), answer));
  return FuncTypeNode::make(new_arg_types, answer, f->type_params, f->type_constraints);
}

Type CPSType(const Type& t, const TypeVar& answer) {
  struct CPSTypeMutator : TypeMutator {
    explicit CPSTypeMutator(const TypeVar& answer) : answer(answer) { }
    TypeVar answer;
    Type VisitType_(const FuncTypeNode* t) final {
      return CPSFuncType(GetRef<FuncType>(t), answer);
    }
  } mut(answer);
  return mut(t);
}

// transform global functions into cps form.
using CPSMap = std::unordered_map<GlobalVar, GlobalVar, ObjectHash, ObjectEqual>;

// transform vars from the original program into new vars, so their type will be correct.
using VarMap = std::unordered_map<Var, Var, ObjectHash, ObjectEqual>;

/*
 * The meta continuation.
 * There is 3 rules on the metacontinuation:
 * 0: It can only use the argument once.
 *    The argument is code, and using it twice will duplicate code.
 *    Bound the argument via let instead.
 * 1: If the size of the metacontinuation is unbounded, it can only be called once.
 *    It contain code, so calling it twice duplicate code.
 *    Reify the continuation and bound it instead.
 *    See the function 'reify' and the if case for more detail.
 * 2: The argument must be effect free.
 *    It might reorder or drop the argument.
 *    Again, bound the argument via let instead.
 *    See the call case for more detail.
 */
using MCont = std::function<Expr(const Expr&)>;

Function ToCPS(const Function& f, const Module& m, CPSMap* cm);

Function ToCPS(const Function& f, const Module& m, CPSMap* cm, VarMap* vm, const TypeVar& answer) {
  std::function<Var(Var)> remap = [&](const Var& v) { return vm->count(v) == 0 ? v : vm->at(v); };
  auto function_type = Downcast<FuncType>(f->checked_type());
  // Each MCont can be used at most once.
  struct CPSFunctor : ExprFunctor<Expr(const Expr&, const MCont&)>, PatternMutator {
    CPSFunctor(const std::function<Var(Var)>& remap,
               const TypeVar& answer,
               const Module& m,
               VarMap* vm,
               CPSMap* cm) : remap(remap), answer(answer), m(m), vm(vm), cm(cm) { }
    const std::function<Var(Var)>& remap;
    TypeVar answer;
    Module m;
    VarMap* vm;
    CPSMap* cm;

    Expr VisitExpr_(const LetNode* op, const MCont& k) final {
      return VisitExpr(op->value, [&](const Expr& v) {
        return LetNode::make(remap(op->var), v, VisitExpr(op->body, k));
      });
    }

    Expr VisitExpr_(const FunctionNode* op, const MCont& k) final {
      CHECK(!op->IsPrimitive()) << "primitive func not supported yet.";
      return k(ToCPS(GetRef<Function>(op), m, cm, vm, answer));
    }

    Expr VisitExpr_(const ConstantNode* op, const MCont& k) final {
      return k(GetRef<Constant>(op));
    }

    Expr VisitExpr_(const VarNode* op, const MCont& k) final {
      return k(remap(GetRef<Var>(op)));
    }

    Pattern VisitPattern_(const PatternVarNode* op) final {
      return PatternVarNode::make(remap(op->var));
    }

    Expr VisitExpr_(const GlobalVarNode* op, const MCont& k) final {
      auto gv = GetRef<GlobalVar>(op);
      if (cm->count(gv) == 0) {
        auto cps_gv = GlobalVar(gv->name_hint + "_cps");
        cm->insert({gv, cps_gv});
        m->Add(cps_gv, ToCPS(m->Lookup(gv), m, cm));
      }
      return k(cm->at(gv));
    }

    Expr VisitExpr_(const RefCreateNode* op, const MCont& k) final {
      return VisitExpr(op->value, [&](const Expr& v) { return k(RefCreateNode::make(v)); });
    }

    Expr reify(const MCont& k) {
      Var arg = VarNode::make("arg", Type());
      return FunctionNode::make({arg}, k(arg), Type(), {}, {});
    }

    Expr reify(const MCont& k, const std::function<Expr(MCont)>& cont) {
      return LetList::Let(reify(k),
                          [&](const Var& f) {
        return cont([&](const Expr& e) { return CallNode::make(f, {e}); });
      });
    }

    Expr VisitExpr_(const IfNode* op, const MCont& k) final {
      return reify(k, [&](const MCont& kf) {
        return VisitExpr(op->cond,
                         [&](const Expr& v) {
          return IfNode::make(v, VisitExpr(op->true_branch, kf), VisitExpr(op->false_branch, kf));
        });
      });
    }

    Expr VisitExpr_(const MatchNode* op, const MCont& k) final {
      return reify(k, [&](const MCont& kf) {
        return VisitExpr(op->data, [&](const Expr& v) {
          tvm::Array<Clause> clauses;
          for (const auto& c : op->clauses) {
            clauses.push_back(ClauseNode::make(VisitPattern(c->lhs), VisitExpr(c->rhs, kf)));
          }
          return MatchNode::make(v, clauses, op->complete);
        });
      });
    }

    Expr VisitExpr_(const RefReadNode* op, const MCont& k) final {
      return VisitExpr(op->ref,
                       [&](const Expr& r) {
        return LetList::Let(RefReadNode::make(r), k);
      });
    }

    Expr VisitExpr_(const RefWriteNode* op, const MCont& k) final {
      return VisitExpr(op->ref,
                       [&](const Expr& r) {
        return VisitExpr(op->value,
                         [&](const Expr& v) {
          return LetList::Let(RefWriteNode::make(r, v), k);
        });
      });
    }

    Expr VisitExpr_(const TupleNode* op, const MCont& k) final {
      tvm::Array<Expr> fields;
      std::function<Expr()> next;
      next = [&]() {
        return (fields.size() == op->fields.size()) ?
          k(TupleNode::make(fields)) :
          VisitExpr(op->fields[fields.size()], [&](const Expr& v) {
            fields.push_back(v);
            return next();
          });
      };
      return next();
    }

    Expr VisitExpr_(const TupleGetItemNode* op, const MCont& k) final {
      return VisitExpr(op->tuple, [&](const Expr& v) {
        return k(TupleGetItemNode::make(v, op->index));
      });
    }

    Expr VisitExpr_(const CallNode* op, const MCont& k) final {
      if (op->op.as<OpNode>() || op->op.as<ConstructorNode>()) {
        tvm::Array<Expr> args;
        std::function<Expr()> next;
        next = [&]() {
          if (args.size() == op->args.size()) {
            return LetList::Let(CallNode::make(op->op, args, op->attrs, op->type_args), k);
          } else {
            return VisitExpr(op->args[args.size()], [&](const Expr& v) {
                args.push_back(v);
                return next();
              });
          }
        };
        return next();
      } else {
        Expr f;
        tvm::Array<Expr> args;
        std::function<Expr()> next;
        next = [&]() {
          if (args.size() == op->args.size()) {
            args.push_back(reify(k));
            return Expr(CallNode::make(f, args, op->attrs, op->type_args));
          } else {
            return VisitExpr(op->args[args.size()], [&](const Expr& v) {
              args.push_back(v);
              return next();
            });
          }
         };
        return VisitExpr(op->op, [&](const Expr& v) {
          f = v;
          return next();
        });
      }
    }
  } mut(remap, answer, m, vm, cm);
  Var k = VarNode::make("k", Arrow(CPSType(function_type->ret_type, answer), answer));
  tvm::Array<Var> new_params;
  for (const Var& v : f->params) {
    new_params.push_back(remap(v));
  }
  new_params.push_back(k);
  return FunctionNode::make(new_params,
                            mut.VisitExpr(f->body,
                                          [&](const Expr& e) { return CallNode::make(k, {e}); }),
                            answer,
                            f->type_params,
                            f->attrs);
}

Function ToCPS(const Function& f, const Module& m, CPSMap* cm) {
  TypeVar answer = TypeVarNode::make("answer", kType);
  VarMap var;
  struct Remapper : ExprVisitor, PatternVisitor {
    Remapper(const TypeVar& answer, VarMap* vm) : answer(answer), vm(vm) { }
    TypeVar answer;
    VarMap* vm;
    void VisitExpr_(const VarNode* vn) final {
      Var v = GetRef<Var>(vn);
      if (vm->count(v) == 0) {
        auto ret = VarNode::make(v->name_hint(), CPSType(v->checked_type(), answer));
        vm->insert({v, ret});
      }
    }

    void VisitPattern(const Pattern& p) final {
      PatternVisitor::VisitPattern(p);
    }

    void VisitPattern_(const PatternVarNode* op) final {
      VisitExpr(op->var);
    }
  } remap(answer, &var);
  remap.VisitExpr(f);
  Function ret = ToCPS(f, m, cm, &var, answer);
  auto new_type_params = ret->type_params;
  new_type_params.push_back(answer);
  return FunctionNode::make(ret->params, ret->body, ret->ret_type, new_type_params, ret->attrs);
}

Function ToCPS(const Function& f, const Module& m) {
  CPSMap cps;
  return ToCPS(f, m, &cps);
}

Function UnCPS(const Function& f) {
  CHECK_GT(f->params.size(), 0);
  std::vector<Var> new_params;
  for (const auto& p : f->params) {
    new_params.push_back(VarNode::make(p->name_hint(), p->checked_type()));
  }
  auto cont_type = Downcast<FuncType>(new_params.back()->type_annotation);
  new_params.pop_back();
  CHECK_EQ(cont_type->arg_types.size(), 1);
  auto new_ret_type = Type(cont_type->arg_types[0]);
  std::vector<TypeVar> new_type_params;
  for (const auto& tp : f->type_params) {
    new_type_params.push_back(TypeVarNode::make(tp->name_hint, tp->kind));
  }
  auto answer_type = new_type_params.back();
  new_type_params.pop_back();
  // TODO(@M.K.): make alphaequal work on free term
  // CHECK(AlphaEqual(cont_type, Arrow(new_ret_type, answer_type)));
  auto x = VarNode::make("x", new_ret_type);
  auto cont = FunctionNode::make({x}, x, new_ret_type, {}, {});
  tvm::Array<Expr> args;
  for (const auto& p : new_params) {
    args.push_back(p);
  }
  args.push_back(cont);
  tvm::Array<Type> type_args;
  for (const auto& tp : new_type_params) {
    type_args.push_back(tp);
  }
  type_args.push_back(new_ret_type);
  return FunctionNode::make(new_params,
                            CallNode::make(f, args, {}, type_args),
                            new_ret_type,
                            new_type_params,
                            f->attrs);
}

TVM_REGISTER_GLOBAL("relay._transform.to_cps")
.set_body_typed(static_cast<Function (*)(const Function&, const Module&)>(ToCPS));

TVM_REGISTER_GLOBAL("relay._transform.un_cps")
.set_body_typed(UnCPS);

namespace transform {

Pass ToCPS() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
    return Function(ToCPS(f, m));
  };
  return CreateFunctionPass(pass_func, 1, "ToCPS", {});
}

TVM_REGISTER_GLOBAL("relay._transform.ToCPS")
.set_body_typed(ToCPS);


Pass UnCPS() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
      return Function(UnCPS(f));
    };
  return CreateFunctionPass(pass_func, 1, "UnCPS", {});
}

TVM_REGISTER_GLOBAL("relay._transform.UnCPS")
.set_body_typed(UnCPS);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
