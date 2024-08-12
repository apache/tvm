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
 * \file higher_order_gradient.cc
 * \brief Higher-order Automatic Differentiation in Relay IR, for non-graph programs.
 */
#include <tvm/ir/type_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/feature.h>
#include <tvm/relay/transform.h>
#include <tvm/te/operation.h>

#include "gradient.h"
#include "let_list.h"
#include "pass_utils.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

/*! What is automatic differentiation(AD) and why is it important?
 * By AD, we roughly mean, given a term which denotes some mathematical function,
 * derive a term which denotes the derivative of that mathematical function.
 * Such a method can be compile-time, which is a macro on completely known function.
 * Formally speaking, such requirement mean that the input function is a closed expression -
 * that is, it only refer to local variable that is it's parameter, or defined inside it.
 * Every top level definition satisfy this criteria.
 * AD can also be run-time, which mean it is merely a function term of AD : (Float[] -> Float[]) ->
 * (Float[] -> Float[]). In relay we currently only support compile-time AD, but it should be enough
 * for a lot of use case.
 *
 * In deep learning, the most common way to train a deep neural network is by gradient descent or
 * some of it's variant. Such optimization method require us to input the gradient of neural
 * network, which can be obtained easily using AD. In fact, back propagation is essentially
 * reverse-mode automatic differentiation, a kind of AD!
 */

/*! In relay, automatic differentiation(AD) is a macro,
 *  that transform closed expr(expr without free variable/free type variable) of type
 *  (x0, x1, x2, ...) -> Float[] to
 *  (x0, x1, x2, ...) -> (Float[], (x0, x1,  x2, ...)),
 *  When x0, x1, x2... are Float of different shape.
 * the return value is a pair, with left hand side as the original value, and right hand side as
 * gradient of the input. WithGradientType will take the type of input, and produce the type of
 * output. There are multiple implementation of AD in relay, with different characteristic. However,
 * they all transform the input expr according to WithGradientType.
 */
Type WithGradientType(const Type& t) {
  // TODO(@M.K.): stricter checking
  auto ty = t.as<FuncTypeNode>();
  ICHECK(ty) << "input should be a function";
  return FuncType(ty->arg_types, TupleType({ty->ret_type, TupleType(ty->arg_types)}), {}, {});
}

//! \brief if the expression is a GlobalVar, transform to it's expression.
Expr DeGlobal(const Optional<IRModule>& mod, const Expr& e) {
  const auto* x = e.as<GlobalVarNode>();

  if (mod.defined() && x) {
    BaseFunc base_func = mod.value()->Lookup(GetRef<GlobalVar>(x));
    if (auto func = base_func.as<Function>()) {
      return func.value();
    } else {
      return e;
    }
  } else {
    return e;
  }
}

static Type bpt = RelayRefType(FuncType({}, TupleType(Array<Type>()), {}, {}));

struct ReverseADType : TypeMutator {
  Type VisitType_(const TensorTypeNode* ttn) final {
    Type t = GetRef<Type>(ttn);
    return TupleType({t, RelayRefType(t)});
  }

  Type VisitType_(const FuncTypeNode* ftn) final {
    std::vector<Type> arg_types;
    for (const auto& t : ftn->arg_types) {
      arg_types.push_back(VisitType(t));
    }
    arg_types.push_back(bpt);
    return FuncType(arg_types, ftn->ret_type, ftn->type_params, ftn->type_constraints);
  }
};

Type ReverseType(const Type& t) { return ReverseADType()(t); }

/*! \brief Lift a function that transform Tensor to a function that also transform more type
 * by doing a structure preserving map.
 */
Expr LiftTensor(const std::function<Expr(const Expr& t)>& f,
                const std::function<Type(const Type&)>& tf, const Type& forward_type, const Expr& e,
                LetList* ll) {
  ICHECK(IsAtomic(e)) << e;
  if (forward_type.as<TensorTypeNode>()) {
    auto ret = ll->Push(f(e));
    ret->checked_type_ = tf(forward_type);
    return std::move(ret);
  } else if (auto* tt = forward_type.as<TupleTypeNode>()) {
    tvm::Array<Expr> fields;
    tvm::Array<Type> types;
    for (size_t i = 0; i < tt->fields.size(); ++i) {
      auto field = LiftTensor(f, tf, tt->fields[i], ll->Push(GetField(e, i)), ll);
      fields.push_back(field);
      types.push_back(field->checked_type_);
    }
    auto ret = ll->Push(Tuple(fields));
    ret->checked_type_ = TupleType(types);
    return std::move(ret);
  } else {
    LOG(FATAL) << "unsupported input/output type: " << tt;
    throw;
  }
}

/*! \brief Transfers the gradients from an Expr to a deep duplication of the Expr,
 * by stitching the references in the AD values.
 */
void TransferGrads(const Type& forward_type, const Expr& from, const Expr& to, LetList* ll) {
  ICHECK(IsAtomic(from)) << from;
  ICHECK(IsAtomic(to)) << to;
  if (forward_type.as<TensorTypeNode>()) {
    auto from_ref = TupleGetItem(from, 1);
    auto to_ref = TupleGetItem(to, 1);
    ll->Push(RefWrite(to_ref, RefRead(from_ref)));
  } else if (auto* tt = forward_type.as<TupleTypeNode>()) {
    for (size_t i = 0; i < tt->fields.size(); ++i) {
      TransferGrads(tt->fields[i], ll->Push(TupleGetItem(from, i)), ll->Push(TupleGetItem(to, i)),
                    ll);
    }
  } else {
    LOG(FATAL) << "Unsupported input/output type: " << forward_type;
    throw;
  }
}

// TODO(@M.K.): why take Expr?
/*! \brief t -> ReverseType(t). Transform to Reverse Mode Value. */
Expr GetRev(const Type& forward_type, const Expr& e, LetList* ll) {
  auto rev = [&](const Expr& e) { return Pair(e, RefCreate(ZerosLike(e))); };
  auto rev_type = [&](const Type& forward_type) { return ReverseType(forward_type); };
  return LiftTensor(rev, rev_type, forward_type, e, ll);
}

/*! \brief ReverseType(t) -> t. Get the original value. */
Expr GetValue(const Type& forward_type, const Expr& e, LetList* ll) {
  auto val = [&](const Expr& e) { return GetField(e, 0); };
  auto val_type = [&](const Type& forward_type) { return forward_type; };
  return LiftTensor(val, val_type, forward_type, e, ll);
}

/*! \brief ReverseType(t) -> t. Get the gradient. */
Expr GetGrad(const Type& forward_type, const Expr& e, LetList* ll) {
  auto grad = [&](const Expr& e) { return RefRead(GetField(e, 1)); };
  auto grad_type = [&](const Type& forward_type) { return forward_type; };
  return LiftTensor(grad, grad_type, forward_type, e, ll);
}

void UpdateGrad(const Type& t, const Expr& arg, const Expr& grad, LetList* ll) {
  if (t.as<TensorTypeNode>()) {
    ll->Push(RefWrite(GetField(arg, 1), Add(RefRead(GetField(arg, 1)), grad)));
  } else if (auto* tt = t.as<TupleTypeNode>()) {
    for (size_t i = 0; i < tt->fields.size(); ++i) {
      UpdateGrad(tt->fields[i], ll->Push(GetField(arg, i)), ll->Push(GetField(grad, i)), ll);
    }
  } else {
    LOG(FATAL) << "unsupported arg type of operator: " << t;
    throw;
  }
}

Expr BPEmpty() {
  Expr unitF = Function({}, Tuple(tvm::Array<Expr>({})), TupleType::Empty(), {});
  return RefCreate(unitF);
}

struct ReverseAD : ExprMutator {
  using ADVarMap = std::unordered_map<Var, Var, ObjectPtrHash, ObjectPtrEqual>;
  using ADGlobalVarMap = std::unordered_map<GlobalVar, GlobalVar, ObjectPtrHash, ObjectPtrEqual>;
  Optional<IRModule> mod;
  // TODO(@M.K.) refactor AD to always use mod.
  Var bp;
  std::shared_ptr<ADVarMap> ad_vars;
  std::shared_ptr<ADGlobalVarMap> ad_gvars;
  const OpAttrMap<FPrimalGradient> rev_map = Op::GetAttrMap<FPrimalGradient>("FPrimalGradient");

  explicit ReverseAD(const Optional<IRModule>& mod, const Var& bp,
                     const std::shared_ptr<ADVarMap>& ad_vars,
                     const std::shared_ptr<ADGlobalVarMap>& ad_gvars)
      : mod(mod), bp(bp), ad_vars(ad_vars), ad_gvars(ad_gvars) {}

  Expr VisitExpr_(const OpNode* op) final {
    LOG(FATAL) << "op should only be inside call";
    throw;
  }

  Expr Remap(const Expr& e) {
    struct Remapper : ExprMutator {
      std::shared_ptr<ADVarMap> ad_vars;
      LetList* ll;
      Remapper(const std::shared_ptr<ADVarMap>& ad_vars, LetList* ll) : ad_vars(ad_vars), ll(ll) {}
      Expr VisitExpr_(const VarNode* var) final {
        // memoize Var -> ADVar so we don't end up with free Vars when checkpointing
        auto var_ref = GetRef<Var>(var);
        if (ad_vars->count(var_ref) == 0) {
          return std::move(var_ref);
        } else {
          return GetValue(var_ref->checked_type(), ad_vars->at(var_ref), ll);
        }
      }
    };
    return LetList::With([&](LetList* ll) { return Remapper(ad_vars, ll)(e); });
  }

  Expr VisitCheckpoint(const CallNode* call) {
    auto optional = call->op.as<Op>();
    ICHECK(optional) << "expected op in call";
    Op op_ref = optional.value();
    ICHECK(op_ref->name == "annotation.checkpoint") << "expected checkpoint annotation";
    auto x = call->args[0];
    return LetList::With([&](LetList* ll) {
      auto x_var = ll->Push(Remap(x));
      auto ret = ll->Push(GetRev(call->checked_type(), x_var, ll));
      auto bpv = ll->Push(RefRead(bp));
      Expr nbp = Function({}, LetList::With([&](LetList* ll) {
                            // we need a new ReverseAD visitor to avoid clobbering the bp local var
                            auto dup_bp = ll->Push(BPEmpty());
                            auto dup_ad =
                                ll->Push(ReverseAD(mod, dup_bp, ad_vars, ad_gvars)(DeDup(x)));
                            TransferGrads(call->checked_type(), ret, dup_ad, ll);
                            ll->Push(Call(RefRead(dup_bp), {}));
                            return Call(bpv, {});
                          }),
                          TupleType::Empty(), {});
      ll->Push(RefWrite(bp, nbp));
      return ret;
    });
  }

  Expr VisitExpr_(const CallNode* call) final {
    if (auto optional = call->op.as<Op>()) {
      Op op_ref = optional.value();

      if (op_ref->name == "annotation.checkpoint") {
        return VisitCheckpoint(call);
      }

      ICHECK(rev_map.count(op_ref)) << op_ref->name << " does not have reverse mode defined";
      return LetList::With([&](LetList* ll) {
        std::vector<Var> args;
        for (const auto& arg : call->args) {
          args.push_back(ll->Push(VisitExpr(arg)));
        }
        std::vector<Expr> orig_args;
        for (size_t i = 0; i < args.size(); i++) {
          orig_args.push_back(GetValue(call->args[i]->checked_type(), args[i], ll));
        }
        Expr orig = Call(call->op, orig_args, call->attrs, call->type_args);
        orig->checked_type_ = call->checked_type();
        Var orig_var = ll->Push(orig);
        orig_var->checked_type_ = call->checked_type();
        auto ret = ll->Push(GetRev(call->checked_type(), orig_var, ll));
        auto bpv = ll->Push(RefRead(bp));
        Expr nbp_body = LetList::With([&](LetList* ll) {
          tvm::Array<Expr> rev = rev_map[op_ref](orig, GetGrad(call->checked_type(), ret, ll));
          ICHECK(args.size() == rev.size());
          for (size_t i = 0; i < args.size(); ++i) {
            UpdateGrad(call->args[i]->checked_type(), args[i], rev[i], ll);
          }
          return Call(bpv, {});
        });
        Expr nbp = Function({}, nbp_body, TupleType::Empty(), {});
        ll->Push(RefWrite(bp, transform::ToANormalForm(nbp)));
        // TODO(@M.K.): ToANF should be called on rev. Enhance ToANF for that.
        return ret;
      });
    } else if (call->op.as<ConstructorNode>()) {
      return ExprMutator::VisitExpr_(call);
    } else {
      std::vector<Expr> args;
      for (const auto& arg : call->args) {
        args.push_back(VisitExpr(arg));
      }
      args.push_back(bp);
      return Call(VisitExpr(call->op), args);
    }
  }

  Expr VisitExpr_(const ConstantNode* op) final {
    return LetList::With([&](LetList* ll) {
      Expr e = ll->Push(GetRef<Expr>(op));
      return Pair(e, RefCreate(ZerosLike(e)));
    });
  }

  Expr VisitExpr_(const IfNode* op) final {
    return If(TupleGetItem(VisitExpr(op->cond), 0), VisitExpr(op->true_branch),
              VisitExpr(op->false_branch));
  }

  Expr VisitExpr_(const VarNode* var) final {
    // memoize Var -> ADVar so we don't end up with free Vars when checkpointing
    auto var_ref = GetRef<Var>(var);
    if (ad_vars->count(var_ref) == 0) {
      auto res = Downcast<Var>(ExprMutator::VisitExpr_(var));
      (*ad_vars)[var_ref] = res;
    }

    return ad_vars->at(var_ref);
  }

  Expr VisitExpr_(const GlobalVarNode* op) final {
    // todo: concatenating string to add attribute seems like a brittle hack.
    // maybe get module indexed by a rose tree of string?
    ICHECK(mod.defined());
    auto orig_gv = GetRef<GlobalVar>(op);
    if (ad_gvars->count(orig_gv) == 0) {
      GlobalVar gv(op->name_hint + "_grad");
      (*ad_gvars)[orig_gv] = gv;
      Function orig_f = Downcast<Function>(DeDup(mod.value()->Lookup(orig_gv)));
      Array<Var> params;
      for (const auto& p : orig_f->params) {
        params.push_back(Downcast<Var>(VisitExpr(p)));
      }
      params.push_back(bp);
      Function f = WithFields(orig_f, params, VisitExpr(orig_f->body), VisitType(orig_f->ret_type));
      std::cout << "gv " << op->name_hint << ": " << AsText(f, false) << std::endl;
      mod.value()->Add(gv, f);
    }
    return ad_gvars->at(orig_gv);
  }

  Expr VisitExpr_(const FunctionNode* func_node) final {
    Array<Var> params;
    for (const auto& var : func_node->params) {
      params.push_back(Downcast<Var>(VisitExpr(var)));
    }
    auto new_bp = Var("bp", bpt);
    params.push_back(new_bp);
    return WithFields(GetRef<Function>(func_node), params,
                      ReverseAD(mod, new_bp, ad_vars, ad_gvars)(func_node->body),
                      VisitType(func_node->ret_type));
  }

  Type VisitType(const Type& t) final { return t.defined() ? ReverseType(t) : t; }
};

bool MissingGrad(const Expr& e) {
  struct MGVisitor : ExprVisitor {
    const OpAttrMap<FPrimalGradient> rev_map = Op::GetAttrMap<FPrimalGradient>("FPrimalGradient");
    std::unordered_set<std::string> op_names;

    void VisitExpr_(const OpNode* op) final {
      Op op_ref = GetRef<Op>(op);
      if (op_ref->name != "annotation.checkpoint" && !rev_map.count(op_ref)) {
        op_names.insert(op_ref->name);
      }
      ExprVisitor::VisitExpr_(op);
    }
  };

  MGVisitor mg;
  mg.VisitExpr(e);

  if (mg.op_names.size() > 0) {
    LOG(WARNING) << "found operators with missing gradients:";
    for (const auto& op : mg.op_names) {
      LOG(WARNING) << "    " << op;
    }
    return true;
  }

  return false;
}

Expr Gradient(const Expr& re, const Optional<IRModule>& mod) {
  CheckFeature(re, FeatureSet::All() - fGraph);
  if (mod.defined()) {
    CheckFeature(mod.value(), FeatureSet::All() - fGraph);
  }
  auto e = DeGlobal(mod, re);
  auto f = e.as<FunctionNode>();
  ICHECK(f) << "input need to be a function";
  ICHECK(f->type_params.size() == 0) << "no polymorphism supported for now";
  for (const auto& p : f->params) {
    ICHECK(p->checked_type().as<TensorTypeNode>()) << "input parameters need to be tensor";
  }
  ICHECK(!MissingGrad(e)) << "input has operators with missing gradients";
  Expr body = LetList::With([&](LetList* ll) {
    Var bp = ll->Push(BPEmpty(), bpt);
    Expr rev = ReverseAD(mod, bp, std::make_shared<ReverseAD::ADVarMap>(),
                         std::make_shared<ReverseAD::ADGlobalVarMap>())(e);
    std::vector<Expr> normal_args, args;
    for (const auto& p : f->params) {
      auto x = ll->Push(Pair(p, RefCreate(ZerosLike(p))));
      normal_args.push_back(x);
      args.push_back(x);
    }
    args.push_back(bp);
    auto c = ll->Push(Call(rev, args));
    std::function<void(const Expr&, const Type&)> init_grad;
    init_grad = [&](const Expr& e, const Type& t) {
      if (t.as<TensorTypeNode>()) {
        ll->Push(RefWrite(GetField(e, 1), OnesLike(GetField(e, 0))));
      } else if (auto tt = t.as<TupleTypeNode>()) {
        ICHECK_GT(tt->fields.size(), 0);
        init_grad(ll->Push(GetField(e, 0)), tt->fields[0]);
      } else {
        LOG(FATAL) << "unhandled type " << t;
        throw;
      }
    };
    init_grad(c, f->body->checked_type());
    ll->Push(Call(RefRead(bp), {}));
    std::vector<Expr> ret;
    for (const auto& a : normal_args) {
      ret.push_back(RefRead(GetField(a, 1)));
    }
    std::function<Expr(const Expr&, const Type&)> get_final_result;
    get_final_result = [&](const Expr& e, const Type& t) -> Expr {
      if (t.as<TensorTypeNode>()) {
        return GetField(e, 0);
      } else if (auto tt = t.as<TupleTypeNode>()) {
        tvm::Array<Expr> fields;
        for (size_t i = 0; i < tt->fields.size(); ++i) {
          fields.push_back(get_final_result(ll->Push(GetField(e, i)), tt->fields[i]));
        }
        return Tuple(fields);
      } else {
        LOG(FATAL) << "unhandled type " << t;
        throw;
      }
    };
    return Pair(get_final_result(c, f->body->checked_type()), Tuple(ret));
  });
  Function ret = WithFields(GetRef<Function>(f), f->params, body, GradRetType(GetRef<Function>(f)),
                            /* erase type params */ Array<TypeVar>());
  CheckFeature(ret, FeatureSet::All() - fGraph);
  return std::move(ret);
}

TVM_REGISTER_GLOBAL("relay._transform.gradient").set_body_typed(Gradient);

}  // namespace relay
}  // namespace tvm
