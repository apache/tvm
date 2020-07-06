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
 * \file gradient.cc
 * \brief API for Automatic Differentiation for the Relay IR.
 */
#include <tvm/ir/type_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/te/operation.h>

#include "let_list.h"
#include "pass_util.h"
#include "pattern_util.h"

namespace tvm {
namespace relay {

using namespace tvm::runtime;

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
Type WithGradientType(const Type&);

/*! return an expression that represent differentiation of e (according to WithGradientType).
 *  This version only work on first order code without control flow.
 */
Expr FirstOrderGradient(const Expr& e, const Optional<IRModule>& mod);

Type WithGradientType(const Type& t) {
  // TODO(M.K.): stricter checking
  auto ty = t.as<FuncTypeNode>();
  CHECK(ty) << "input should be a function";
  return FuncType(ty->arg_types, TupleType({ty->ret_type, TupleType(ty->arg_types)}), {}, {});
}

//! \brief if the expression is a GlobalVar, transform to it's expression.
Expr DeGlobal(const Optional<IRModule>& mod, const Expr& e) {
  const auto* x = e.as<GlobalVarNode>();

  if (mod.defined() && (x)) {
    BaseFunc base_func = mod.value()->Lookup(GetRef<GlobalVar>(x));
    if (auto* n = base_func.as<FunctionNode>()) {
      return n->body;
    } else {
      return e;
    }
  } else {
    return e;
  }
}

/*! \brief A fragment of the program being built by the automatic differentation
 *  pass.
 */
struct ADValueNode {
  virtual ~ADValueNode() {}
  template <typename T>
  T& get() {
    auto ret = dynamic_cast<T*>(this);
    CHECK(ret) << "cannot downcast";
    return *ret;
  }
};

template <typename F>
Expr MultiFactory(const Type& t, F factory) {
  if (auto* tt = t.as<TensorTypeNode>()) {
    return factory(tt->shape, tt->dtype);
  } else if (auto* tt = t.as<TupleTypeNode>()) {
    std::vector<Expr> res;
    for (size_t i = 0; i < tt->fields.size(); i++) {
      res.push_back(MultiFactory(tt->fields[i], factory));
    }
    return Tuple(res);
  } else {
    LOG(FATAL) << "unsupported type to create tensors of: " << tt;
    throw;
  }
}

template <typename F, typename F2>
Expr MultiFactoryLike(const Expr& e, const Type& t, F factory, F2 factory_like) {
  if (t.as<TensorTypeNode>()) {
    return factory_like(e);
  } else if (auto* tt = t.as<TupleTypeNode>()) {
    return MultiFactory(t, factory);
  } else {
    LOG(FATAL) << "unsupported type to tensors of: " << tt;
    throw;
  }
}

using ADValue = std::shared_ptr<ADValueNode>;

/*! \brief AD over a program which generates a tensor output. */
struct ADTensor : ADValueNode {
  Expr forward;
  mutable Expr reverse;  // must be a variable to avoid duplication
  ADTensor(LetList* ll, const Expr& forward)
      : forward(ll->Push(forward)),
        reverse(
            ll->Push(MultiFactoryLike(this->forward, forward->checked_type(), Zeros, ZerosLike))) {
    this->forward->checked_type_ = forward->checked_type();
  }
};

/*! \brief A staged representation of the program, we reflect
 * Relay functions into a function over fragments of AD. We
 * can compute away this function to obtain a reverse mode program.
 */
struct ADFunction : ADValueNode {
  std::function<ADValue(const Type&, const std::vector<ADValue>&, const Attrs&,
                        const tvm::Array<Type>&)>
      func;
  explicit ADFunction(const std::function<ADValue(const Type&, const std::vector<ADValue>&,
                                                  const Attrs&, const tvm::Array<Type>&)>& func)
      : func(func) {}
};

struct FirstOrderReverseAD : ExprFunctor<ADValue(const Expr&)> {
  using TBase = ExprFunctor<ADValue(const Expr&)>;
  const OpAttrMap<FPrimalGradient> rev_map = Op::GetAttrMap<FPrimalGradient>("FPrimalGradient");
  std::vector<std::function<void(LetList* ll)>> backprop_actions;
  // we assume no closure so no need for lexical scoping
  std::unordered_map<Expr, ADValue, ObjectPtrHash, ObjectPtrEqual> env;
  LetList* ll;

  FirstOrderReverseAD(LetList* ll) : ll(ll) {}

  ADValue VisitExpr(const Expr& n) final {
    if (env.count(n)) {
      return env.at(n);
    }
    auto ret = TBase::VisitExpr(n);
    env[n] = ret;
    return ret;
  }

  ADValue VisitExpr_(const OpNode* op) final {
    Op op_ref = GetRef<Op>(op);
    CHECK(rev_map.count(op_ref)) << op->name << " does not have reverse mode defined";
    return std::make_shared<ADFunction>(
        [this, op_ref](const Type& orig_type, const std::vector<ADValue>& args, const Attrs& attrs,
                       const tvm::Array<Type>& type_args) {
          std::vector<Expr> call_args;
          for (const ADValue& adval : args) {
            call_args.push_back(adval->get<ADTensor>().forward);
          }
          auto orig = Call(op_ref, call_args, attrs, type_args);
          orig->checked_type_ = orig_type;
          auto ret = std::make_shared<ADTensor>(ll, orig);
          backprop_actions.push_back([this, args, orig, ret, op_ref](LetList* ll) {
            tvm::Array<Expr> rev = rev_map[op_ref](orig, ret->reverse);
            CHECK(args.size() == rev.size());
            for (size_t i = 0; i < args.size(); ++i) {
              args[i]->get<ADTensor>().reverse =
                  ll->Push(Add(args[i]->get<ADTensor>().reverse, rev[i]));
            }
          });
          return ret;
        });
  }

  ADValue VisitExpr_(const TupleGetItemNode* op) final {
    Expr e = GetRef<Expr>(op);
    ADValue tup = VisitExpr(op->tuple);
    auto tt = op->tuple->checked_type().as<TupleTypeNode>();
    size_t size = tt->fields.size();
    size_t idx = op->index;
    auto ret = std::make_shared<ADTensor>(ll, e);
    backprop_actions.push_back([tup, idx, size, ret](LetList* ll) {
      auto rev = tup->get<ADTensor>().reverse;
      // special-case Tuple, to avoid long chains of GetItem/Tuple,
      // but we might have functions using tuples, so we don't know
      // that the reverse node is always a tuple
      std::vector<Expr> grfields;
      if (auto tup_node = rev.as<TupleNode>()) {
        for (size_t i = 0; i < size; ++i) {
          grfields.push_back(i != idx ? tup_node->fields[i]
                                      : Add(tup_node->fields[i], ret->reverse));
        }
      } else {
        for (size_t i = 0; i < size; ++i) {
          grfields.push_back(i != idx ? TupleGetItem(rev, i)
                                      : Add(TupleGetItem(rev, i), ret->reverse));
        }
      }
      tup->get<ADTensor>().reverse = ll->Push(Tuple(grfields));
    });
    return ret;
  }

  ADValue VisitExpr_(const TupleNode* op) final {
    Expr e = GetRef<Expr>(op);
    std::vector<ADValue> fields;
    for (const auto& f : op->fields) {
      fields.push_back(VisitExpr(f));
    }
    auto ret = std::make_shared<ADTensor>(ll, e);
    backprop_actions.push_back([fields, ret](LetList* ll) {
      for (size_t i = 0; i < fields.size(); ++i) {
        fields[i]->get<ADTensor>().reverse =
            ll->Push(Add(fields[i]->get<ADTensor>().reverse, TupleGetItem(ret->reverse, i)));
      }
    });
    return ret;
  }

  ADValue VisitExpr_(const ConstantNode* op) final {
    Expr e = GetRef<Expr>(op);
    return std::make_shared<ADTensor>(ll, e);
  }

  ADValue VisitExpr_(const CallNode* op) final {
    ADValue f = VisitExpr(op->op);
    std::vector<ADValue> args;
    for (const auto& arg : op->args) {
      args.push_back(VisitExpr(arg));
    }
    return f->get<ADFunction>().func(op->checked_type(), args, op->attrs, op->type_args);
  }

  ADValue VisitExpr_(const FunctionNode* op) final {
    Function f = GetRef<Function>(op);
    // todo: assert no closure
    return std::make_shared<ADFunction>(
        [this, f](const Type& orig_type, const std::vector<ADValue>& args, const Attrs& attrs,
                  const tvm::Array<Type>& type_args) {
          CHECK_EQ(f->params.size(), args.size());
          for (size_t i = 0; i < f->params.size(); ++i) {
            env[f->params[i]] = args[i];
          }
          return VisitExpr(f->body);
        });
  }

  // Var will always be in env, handled in VisitExpr (without _), so we don't need
  // to implement its VisitExpr_.
};

Type GradRetType(const Function& f) {
  // if type annotations are provided, we will construct a ret type;
  // otherwise, leave it to be inferred
  if (!f->ret_type.defined()) {
    return Type();
  }
  std::vector<Type> vt;
  for (const auto& p : f->params) {
    if (!p->type_annotation.defined()) {
      return Type();
    }
    vt.push_back(p->type_annotation);
  }

  return TupleType({f->ret_type, TupleType(vt)});
}

Expr FirstOrderGradient(const Expr& re, const Optional<IRModule>& mod) {
  // Currently we first remove any global functions for the first
  // order case.
  auto e = DeGlobal(mod, re);
  auto f = e.as<FunctionNode>();
  CHECK(f) << "FOWithGradient expects its argument to be a function: " << f;
  CHECK(f->type_params.size() == 0) << "no polymorphism supported for now";

  // We will then build a sequence of lets which implement reverse mode.
  Expr body = LetList::With([&](LetList* ll) {
    FirstOrderReverseAD reverse_ad(ll);
    ADValue rev = reverse_ad(e);
    std::vector<ADValue> args;
    for (const auto& p : f->params) {
      args.push_back(std::make_shared<ADTensor>(ll, p));
    }
    auto c = rev->get<ADFunction>().func(f->checked_type(), args, Attrs(), {});
    const auto& res = c->get<ADTensor>();
    Expr grad = LetList::With([&](LetList* ll) {
      res.reverse = MultiFactoryLike(res.forward, res.forward->checked_type(), Ones, OnesLike);
      for (auto it = reverse_ad.backprop_actions.rbegin(); it != reverse_ad.backprop_actions.rend();
           ++it) {
        (*it)(ll);
      }
      std::vector<Expr> grad_res;
      for (const auto& a : args) {
        grad_res.push_back(a->get<ADTensor>().reverse);
      }
      return Tuple(grad_res);
    });
    return Pair(res.forward, grad);
  });

  return Function(f->params, body, GradRetType(GetRef<Function>(f)), {});
}

TVM_REGISTER_GLOBAL("relay._transform.first_order_gradient").set_body_typed(FirstOrderGradient);

struct ReverseADType : TypeMutator {
  Type VisitType_(const TensorTypeNode* ttn) final {
    Type t = GetRef<Type>(ttn);
    return TupleType({t, RelayRefType(t)});
  }
};

Type ReverseType(const Type& t) { return ReverseADType()(t); }

/*! \brief Lift a function that transform Tensor to a function that also transform more type
 * by doing a structure preserving map.
 */
Expr LiftTensor(const std::function<Expr(const Expr& t)>& f,
                const std::function<Type(const Type&)>& tf, const Type& forward_type, const Expr& e,
                LetList* ll) {
  CHECK(IsAtomic(e)) << e;
  if (forward_type.as<TensorTypeNode>()) {
    auto ret = f(e);
    ret->checked_type_ = tf(forward_type);
    return ret;
  } else if (auto* tt = forward_type.as<TupleTypeNode>()) {
    tvm::Array<Expr> fields;
    tvm::Array<Type> types;
    for (size_t i = 0; i < tt->fields.size(); ++i) {
      auto field = LiftTensor(f, tf, tt->fields[i], ll->Push(GetField(e, i)), ll);
      fields.push_back(field);
      types.push_back(field->checked_type_);
    }
    auto ret = Tuple(fields);
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
  CHECK(IsAtomic(from)) << from;
  CHECK(IsAtomic(to)) << to;
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

/*! \brief t -> ReverseType(t). Transform to Reverse Mode Value. */
Expr GetRev(const Type& forward_type, const Expr& e, LetList* ll) {
  auto rev = [&](const Expr& e) { return Pair(e, ll->Push(RefCreate(ZerosLike(e)))); };
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
  auto grad = [&](const Expr& e) { return ll->Push(RefRead(GetField(e, 1))); };
  auto grad_type = [&](const Type& forward_type) { return forward_type; };
  return LiftTensor(grad, grad_type, forward_type, e, ll);
}

void UpdateGrad(const Type& t, const Expr& arg, const Expr& grad, LetList* ll) {
  if (t.as<TensorTypeNode>()) {
    ll->Push(RefWrite(GetField(arg, 1), Add(ll->Push(RefRead(GetField(arg, 1))), grad)));
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

  Var bp;
  std::shared_ptr<ADVarMap> ad_vars;
  const OpAttrMap<FPrimalGradient> rev_map = Op::GetAttrMap<FPrimalGradient>("FPrimalGradient");

  explicit ReverseAD(const Var& bp, std::shared_ptr<ADVarMap> ad_vars) : bp(bp), ad_vars(ad_vars) {}

  Expr VisitExpr_(const OpNode* op) final {
    LOG(FATAL) << "op should only be inside call";
    throw;
  }

  Expr VisitCheckpoint(const CallNode* call) {
    const OpNode* op_node = call->op.as<OpNode>();
    CHECK(op_node) << "expected op in call";
    Op op_ref = GetRef<Op>(op_node);
    CHECK(op_ref->name == "annotation.checkpoint") << "expected checkpoint annotation";
    auto x = call->args[0];
    return LetList::With([&](LetList* ll) {
      auto x_var = ll->Push(x);
      auto ret = ll->Push(GetRev(call->checked_type(), x_var, ll));
      auto bpv = ll->Push(RefRead(bp));
      Expr nbp = Function({}, LetList::With([&](LetList* ll) {
                            // we need a new ReverseAD visitor to avoid clobbering the bp local var
                            auto dup_bp = ll->Push(BPEmpty());
                            ReverseAD dup_diff(dup_bp, ad_vars);
                            auto dup_ad = ll->Push(dup_diff.VisitExpr(DeDup(x)));

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
    if (const OpNode* op_node = call->op.as<OpNode>()) {
      Op op_ref = GetRef<Op>(op_node);

      if (op_ref->name == "annotation.checkpoint") {
        return VisitCheckpoint(call);
      }

      CHECK(rev_map.count(op_ref)) << op_node->name << " does not have reverse mode defined";
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
        Expr nbp = Function({}, LetList::With([&](LetList* ll) {
                              tvm::Array<Expr> rev =
                                  rev_map[op_ref](orig, GetGrad(call->checked_type(), ret, ll));
                              CHECK(args.size() == rev.size());
                              for (size_t i = 0; i < args.size(); ++i) {
                                UpdateGrad(call->args[i]->checked_type(), args[i], rev[i], ll);
                              }
                              return Call(bpv, {});
                            }),
                            TupleType::Empty(), {});
        ll->Push(RefWrite(bp, nbp));
        return ret;
      });
    }
    return ExprMutator::VisitExpr_(call);
  }

  Expr VisitExpr_(const ConstantNode* op) final {
    Expr e = GetRef<Expr>(op);
    return Pair(e, RefCreate(ZerosLike(e)));
  }

  Expr VisitExpr_(const IfNode* op) final {
    return If(TupleGetItem(VisitExpr(op->cond), 0), VisitExpr(op->true_branch),
              VisitExpr(op->false_branch));
  }

  Expr VisitExpr_(const VarNode* var) final {
    // memoize Var -> ADVar so we don't end up with free Vars when checkpointing
    auto var_ref = GetRef<Var>(var);
    if (!ad_vars->count(var_ref)) {
      auto res = Downcast<Var>(ExprMutator::VisitExpr_(var));
      (*ad_vars)[var_ref] = res;
    }

    return ad_vars->at(var_ref);
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
  auto e = DeGlobal(mod, re);
  auto f = e.as<FunctionNode>();
  CHECK(f) << "input need to be a function";
  CHECK(f->type_params.size() == 0) << "no polymorphism supported for now";
  for (const auto& p : f->params) {
    CHECK(p->checked_type().as<TensorTypeNode>()) << "input parameters need to be tensor";
  }
  CHECK(!MissingGrad(e)) << "input has operators with missing gradients";
  Expr body = LetList::With([&](LetList* ll) {
    Var bp = ll->Push(BPEmpty());
    Expr rev = ReverseAD(bp, std::make_shared<ReverseAD::ADVarMap>())(e);
    std::vector<Expr> args;
    for (const auto& p : f->params) {
      args.push_back(ll->Push(Pair(p, RefCreate(ZerosLike(p)))));
    }
    auto c = ll->Push(Call(rev, args));
    std::function<void(const Expr&, const Type&)> init_grad;
    init_grad = [&](const Expr& e, const Type& t) {
      if (t.as<TensorTypeNode>()) {
        ll->Push(RefWrite(GetField(e, 1), OnesLike(GetField(e, 0))));
      } else if (auto tt = t.as<TupleTypeNode>()) {
        CHECK_GT(tt->fields.size(), 0);
        init_grad(ll->Push(GetField(e, 0)), tt->fields[0]);
      } else {
        LOG(FATAL) << "unhandled type " << t;
        throw;
      }
    };
    init_grad(c, f->body->checked_type());
    ll->Push(Call(RefRead(bp), {}));
    std::vector<Expr> ret;
    for (const auto& a : args) {
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
  return Function(f->params, body, GradRetType(GetRef<Function>(f)), {});
}

TVM_REGISTER_GLOBAL("relay._transform.gradient").set_body_typed(Gradient);

}  // namespace relay
}  // namespace tvm
