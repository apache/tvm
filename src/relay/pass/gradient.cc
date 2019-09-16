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
 *  Copyright (c) 2018 by Contributors
 * \file ad.cc
 * \brief API for Automatic Differentiation for the Relay IR.
 */

#include <tvm/lowered_func.h>
#include <tvm/operation.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include "pattern_util.h"
#include "pass_util.h"
#include "let_list.h"
#include "../ir/type_functor.h"

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
 * AD can also be run-time, which mean it is merely a function term of AD : (Float[] -> Float[]) -> (Float[] -> Float[]).
 * In relay we currently only support compile-time AD, but it should be enough for a lot of use case.
 *
 * In deep learning, the most common way to train a deep neural network is by gradient descent or some of it's variant.
 * Such optimization method require us to input the gradient of neural network, which can be obtained easily using AD.
 * In fact, back propagation is essentially reverse-mode automatic differentiation, a kind of AD!
 */

/*! In relay, automatic differentiation(AD) is a macro,
 *  that transform closed expr(expr without free variable/free type variable) of type
 *  (x0, x1, x2, ...) -> Float[] to
 *  (x0, x1, x2, ...) -> (Float[], (x0, x1,  x2, ...)),
 *  When x0, x1, x2... are Float of different shape.
 * the return value is a pair, with left hand side as the original value, and right hand side as gradient of the input.
 *  WithGradientType will take the type of input, and produce the type of output.
 *  There are multiple implementation of AD in relay, with different characteristic.
 *  However, they all transform the input expr according to WithGradientType.
 */
Type WithGradientType(const Type&);

/*! return an expression that represent differentiation of e (according to WithGradientType).
 *  This version only work on first order code without control flow.
 */
Expr FirstOrderGradient(const Expr& e, const Module& mod);

Type WithGradientType(const Type& t) {
  // TODO(M.K.): stricter checking
  auto ty = t.as<FuncTypeNode>();
  CHECK(ty) << "input should be a function";
  return FuncTypeNode::make(ty->arg_types,
                            TupleTypeNode::make({
                              ty->ret_type,
                              TupleTypeNode::make(ty->arg_types)}), {}, {});
}

//! \brief if the expression is a GlobalVar, transform to it's expression.
Expr DeGlobal(const Module& mod, const Expr& e) {
  if (const auto* x = e.as<GlobalVarNode>()) {
    return mod->Lookup(GetRef<GlobalVar>(x))->body;
  } else {
    return e;
  }
}

/*! \brief A fragment of the program being built by the automatic differentation
 *  pass.
 */
struct ADValueNode {
  virtual ~ADValueNode() { }
  template <typename T>
  T& get() {
    auto ret = dynamic_cast<T*>(this);
    CHECK(ret) << "cannot downcast";
    return *ret;
  }
};

using ADValue = std::shared_ptr<ADValueNode>;

/*! \brief AD over a program which generates a tensor output. */
struct ADTensor : ADValueNode {
  Expr forward;
  mutable Expr reverse;  // must be a variable to avoid duplication
  ADTensor(LetList* ll, const Expr& forward) :
    forward(ll->Push(forward)), reverse(ll->Push(ZerosLike(this->forward))) {
    this->forward->checked_type_ = forward->checked_type();
  }
};

/*! \brief A staged representation of the program, we reflect
 * Relay functions into a function over fragments of AD. We
 * can compute away this function to obtain a reverse mode program.
 */
struct ADFunction : ADValueNode {
  std::function<ADValue(const Type&,
                        const std::vector<ADValue>&,
                        const Attrs&,
                        const tvm::Array<Type>&)> func;
  explicit ADFunction(const std::function<ADValue(const Type&,
                                                  const std::vector<ADValue>&,
                                                  const Attrs&,
                                                  const tvm::Array<Type>&)>& func) :
    func(func) { }
};

struct FirstOrderReverseAD : ExprFunctor<ADValue(const Expr &)> {
  const OpMap<FPrimalGradient> rev_map = Op::GetAttr<FPrimalGradient>("FPrimalGradient");
  std::vector<std::function<void(LetList* ll)>> backprop_actions;
  // we assume no closure so no need for lexical scoping
  std::unordered_map<Var, ADValue, NodeHash, NodeEqual> env;
  LetList* ll;

  FirstOrderReverseAD(LetList* ll) : ll(ll) { }

  ADValue VisitExpr_(const OpNode* op) final {
    Op op_ref = GetRef<Op>(op);
    CHECK(rev_map.count(op_ref))
      << op->name << " does not have reverse mode defined";
    return std::make_shared<ADFunction>([this, op_ref](const Type& orig_type,
                                                       const std::vector<ADValue>& args,
                                                       const Attrs& attrs,
                                                       const tvm::Array<Type>& type_args) {
      std::vector<Expr> call_args;
      for (const ADValue& adval : args) {
        call_args.push_back(adval->get<ADTensor>().forward);
      }
      auto orig = CallNode::make(op_ref, call_args, attrs, type_args);
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
    return std::make_shared<ADFunction>([this, f](const Type& orig_type,
                                                  const std::vector<ADValue>& args,
                                                  const Attrs& attrs,
                                                  const tvm::Array<Type>& type_args) {
        CHECK_EQ(f->params.size(), args.size());
        for (size_t i = 0; i < f->params.size(); ++i) {
          env[f->params[i]] = args[i];
        }
        return VisitExpr(f->body);
      });
  }

  ADValue VisitExpr_(const VarNode* op) final {
    Var v = GetRef<Var>(op);
    return env.at(v);
  }
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

  return TupleTypeNode::make({f->ret_type, TupleTypeNode::make(vt)});
}

Expr FirstOrderGradient(const Expr& re, const Module& mod) {
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
      res.reverse = OnesLike(res.forward);
      for (auto it = reverse_ad.backprop_actions.rbegin();
           it != reverse_ad.backprop_actions.rend();
           ++it) {
        (*it)(ll);
      }
      std::vector<Expr> grad_res;
      for (const auto& a : args) {
        grad_res.push_back(a->get<ADTensor>().reverse);
      }
      return TupleNode::make(grad_res);
    });
    return Pair(res.forward, grad);
  });

  return FunctionNode::make(f->params, body, GradRetType(GetRef<Function>(f)), {});
}

TVM_REGISTER_API("relay._transform.first_order_gradient")
.set_body_typed(FirstOrderGradient);

struct ReverseADType : TypeMutator {
  Type VisitType_(const TensorTypeNode* ttn) final {
    Type t = GetRef<Type>(ttn);
    return TupleTypeNode::make({t, RefTypeNode::make(t)});
  }
};

Type ReverseType(const Type& t) {
  return ReverseADType()(t);
}

/*! \brief Lift a function that transform Tensor to a function that also transform more type
 * by doing a structure preserving map.
 */
Expr LiftTensor(const std::function<Expr(const Expr& t)>& f,
                const Type& t,
                const Expr& e,
                LetList* ll) {
  CHECK(IsAtomic(e)) << e;
  if (t.as<TensorTypeNode>()) {
    auto ret = f(e);
    ret->checked_type_ = t;
    return ret;
  } else if (auto* tt = t.as<TupleTypeNode>()) {
    tvm::Array<Expr> fields;
    for (size_t i = 0; i < tt->fields.size(); ++i) {
      fields.push_back(LiftTensor(f,
                                  tt->fields[i],
                                  ll->Push(GetField(e, i)),
                                  ll));
    }
    auto ret = TupleNode::make(fields);
    ret->checked_type_ = t;
    return std::move(ret);
  } else {
    LOG(FATAL) << "unsupported input/output type: " << tt;
    throw;
  }
}

/*! \brief t -> ReverseType(t). Transform to Reverse Mode Value. */
Expr GetRev(const Type& t, const Expr& e, LetList* ll) {
  auto rev = [&](const Expr& e) {
    return Pair(e, ll->Push(RefCreateNode::make(ZerosLike(e))));
  };
  return LiftTensor(rev, t, e, ll);
}

/*! \brief ReverseType(t) -> t. Get the original value. */
Expr GetValue(const Type& t, const Expr& e, LetList* ll) {
  return LiftTensor([&](const Expr& e) { return GetField(e, 0); }, t, e, ll);
}

/*! \brief ReverseType(t) -> t. Get the gradient. */
Expr GetGrad(const Type& t, const Expr& e, LetList* ll) {
  auto grad = [&](const Expr& e) {
    return ll->Push(RefReadNode::make(GetField(e, 1)));
  };
  return LiftTensor(grad, t, e, ll);
}

void UpdateGrad(const Type& t, const Expr& arg, const Expr& grad, LetList* ll) {
  if (t.as<TensorTypeNode>()) {
    ll->Push(RefWriteNode::make(GetField(arg, 1),
                                Add(ll->Push(RefReadNode::make(GetField(arg, 1))),
                                    grad)));
  } else if (auto* tt = t.as<TupleTypeNode>()) {
    for (size_t i = 0; i < tt->fields.size(); ++i) {
      UpdateGrad(tt->fields[i],
                 ll->Push(GetField(arg, i)),
                 ll->Push(GetField(grad, i)),
                 ll);
    }
  } else {
    LOG(FATAL) << "unsupported arg type of operator: " << t;
    throw;
  }
}

struct ReverseAD : ExprMutator {
  Var bp;
  const OpMap<FPrimalGradient> rev_map = Op::GetAttr<FPrimalGradient>("FPrimalGradient");

  explicit ReverseAD(const Var& bp) : bp(bp) { }

  Expr VisitExpr_(const OpNode* op) final {
    LOG(FATAL) << "op should only be inside call";
    throw;
  }

  Expr VisitExpr_(const CallNode* op) final {
    if (const OpNode* op_node = op->op.as<OpNode>()) {
      Op op_ref = GetRef<Op>(op_node);
      CHECK(rev_map.count(op_ref))
        << op_node->name << " does not have reverse mode defined";
      return LetList::With([&](LetList* ll) {
        std::vector<Var> args;
        for (const auto& arg : op->args) {
          args.push_back(ll->Push(VisitExpr(arg)));
        }
        std::vector<Expr> orig_args;
        for (size_t i = 0; i < args.size(); i++) {
          orig_args.push_back(GetValue(op->args[i]->checked_type(), args[i], ll));
        }
        Expr orig = CallNode::make(op->op, orig_args, op->attrs, op->type_args);
        orig->checked_type_ = op->checked_type();
        Var orig_var = ll->Push(orig);
        orig_var->checked_type_ = op->checked_type();
        auto ret = ll->Push(GetRev(op->checked_type(), orig_var, ll));
        auto bpv = ll->Push(RefReadNode::make(bp));
        Expr nbp = FunctionNode::make(
          {},
          LetList::With([&](LetList* ll) {
            tvm::Array<Expr> rev = rev_map[op_ref](orig, GetGrad(op->checked_type(), ret, ll));
            CHECK(args.size() == rev.size());
            for (size_t i = 0; i < args.size(); ++i) {
              UpdateGrad(op->args[i]->checked_type(), args[i], rev[i], ll);
            }
            return CallNode::make(bpv, {});
          }),
          TupleTypeNode::make({}),
          {});
        ll->Push(RefWriteNode::make(bp, nbp));
        return ret;
      });
    }
    return ExprMutator::VisitExpr_(op);
  }

  Expr VisitExpr_(const ConstantNode* op) final {
    Expr e = GetRef<Expr>(op);
    return Pair(e, RefCreateNode::make(ZerosLike(e)));
  }

  Expr VisitExpr_(const IfNode* op) final {
    return IfNode::make(TupleGetItemNode::make(VisitExpr(op->cond), 0),
                        VisitExpr(op->true_branch),
                        VisitExpr(op->false_branch));
  }

  Type VisitType(const Type& t) final {
    return t.defined() ? ReverseType(t) : t;
  }
};

Expr BPEmpty() {
  Expr unitF = FunctionNode::make({}, TupleNode::make({}), TupleTypeNode::make({}), {});
  return RefCreateNode::make(unitF);
}

Expr Gradient(const Expr& re, const Module& mod) {
  auto e = DeGlobal(mod, re);
  auto f = e.as<FunctionNode>();
  CHECK(f) << "input need to be a function";
  CHECK(f->type_params.size() == 0) << "no polymorphism supported for now";
  for (const auto& p : f->params) {
    CHECK(p->checked_type().as<TensorTypeNode>()) << "input parameters need to be tensor";
  }
  Expr body = LetList::With([&](LetList* ll) {
    Var bp = ll->Push(BPEmpty());
    Expr rev = ReverseAD(bp)(e);
    std::vector<Expr> args;
    for (const auto& p : f->params) {
      args.push_back(ll->Push(Pair(p, RefCreateNode::make(ZerosLike(p)))));
    }
    auto c = ll->Push(CallNode::make(rev, args));
    std::function<void(const Expr&, const Type&)> init_grad;
    init_grad = [&](const Expr& e, const Type& t) {
      if (t.as<TensorTypeNode>()) {
        ll->Push(RefWriteNode::make(GetField(e, 1), OnesLike(GetField(e, 0))));
      } else if (auto tt = t.as<TupleTypeNode>()) {
        CHECK_GT(tt->fields.size(), 0);
        init_grad(ll->Push(GetField(e, 0)), tt->fields[0]);
      } else {
        LOG(FATAL) << "unhandled type " << t;
        throw;
      }
    };
    init_grad(c, f->body->checked_type());
    ll->Push(CallNode::make(RefReadNode::make(bp), {}));
    std::vector<Expr> ret;
    for (const auto& a : args) {
      ret.push_back(RefReadNode::make(GetField(a, 1)));
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
        return TupleNode::make(fields);
      } else {
        LOG(FATAL) << "unhandled type " << t;
        throw;
      }
    };
    return Pair(get_final_result(c, f->body->checked_type()), TupleNode::make(ret));
  });
  return FunctionNode::make(f->params, body, GradRetType(GetRef<Function>(f)), {});
}

TVM_REGISTER_API("relay._transform.gradient")
.set_body_typed(Gradient);

}  // namespace relay
}  // namespace tvm
