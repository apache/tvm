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
 * \file first_order_ad.cc
 * \brief First-order AD for Relay.
 */
#include <tvm/ir/type_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/feature.h>
#include <tvm/relay/transform.h>
#include <tvm/te/operation.h>

#include "let_list.h"
#include "pass_utils.h"
#include "pattern_utils.h"

namespace tvm {
namespace relay {

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

/*! \brief A fragment of the program being built by the automatic differentation
 *  pass.
 */
struct ADValueNode {
  virtual ~ADValueNode() {}
  template <typename T>
  T& get() {
    auto ret = dynamic_cast<T*>(this);
    ICHECK(ret) << "cannot downcast";
    return *ret;
  }
};

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
  // (ad_args, orig) -> ad_ret
  using ADFunctionType = ADValue(const std::vector<ADValue>&, const Call&);
  std::function<ADFunctionType> func;
  explicit ADFunction(const std::function<ADFunctionType>& func) : func(func) {}
};

struct FirstOrderReverseAD : ExprFunctor<ADValue(const Expr&)> {
  const OpAttrMap<FPrimalGradient> rev_map = Op::GetAttrMap<FPrimalGradient>("FPrimalGradient");
  std::vector<std::function<void(LetList* ll)>> backprop_actions;
  // we assume no closure so no need for lexical scoping
  std::unordered_map<Expr, ADValue, ObjectPtrHash, ObjectPtrEqual> env;
  LetList* ll;
  DiagnosticContext diag_ctx;

  FirstOrderReverseAD(LetList* ll, DiagnosticContext diag_ctx) : ll(ll), diag_ctx(diag_ctx) {}

  ADValue VisitExpr(const Expr& n) final {
    if (env.count(n)) {
      return env.at(n);
    }
    auto ret = ExprFunctor::VisitExpr(n);
    env[n] = ret;
    return ret;
  }

  static Expr LiftedAdd(const Type& t, const Expr& x, const Expr& y, LetList* ll) {
    if (t.as<TensorTypeNode>()) {
      return ll->Push(Add(x, y));
    } else if (auto* tt = t.as<TupleTypeNode>()) {
      Array<Expr> fields;
      for (size_t i = 0; i < tt->fields.size(); ++i) {
        fields.push_back(
            LiftedAdd(tt->fields[i], ll->Push(GetField(x, i)), ll->Push(GetField(y, i)), ll));
      }
      return ll->Push(Tuple(fields));
    } else {
      LOG(FATAL) << "cannot lift addition for type " << PrettyPrint(t);
      throw;
    }
  }

  ADValue VisitExpr_(const OpNode* op) final {
    Op op_ref = GetRef<Op>(op);
    if (!rev_map.count(op_ref)) {
      diag_ctx.EmitFatal(Diagnostic::Error(op->span)
                         << "the operator " << op->name << " does not have a registered gradient.");
    }
    return std::make_shared<ADFunction>([this, op_ref](const std::vector<ADValue>& ad_args,
                                                       const Call& orig) {
      std::vector<Expr> orig_args;
      for (const ADValue& adval : ad_args) {
        orig_args.push_back(adval->get<ADTensor>().forward);
      }
      auto orig_new = Call(op_ref, orig_args, orig->attrs, orig->type_args);
      orig_new->checked_type_ = orig->checked_type();
      auto ret = std::make_shared<ADTensor>(ll, orig_new);
      backprop_actions.push_back([this, ad_args, orig_new, ret, op_ref](LetList* ll) {
        tvm::Array<Expr> rev = rev_map[op_ref](orig_new, ret->reverse);
        if (ad_args.size() != rev.size()) {
          diag_ctx.EmitFatal(Diagnostic::Error(op_ref->span)
                             << "arity mismatch for operator " << op_ref->name
                             << " and its registered gradient: expected " << ad_args.size()
                             << " but got " << rev.size() << " gradients.");
        }
        for (size_t i = 0; i < ad_args.size(); ++i) {
          auto& ad_arg = ad_args[i]->get<ADTensor>();
          ad_arg.reverse = LiftedAdd(ad_arg.forward->checked_type(), ad_arg.reverse, rev[i], ll);
        }
      });
      return ret;
    });
  }

  ADValue VisitExpr_(const TupleGetItemNode* op) final {
    Expr e = GetRef<Expr>(op);
    ADValue tup = VisitExpr(op->tuple);
    auto tt = op->tuple->checked_type().as<TupleTypeNode>();
    size_t idx = op->index;
    auto ret = std::make_shared<ADTensor>(ll, e);
    backprop_actions.push_back([tup, tt, idx, ret](LetList* ll) {
      auto& ad_tup = tup->get<ADTensor>();
      std::vector<Expr> updated_grads;
      for (size_t i = 0; i < tt->fields.size(); ++i) {
        Expr grad_pre = GetField(ad_tup.reverse, i);
        updated_grads.push_back(i != idx ? grad_pre
                                         : LiftedAdd(tt->fields[i], grad_pre, ret->reverse, ll));
      }
      ad_tup.reverse = ll->Push(Tuple(updated_grads));
    });
    return ret;
  }

  ADValue VisitExpr_(const TupleNode* op) final {
    Expr e = GetRef<Expr>(op);
    std::vector<ADValue> fields;
    for (const auto& f : op->fields) {
      fields.push_back(VisitExpr(f));
    }
    auto tt = op->checked_type().as<TupleTypeNode>();
    auto ret = std::make_shared<ADTensor>(ll, e);
    backprop_actions.push_back([fields, tt, ret](LetList* ll) {
      for (size_t i = 0; i < fields.size(); ++i) {
        auto& ad_field = fields[i]->get<ADTensor>();
        ad_field.reverse =
            LiftedAdd(tt->fields[i], ad_field.reverse, GetField(ret->reverse, i), ll);
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
    return f->get<ADFunction>().func(args, GetRef<Call>(op));
  }

  ADValue VisitExpr_(const FunctionNode* op) final {
    Function f = GetRef<Function>(op);
    // todo: assert no closure
    return std::make_shared<ADFunction>(
        [this, f](const std::vector<ADValue>& ad_args, const Call& orig) {
          ICHECK_EQ(f->params.size(), ad_args.size());
          for (size_t i = 0; i < f->params.size(); ++i) {
            env[f->params[i]] = ad_args[i];
          }
          return VisitExpr(f->body);
        });
  }

  // Var will always be in env, handled in VisitExpr (without _), so we don't need
  // to implement its VisitExpr_.
};

Type GradientReturnType(const Function& f) {
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

namespace transform {

Pass FirstOrderAD() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> f = [](IRModule mod, PassContext ctx) {
    IRModule ad_mod = GetRef<IRModule>(mod.CopyOnWrite());
    DiagnosticContext diag_ctx = DiagnosticContext::Default(ad_mod);

    for (const auto& pr : mod->functions) {
      const FunctionNode* func = pr.second.as<FunctionNode>();
      if (!func) {
        diag_ctx.EmitFatal(Diagnostic::Error(pr.second->span)
                           << "AD can only be performed on Relay functions.");
      }
      if (func->type_params.size() > 0) {
        diag_ctx.EmitFatal(Diagnostic::Error(pr.second->span)
                           << "first-order AD does not support polymorphism yet.");
      }
      Expr body = LetList::With([&](LetList* ll) {
        FirstOrderReverseAD reverse_ad(ll, diag_ctx);
        ADValue rev = reverse_ad(pr.second);
        std::vector<ADValue> args;
        for (const auto& p : func->params) {
          args.push_back(std::make_shared<ADTensor>(ll, p));
        }
        Call placeholder = Call(GetRef<Function>(func), {});
        placeholder->checked_type_ = func->checked_type().as<FuncTypeNode>()->ret_type;
        auto grad_call = rev->get<ADFunction>().func(args, placeholder);
        auto& res = grad_call->get<ADTensor>();
        Expr grad_tuple = LetList::With([&](LetList* ll) {
          res.reverse = MultiFactoryLike(res.forward, res.forward->checked_type(), Ones, OnesLike);
          for (auto it = reverse_ad.backprop_actions.rbegin();
               it != reverse_ad.backprop_actions.rend(); ++it) {
            (*it)(ll);
          }
          std::vector<Expr> grads;
          for (const auto& a : args) {
            grads.push_back(a->get<ADTensor>().reverse);
          }
          return Tuple(grads);
        });
        return Pair(res.forward, grad_tuple);
      });
      ad_mod->Update(pr.first,
                     Function(func->params, body, GradientReturnType(GetRef<Function>(func)), {}));
    }

    return ad_mod;
  };
  return CreateModulePass(f, 0, "FirstOrderAD", {});
}

TVM_REGISTER_GLOBAL("relay._transform.FirstOrderAD").set_body_typed(FirstOrderAD);

}  // namespace transform

}  // namespace relay
}  // namespace tvm