/*!
 *  Copyright (c) 2018 by Contributors
 * \file ad.cc
 * \brief API for Automatic Differentiation for the Relay IR.
 */

#include <tvm/lowered_func.h>
#include <tvm/operation.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pass.h>
#include "pattern_util.h"
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
  Expr foward;
  mutable Expr reverse;  // must be a variable to avoid duplication
  ADTensor(LetList* ll, const Expr& foward) :
    foward(ll->Push(foward)), reverse(ll->Push(ZeroLike(this->foward))) { }
};

/*! \brief A staged representation of the program, we reflect
 * Relay functions into a function over fragments of AD. We
 * can compute away this function to obtain a reverse mode program.
 */
struct ADFunction : ADValueNode {
  std::function<ADValue(const std::vector<ADValue>&,
                        const Attrs&,
                        const tvm::Array<Type>&)> func;
  explicit ADFunction(const std::function<ADValue(const std::vector<ADValue>&,
                                                  const Attrs&,
                                                  const tvm::Array<Type>&)>& func) :
    func(func) { }
};

struct ReverseAD : ExprFunctor<ADValue(const Expr &)> {
  const OpMap<FPrimalGradient> rev_map = Op::GetAttr<FPrimalGradient>("FPrimalGradient");
  std::vector<std::function<void(LetList* ll)>> backprop_actions;
  // we assume no closure so no need for lexical scoping
  std::unordered_map<Var, ADValue, NodeHash, NodeEqual> env;
  LetList* ll;

  ReverseAD(LetList* ll) : ll(ll) { }

  ADValue VisitExpr_(const OpNode* op) final {
    Op op_ref = GetRef<Op>(op);
    CHECK(rev_map.count(op_ref))
      << op->name << " does not have reverse mode defined";
    return std::make_shared<ADFunction>([this, op_ref](const std::vector<ADValue>& args,
                                                       const Attrs& attrs,
                                                       const tvm::Array<Type>& type_args) {
        std::vector<Expr> call_args;
        for (const ADValue& adval : args) {
          call_args.push_back(adval->get<ADTensor>().foward);
        }
        auto orig = CallNode::make(op_ref, call_args, attrs, type_args);
        auto ret = std::make_shared<ADTensor>(ll, orig);
        backprop_actions.push_back([this, args, orig, ret, op_ref](LetList* ll) {
            tvm::Array<Expr> rev = rev_map[op_ref](orig, ret->reverse);
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
    return f->get<ADFunction>().func(args, op->attrs, op->type_args);
  }

  ADValue VisitExpr_(const FunctionNode* op) final {
    Function f = GetRef<Function>(op);
    // todo: assert no closure
    return std::make_shared<ADFunction>([this, f](const std::vector<ADValue>& args,
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

Expr FirstOrderGradient(const Expr& re, const Module& mod) {
  // Currently we first remove any global functions for the first
  // order case.
  auto e = DeGlobal(mod, re);
  auto f = e.as<FunctionNode>();
  CHECK(f) << "FOWithGradient expects its argument to be a function: " << f;
  CHECK(f->type_params.size() == 0) << "no polymorphism supported for now";

  // We will then build a sequence of lets which implement reverse mode.
  Expr body = LetList::With([&](LetList* ll) {
    ReverseAD reverse_ad(ll);
    ADValue rev = reverse_ad(e);
    std::vector<ADValue> args;
    for (const auto& p : f->params) {
      args.push_back(std::make_shared<ADTensor>(ll, p));
    }
    auto c = rev->get<ADFunction>().func(args, Attrs(), {});
    const auto& res = c->get<ADTensor>();
    Expr grad = LetList::With([&](LetList* ll) {
        res.reverse = OneLike(res.foward);
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
    return Pair(res.foward, grad);
  });

  // if type annotations are provided, we will construct a ret type;
  // otherwise, leave it to be inferred
  Type ret_type = Type();
  std::vector<Type> vt;
  bool missing = !f->ret_type.defined();
  for (const auto& p : f->params) {
    if (missing || !p->type_annotation.defined()) {
      missing = true;
      break;
    }
    vt.push_back(p->type_annotation);
  }

  if (!missing) {
    ret_type = TupleTypeNode::make({f->ret_type, TupleTypeNode::make(vt)});
  }

  return FunctionNode::make(f->params, body, ret_type, {});
}

TVM_REGISTER_API("relay._ir_pass.first_order_gradient")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
      CHECK_EQ(args.size(), 2);
      *ret = FirstOrderGradient(args[0], args[1]);
    });

}  // namespace relay
}  // namespace tvm
