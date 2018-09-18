/*!
 *  Copyright (c) 2018 by Contributors
 * \file ad.cc
 * \brief API for Automatic Differentiation for the Relay IR.
 */

#include <tvm/relay/ad.h>
#include <tvm/relay/builder.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pass.h>
#include "let_list.h"
#include "type_functor.h"

namespace tvm {
namespace relay {

using namespace tvm::runtime;

Type WithGradientType(const Type& t) {
  // TODO(M.K.): stricter checking
  auto ty = t.as<FuncTypeNode>();
  CHECK(ty) << "input should be a function";
  return FuncTypeNode::make(ty->arg_types,
                            TupleTypeNode::make({
                              ty->ret_type,
                              TupleTypeNode::make(ty->arg_types)}), {}, {});
}

/*! First Order Reverse mode Automatic Differentiation(FORAD) is implemented using wengert list.
 *
 *  Traditionally, wengert list is a runtime datastructure used to record value of all variables,
 *  with their gradient, which is originally set to 0.
 *
 *  After the function calculation is complete (with wengert list being updated),
 *  we can set the result gradient to 1, replay all computation backward, propagating the gradient,
 *  from operator result, to operator argument.
 *
 *  Since the operation is play in backward, it is called reverse mode automatic differentiation.
 *
 *  In relay, we want the list to only exists at compile time,
 *  as we do not want to place burden on tvm runtime.
 *  Thus instead of holding runtime value of floats,
 *  the wengert list hold compile time value: it is a list of pair,
 *  with left hand side being Expr, representing the gradient,
 *  and right hand side a function, representing code that will propagate the gradient.
 *
 *  After the wengert list finish updating (local mutation at compile time),
 *  we can use it to generate fully first order code, without mutation of any kind.
 *
 *  More specifically, the work is done by transforming complete function literal,
 *  which transform Real to a compile time pair,
 *  containing the original code and an location in the wengert list.
 *
 *  All others datatype/function get lifted to compile time to hold/transform such value,
 *  so we cannot support dynamic length list,
 *  as we have no idea how many element we should have on the wengert list.
 *
 *  Right now we only support float and first order function,
 *  but extending it under the hood of lifting to compile time should be straight forward.
 *
 *  Note that if, instead of transforming value into compile time structure,
 *  we do the exact opposite, transforming the wengert list into run time term,
 *  where grad_loc get translated to reference, wengert list get translated to stack of callback,
 *  we become basically demystifying differentiable programming.
 */
struct FORAD {
  virtual ~FORAD() = 0;
  template <typename T>
  T& get() {
    auto ret = dynamic_cast<T*>(this);
    CHECK(ret) << "cannot downcast";
    return *ret;
  }
};

FORAD::~FORAD() { }

//! \brief a compile time representation of WengertList.
struct WengertList {
  //! \brief the LetList to store Exp generated when going backward.
  LetList ll;
  //! \brief the gradient of param
  std::unordered_map<Var, Expr, NodeHash, NodeEqual> grad_param;

  /*! \brief the stack that store all value calculated.
   *  The Expr represent gradient, and the function is a callback that propagate gradient,
   *  with the input being it's gradient.
   */
  std::vector<std::pair<Expr, std::function<void(const Expr &)> > > stack;
};

/*! \brief lifted tensor.
 *  It hold the original tensor, and it's location of gradient on the stack.
 */
struct FloatFORAD : FORAD {
  Expr orig;
  size_t grad_loc;
  FloatFORAD(const Expr& orig, size_t grad_loc) : orig(orig), grad_loc(grad_loc) { }
};

/* TODO(M.K.) a cleaner way to express, is to have grad_loc be std::shared_ptr<Expr> instead.
 * we can then strip grad_param and the Expr inside stack from WengertList,
 * and the translation to Demystifying become more apparent:
 * remove LetList (we are in value world),
 * turn shared_ptr<Expr> into Expr holding reference,
 * let the whole FloatFORAD hold Expr of pair instead of pair of Expr,
 * and unlift other lifted structure in similar manner.
 * it is almost mechanical!
 */

/*! \brief lifted function.
 *  It hold function with same spec as those of Op::reverse_mode().
 *  Generally speaking, we should lift it to std::function<FORAD(const std::vector<FORAD>&)>,
 *  But for simplicity we keep them in current form.
 *  Later pass might fix it.
 */
struct FuncFORAD : FORAD {
  OpNode::reverse_mode_type rev;
  explicit FuncFORAD(const OpNode::reverse_mode_type& rev) : rev(rev) {}
};

std::vector<Type> from_params(const tvm::Array<Param>& params) {
  std::vector<Type> v;
  for (const auto& param : params) {
    v.push_back(param->type);
  }
  return v;
}

class Transformer : private ExprFunctor<std::shared_ptr<FORAD>(const Expr&)> {
 private:
  Environment env;
  LetList ll;
  WengertList wl;
  std::unordered_map<Var, std::shared_ptr<FORAD>, NodeHash, NodeEqual> transformed;

  Transformer(const Environment& env) : env(env) { }

  std::shared_ptr<FORAD> VisitExpr_(const ConstantNode* n) final {
    wl.stack.push_back(std::make_pair(ZeroLike(GetRef<Constant>(n)), [](const Expr &exp) { }));
    return std::shared_ptr<FORAD>(new FloatFORAD(GetRef<Constant>(n), wl.stack.size() - 1));
  }

  std::shared_ptr<FORAD> VisitExpr_(const VarNode* n) final {
    return transformed[GetRef<Var>(n)];
  }

  std::shared_ptr<FORAD> VisitExpr_(const OpNode* n) final {
    CHECK(n->reverse_mode) << n->name << " does not has reverse mode";
    return std::make_shared<FuncFORAD>(FuncFORAD(n->reverse_mode));
  }

  std::shared_ptr<FORAD> VisitExpr_(const LetNode* n) final {
    // assume let value and body to be float for now.
    auto vadop = (*this)(n->value);
    transformed.insert(std::make_pair(n->var, vadop));
    return (*this)(n->body);
  }

  std::shared_ptr<FORAD> VisitExpr_(const CallNode* n) final {
    // assume param and return to be float for now.
    auto adfn = (*this)(n->op)->get<FuncFORAD>().rev;
    std::vector<Expr> orig_args;
    std::vector<size_t> args_grad_loc;
    for (const Expr& arg : n->args) {
      const auto forad = (*this)(arg)->get<FloatFORAD>();
      orig_args.push_back(forad.orig);
      args_grad_loc.push_back(forad.grad_loc);
    }
    auto adfnr = adfn(&ll, orig_args);
    auto retres = adfnr.first;
    auto retgrad = adfnr.second;
    wl.stack.push_back(std::make_pair(ZeroLike(GetRef<Call>(n)),
                                      [this, args_grad_loc, retgrad](const Expr& exp) {
                                        std::vector<Expr> old_grad;
                                        for (size_t i = 0; i < args_grad_loc.size(); ++i) {
                                          size_t idx = args_grad_loc[i];
                                          old_grad.push_back(wl.stack[idx].first);
                                        }
                                        auto grad = retgrad(&wl.ll, exp, old_grad);
                                        for (size_t i = 0; i < args_grad_loc.size(); ++i) {
                                          size_t idx = args_grad_loc[i];
                                          wl.stack[idx].first = grad[i];
                                        }
                                      }));
    return std::shared_ptr<FORAD>(new FloatFORAD(retres, wl.stack.size() - 1));
  }

  std::vector<Expr> YieldWrt(const tvm::Array<Var>& wrt) {
    std::vector<Expr> ret;
    for (const Var& v : wrt) {
      ret.push_back(wl.grad_param.at(v));
    }
    return ret;
  }

 public:
  static Expr Rev(const Environment& env, const Expr& e, const tvm::Array<Var>& wrt) {
    Transformer tf(env);
    auto f = e.as<FunctionNode>();
    CHECK(f) << "input need to be a function";
    // push the param onto the Wengert list
    for (const Param& p : f->params) {
      size_t idx = tf.wl.stack.size();
      tf.wl.grad_param.insert(std::pair<Var, Expr>(p->var, ZeroLike(p->var)));
      tf.wl.stack.push_back(std::make_pair(ZeroLike(p->var), [&tf, p, env](const Expr &e) {
            tf.wl.grad_param[p->var] = Plus(tf.wl.grad_param[p->var], e);
          }));
      tf.transformed.insert(std::make_pair(
        p->var,
        std::shared_ptr<FORAD>(new FloatFORAD(p->var, idx))));
    }
    auto ret = tf(f->body);
    auto fret = ret->get<FloatFORAD>();
    auto ret_grad = VarNode::make("ret_grad");
    // init gradient of result
    tf.wl.stack[fret.grad_loc].first = ret_grad;
    // back propagation
    while (!tf.wl.stack.empty()) {
      tf.wl.stack.back().second(tf.wl.stack.back().first);
      tf.wl.stack.pop_back();
    }
    auto ty = from_params(f->params);
    auto type = TupleTypeNode::make({
      f->ret_type,
      FuncTypeNode::make({f->ret_type},
                         TupleTypeNode::make(ty),
                         {},
                         {})});
    return FunctionNode::make(
      f->params, type,
      tf.ll.Get(Pair(
        fret.orig,
        FunctionNode::make({ParamNode::make(ret_grad, f->ret_type)},
                           TupleTypeNode::make(ty),
                           tf.wl.ll.Get(TupleNode::make(tf.YieldWrt(wrt))),
                           {}))),
      f->type_params);
  }
};

//! \brief if the expression is a GlobalVar, transform to it's expression.
Expr DeGlobal(const Environment& env, const Expr& e) {
  if (auto x = e.as<GlobalVarNode>()) {
    return env->Lookup(GetRef<GlobalVar>(x))->body;
  } else {
    return e;
  }
}

Expr FOWithGradient(const Environment& env, const Expr& re, const tvm::Array<Var>& wrt) {
  auto e = DeGlobal(env, re);
  auto f = e.as<FunctionNode>();
  CHECK(f) << "input need to be a function";
  std::vector<Expr> args;
  for (const Param& p : f->params) {
    args.push_back(p->var);
  }
  return FunctionNode::make(
    f->params,
    TupleTypeNode::make({
      f->ret_type,
      TupleTypeNode::make(from_params(f->params))
    }),
    LetList::With([&](LetList* ll) {
      auto res = ll->Push(CallNode::make(Transformer::Rev(env, e, wrt), args));
      return Pair(GetField(res, 0),
                  CallNode::make(GetField(res, 1),
                                 {OneLike(GetField(res, 0))}));
      }),
    f->type_params);
}

Expr FOWithGradient(const Environment& env, const Expr& re) {
  auto e = DeGlobal(env, re);
  auto f = e.as<FunctionNode>();
  CHECK(f) << "input need to be a function";
  std::vector<Var> args;
  for (const Param & p : f->params) {
    args.push_back(p->var);
  }
  return FOWithGradient(env, re, tvm::Array<Var>(args));
}

TVM_REGISTER_API("relay._ir_pass.fo_with_gradient")
  .set_body([](TVMArgs args, TVMRetValue* ret) {
      if (args.size() == 2) {
        *ret = FOWithGradient(args[0], args[1]);
      } else {
        tvm::Array<Var> vars = args[2];
        *ret = FOWithGradient(args[0], args[1], vars);
      }
    });

}  // namespace relay
}  // namespace tvm
