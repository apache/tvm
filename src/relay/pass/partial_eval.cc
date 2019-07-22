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
 * Copyright (c) 2019 by Contributors
 *
 * \file partial_eval.cc
 *
 * \brief Perform known computation in compile time.
 *
 * The partial evaluator try to do computation at compile time,
 * so it can generate code that do less work.
 * Additionally, it might open more chance for further optimization,
 * since the high level, structural part of the code (closure, reference, control flow)
 * might get partially evaluated away, and the subsequent optimization (for example, kernel fusion)
 * can reason across those structural code as it got removed.
 * In the extreme case, partial evaluation can even turn the whole program
 * into pure first order computation with no control flow.
 * In such a case, we can compile the whole computation onto SIMD Instruction/GPU/FPGA,
 * and get huge speedup.
 *
 * It works by making the following modifications to the standard relay interpreter:
 *
 * 0: The values become partially static value.
 * Since we cannot know the value of every term at compile time,
 * Term might get partially evaluated to 'Unknown Value'.
 * Every partially static value is, hence,
 * a static fragment that might not be there (partially static),
 * and a dynamic fragment that is semantically equivalent to the original term,
 * so the unknown part will be computed at runtime, using the dynamic fragment.
 *
 * 1: The interpreter holds a LetList, which preserves A Normal Form for the generated code.
 * More specifically, we require that all dynamic is an atom.
 * This avoids code duplication (which is both inefficient and incorrect), as atom has constant size
 * and allow us to not handle capture-avoidance substitution (as atom has no binder).
 *
 * 2: The map of References to partially static values is reified, as described below.
 * Instead of Reference having mutable field, Reference only has an unique identifier.
 * There will be a mutable mapping of id to partially static value, called the store.
 * This allow us to rollback the store:
 * when a path may or may not be executed (as in a conditional), we copy the store,
 * recurse with the copy, and reinstate the original when the call returns
 * so that the effects of the computation are not preserved.
 * We do this in if else, pattern matching, and in function,
 * as, when we see a function, we partially evaluate it with all the argument as dynamic,
 * to generate efficient dynamic for that function.
 *
 * 3: The generated code reuses bindings (although they are not shadowed),
 * so we have to deduplicate them.
 *
 * 4: In the generated code, as it call TypeSubst, multiple VarNode might have same Id.
 * While it is permitted, most pass use NodeHash for Var,
 * and having multiple VarNode for same Id break them.
 * Thus we remap them to a single Id for now.
 *
 * Also, It will also generate lots of dead code,
 * so it is a good idea to feed it through the dead code eliminator after partial evaluation.
 *
 * The partial evaluator makes several assumptions, so there is room for improvement:
 *
 * 0: Every time an unknown effect happened, we clear the whole store.
 * It is too conservative: if a local reference is created (and do not get passed outside),
 * An unknown global function call/global reference write can not modify it.
 * We can pair PE with escape analysis/alias analysis.
 *
 * 1: We assume all unknown code has effect. Doing effect analysis can make the store more precise.
 *
 * 2: When doing pattern matching, we can simplify the match even for dynamic case.
 * Right now it is all or nothing: either a complete match, or the original dynamic code.
 * Instead, we can get a match tree, pair it with the data and evaluate it to a normal form.
 * We then can reify the result.
 *
 * 3: Every time a function is called, its code will get expanded and partially evaluated.
 * We can do a binding time analysis to cache the result and avoid re-partial evaluation.
 *
 * These assumptions do not affect the correctness of the algorithm, however.
 */
#include <tvm/relay/analysis.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relay/interpreter.h>
#include "../ir/type_functor.h"
#include "pass_util.h"
#include "let_list.h"

namespace tvm {
namespace relay {
namespace partial_eval {

using namespace runtime;

/*! \brief Hash Var by it's id.
 * Different VarNode might has same vid, and they are considered to be the same var in such case.
 * Use VarHash to hash Var by id.
 */
struct VarHash {
  size_t operator()(const Var& v) const {
    return v->vid.hash();
  }
};

/*! \brief Compare Var by it's id.
 * Different VarNode might has same vid, and they are considered to be the same var in such case.
 * Use VarEqual to compare Var by id.
 */
struct VarEqual {
  bool operator()(const Var& l, const Var& r) const {
    return l->vid.get() == r->vid.get();
  }
};

Expr PostProcess(const Expr&);

/*! \brief The base container type of Relay values. */
class StaticNode : public RelayNode {
 public:
  static constexpr const char* _type_key = "relay.Value";
  TVM_DECLARE_BASE_NODE_INFO(ValueNode, RelayNode);
};

class Static : public NodeRef {
 public:
  Static() {}
  explicit Static(NodePtr<Node> n) : NodeRef(n) {}
  const ValueNode* operator->() const {
    return static_cast<const ValueNode*>(node_.get());
  }

  using ContainerType = StaticNode;
};

using Time = size_t;

struct PStaticNode : Node {
  static Time time() {
    static Time time_ = 0;
    Time ret = time_;
    time_++;
    return ret;
  }
  Static pstatic;  // may be null
  Expr dynamic;
  Time created_time;
  PStaticNode(const Static& pstatic, const Expr& dynamic) :
    pstatic(pstatic), dynamic(dynamic), created_time(time()) { }
  explicit PStaticNode(const Expr& dynamic) : PStaticNode(Static(), dynamic) { }
  TVM_DECLARE_NODE_TYPE_INFO(PStaticNode, Node);
};

RELAY_DEFINE_NODE_REF(PStatic, PStaticNode, NodeRef);

struct STupleNode : StaticNode {
  std::vector<PStatic> fields;
  explicit STupleNode(const std::vector<PStatic>& fields) : fields(fields) { }
  TVM_DECLARE_NODE_TYPE_INFO(STupleNode, StaticNode);
};

RELAY_DEFINE_NODE_REF(STuple, STupleNode, Value);

Static MkSTuple(const std::vector<PStatic>& fields) {
  return Static(make_node<STupleNode>(fields));
}

struct STensorNode : StaticNode {
  runtime::NDArray data;
  explicit STensorNode(const NDArray& data) : data(data) { }
  TVM_DECLARE_NODE_TYPE_INFO(STupleNode, StaticNode);
};

RELAY_DEFINE_NODE_REF(STensor, STensorNode, Value);

Static MkSTensor(const NDArray& data) {
  return Static(make_node<STensorNode>(data));
}

struct SConstructorNode : StaticNode {
  Constructor constructor;
  std::vector<PStatic> fields;
  SConstructorNode(const Constructor& constructor, const std::vector<PStatic>& fields) :
    constructor(constructor), fields(fields) { }
  TVM_DECLARE_NODE_TYPE_INFO(SConstructorNode, StaticNode);
};

RELAY_DEFINE_NODE_REF(SConstructor, SConstructorNode, Value);

Static MkSConstructor(const Constructor& constructor, const std::vector<PStatic>& fields) {
  return Static(make_node<SConstructorNode>(constructor, fields));
}

struct SRefNode : StaticNode {
  // we will use the address as the guid for hashing
  TVM_DECLARE_NODE_TYPE_INFO(SRefNode, StaticNode);
};

RELAY_DEFINE_NODE_REF(SRef, SRefNode, Value);

Static MkSRef() {
  return Static(make_node<SRefNode>());
}

using Func = std::function<PStatic(const std::vector<PStatic>&,
                                   const Attrs&,
                                   const Array<Type>&,
                                   LetList*)>;

struct SFuncNode : StaticNode {
  Func func;
  explicit SFuncNode(const Func& func) : func(func) { }
  TVM_DECLARE_NODE_TYPE_INFO(SFuncNode, StaticNode);
};

RELAY_DEFINE_NODE_REF(SFunc, SFuncNode, Value);

Static MkSFunc(const Func& func) {
  return Static(make_node<SFuncNode>(func));
}

/*!
 * \brief A stack frame in the Relay interpreter.
 *
 * Contains a mapping from relay::Var to relay::Value.
 */
struct Frame {
  /*! \brief The set of local variables and arguments for the frame. */
  std::unordered_map<Var, PStatic, VarHash, VarEqual> locals;
  Frame() = default;
};

class Environment {
 public:
  Environment() : env_({Frame()}) { }
  Environment(const Environment&) = delete;

  template<typename T>
  T Extend(const std::function<T()>& body) {
    FrameContext fc(this);
    return body();
  }

  void Insert(const Var& v, const PStatic& ps) {
    CHECK(ps.defined());
    CHECK_EQ(env_.back().locals.count(v), 0);
    env_.back().locals[v] = ps;
  }

  PStatic Lookup(const Var& v) {
    auto rit = env_.rbegin();
    while (rit != env_.rend()) {
      if (rit->locals.find(v) != rit->locals.end()) {
        return rit->locals.find(v)->second;
      }
      ++rit;
    }
    LOG(FATAL) << "Unknown Variable: " << v;
    throw;
  }

 private:
  std::list<Frame> env_;

  struct FrameContext {
    Environment* env_;
    explicit FrameContext(Environment* env) : env_(env) {
      env_->env_.push_back(Frame());
    }
    ~FrameContext() {
      env_->env_.pop_back();
    }
  };
};

/*!
 * \brief As our store require rollback, we implement it as a frame.
 *
 * Every time we need to copy the store, a new frame is insert.
 * Every time we roll back, a frame is popped.
 */
struct StoreFrame {
  std::unordered_map<const SRefNode*, PStatic> store;
  /*!
   * \brief On unknown effect, history_valid is set to true to signal above frame is outdated.
   *
   * It only outdate the frame above it, but not the current frame.
   */
  bool history_valid = true;
  explicit StoreFrame(const std::unordered_map<const SRefNode*, PStatic>& store) : store(store) { }
  StoreFrame() = default;
};

class Store {
 public:
  Store() : store_({StoreFrame()}) { }
  Store(const Store&) = delete;

  template<typename T>
  T Extend(const std::function<T()>& body) {
    StoreFrameContext sfc(this);
    return body();
  }

  void Insert(const SRefNode* r, const PStatic& ps) {
    CHECK(r);
    store_.back().store[r] = ps;
  }

  // return null if not found
  PStatic Lookup(const SRefNode* r) {
    auto rit = store_.rbegin();
    while (rit != store_.rend()) {
      if (rit->store.find(r) != rit->store.end()) {
        return rit->store.find(r)->second;
      }
      if (!rit->history_valid) {
        return PStatic();
      }
      ++rit;
    }
    return PStatic();
  }

  void Invalidate() {
    StoreFrame sf;
    sf.history_valid = false;
    store_.push_back(sf);
  }

 private:
  std::list<StoreFrame> store_;

  struct StoreFrameContext {
    Store* store_;
    explicit StoreFrameContext(Store* store) : store_(store) {
      store_->store_.push_back(StoreFrame());
    }
    ~StoreFrameContext() {
      // push one history valid frame off.
      while (!store_->store_.back().history_valid) {
        store_->store_.pop_back();
      }
      store_->store_.pop_back();
    }
  };
};

PStatic HasStatic(const Static& stat, const Expr& dynamic) {
  CHECK(stat.defined());
  return PStatic(make_node<PStaticNode>(stat, dynamic));
}

PStatic NoStatic(const Expr& dynamic) {
  return PStatic(make_node<PStaticNode>(dynamic));
}

enum struct MatchStatus {
  Match, NoMatch, Unknown
};

bool StatefulOp(const Expr& e) {
  static auto op_stateful = Op::GetAttr<TOpIsStateful>("TOpIsStateful");
  struct StatefulOpVisitor : ExprVisitor {
    bool stateful = false;
    void VisitExpr_(const OpNode* op) {
      stateful = stateful || op_stateful.get(GetRef<Op>(op), false);
    }
  };
  StatefulOpVisitor sov;
  sov(e);
  return sov.stateful;
}

using FInterpreter = runtime::TypedPackedFunc<Value(Expr)>;

DLContext CPUContext() {
  DLContext ctx;
  ctx.device_type = kDLCPU;
  ctx.device_id = 0;
  return ctx;
}

FInterpreter CPUInterpreter() {
  Target target = Target::Create("llvm");
  // use a fresh build context
  // in case we are already in a build context.
  With<BuildConfig> fresh_build_ctx(BuildConfig::Create());

  return CreateInterpreter(Module(nullptr), CPUContext(), target);
}

using FuncId = int;

/*!
 * \brief Annotate a function with a FuncId.
 */
struct WithFuncIdAttrs : public tvm::AttrsNode<WithFuncIdAttrs> {
  FuncId fid;

  TVM_DECLARE_ATTRS(WithFuncIdAttrs, "relay.attrs.WithFuncIdAttrs") {
    TVM_ATTR_FIELD(fid)
      .describe("The FuncId that an function is annotated with.")
      .set_default(-1);
  }
};

TVM_REGISTER_NODE_TYPE(WithFuncIdAttrs);

Op WithFuncIdOp() {
  static const Op& op = Op::Get("annotation.with_funcid");
  return op;
}

Expr MkWithFuncId(const Expr& expr, FuncId fid) {
  auto attrs = make_node<WithFuncIdAttrs>();
  attrs->fid = fid;
  return CallNode::make(WithFuncIdOp(), {expr}, Attrs(attrs), {});
}

RELAY_REGISTER_OP("annotation.with_funcid")
.describe(R"code(Annotate a function with a funcid.)code"
TVM_ADD_FILELINE)
.set_num_inputs(1)
.add_argument("func", "Function", "The input data.");

Expr StripWithFuncId(const Expr& e);

Function AsFunc(const Expr& e) {
  if (e.as<FunctionNode>()) {
    return Downcast<Function>(e);
  } else if (const CallNode* c = e.as<CallNode>()) {
    CHECK(c->op.same_as(WithFuncIdOp()));
    CHECK_EQ(c->args.size(), 1);
    return AsFunc(c->args[0]);
  } else {
    LOG(FATAL) << "Unknown case";
    throw;
  }
}

class PartialEvaluator : public ExprFunctor<PStatic(const Expr& e, LetList* ll)>,
                         public PatternFunctor<MatchStatus(const Pattern&, const PStatic&)> {
 public:
  PartialEvaluator(const Module& mod) : mod_(mod) { }

  PStatic VisitExpr(const Expr& e, LetList* ll) final {
    PStatic ret = ExprFunctor<PStatic(const Expr&, LetList*)>::VisitExpr(e, ll);
    CHECK(IsAtomic(ret->dynamic)) << ret->dynamic;
    return ret;
  }

  PStatic VisitExpr_(const ConstantNode* op, LetList* ll) final {
    return HasStatic(MkSTensor(op->data.CopyTo(context_)), ll->Push(GetRef<Expr>(op)));
  }

  PStatic VisitExpr_(const TupleNode* op, LetList* ll) final {
    std::vector<PStatic> value;
    tvm::Array<Expr> expr;
    for (const Expr& e : op->fields) {
      PStatic ps = VisitExpr(e, ll);
      value.push_back(ps);
      expr.push_back(ps->dynamic);
    }
    return HasStatic(MkSTuple(value), ll->Push(TupleNode::make(expr)));
  }

  PStatic VisitExpr_(const TupleGetItemNode* op, LetList* ll) final {
    PStatic ps = VisitExpr(op->tuple, ll);
    if (ps->pstatic.defined()) {
      return Downcast<STuple>(ps->pstatic)->fields[op->index];
    } else {
      return NoStatic(ll->Push(TupleGetItemNode::make(ps->dynamic, op->index)));
    }
  }

  PStatic VisitExpr_(const VarNode* op, LetList* ll) final {
    return env_.Lookup(GetRef<Var>(op));
  }

  PStatic VisitGlobalVar(const GlobalVar& gv) {
    CHECK(mod_.defined());
    if (gv_map_.count(gv) == 0) {
      Function func = mod_->Lookup(gv);
      InitializeFuncId(func);
      Func f = VisitFuncStatic(func, gv);
      gv_map_.insert({gv, HasStatic(MkSFunc(f), gv)});
      func = AsFunc(PostProcess(VisitFuncDynamic(func, f)));
      mod_->Update(gv, func);
    }
    return gv_map_.at(gv);
  }

  PStatic VisitExpr_(const GlobalVarNode* op, LetList* ll) final {
    return VisitGlobalVar(GetRef<GlobalVar>(op));
  }

  PStatic VisitExpr_(const LetNode* op, LetList* ll) final {
    env_.Insert(op->var, VisitExpr(op->value, ll));
    return VisitExpr(op->body, ll);
  }

  PStatic VisitExpr_(const IfNode* op, LetList* ll) final {
    PStatic c = VisitExpr(op->cond, ll);
    if (c->pstatic.defined()) {
      NDArray cpu_array = Downcast<STensor>(c->pstatic)->data.CopyTo(CPUContext());
      CHECK_EQ(TVMType2Type(cpu_array->dtype), Bool());
      if (reinterpret_cast<uint8_t*>(cpu_array->data)[0]) {
        return VisitExpr(op->true_branch, ll);
      } else {
        return VisitExpr(op->false_branch, ll);
      }
    } else {
      Expr t = store_.Extend<Expr>([&]() {
          return LetList::With([&](LetList* ll) {
              return VisitExpr(op->true_branch, ll)->dynamic;
            });
        });
      Expr f = store_.Extend<Expr>([&]() {
          return LetList::With([&](LetList* ll) {
              return VisitExpr(op->false_branch, ll)->dynamic;
            });
        });
      store_.Invalidate();
      return NoStatic(ll->Push(IfNode::make(c->dynamic, t, f)));
    }
  }

  PStatic VisitExpr_(const RefCreateNode* op, LetList* ll) final {
    PStatic ps = VisitExpr(op->value, ll);
    Static r = MkSRef();
    store_.Insert(r.as<SRefNode>(), ps);
    return HasStatic(r, ll->Push(RefCreateNode::make(ps->dynamic)));
  }

  PStatic VisitExpr_(const RefWriteNode* op, LetList* ll) final {
    PStatic r = VisitExpr(op->ref, ll);
    PStatic v = VisitExpr(op->value, ll);
    if (r->pstatic.defined()) {
      store_.Insert(r->pstatic.as<SRefNode>(), v);
    } else {
      store_.Invalidate();
    }
    return HasStatic(MkSTuple({}), ll->Push(RefWriteNode::make(r->dynamic, v->dynamic)));
  }

  PStatic VisitExpr_(const RefReadNode* op, LetList* ll) final {
    PStatic r = VisitExpr(op->ref, ll);
    if (r->pstatic.defined()) {
      PStatic ret = store_.Lookup(r->pstatic.as<SRefNode>());
      if (ret) {
        return ret;
      }
    }
    return NoStatic(ll->Push(RefReadNode::make(r->dynamic)));
  }

  PStatic VisitExpr_(const CallNode* op, LetList* ll) final {
    if (op->op.same_as(WithFuncIdOp())) {
      CHECK_EQ(op->args.size(), 1);
      return VisitExpr(op->args[0], ll);
    }
    PStatic f = VisitExpr(op->op, ll);
    std::vector<PStatic> x;
    tvm::Array<Expr> x_dyn;
    for (const Expr& e : op->args) {
      PStatic ps = VisitExpr(e, ll);
      x.push_back(ps);
      x_dyn.push_back(ps->dynamic);
    }
    if (f->pstatic.defined()) {
      return Downcast<SFunc>(f->pstatic)->func(x, op->attrs, op->type_args, ll);
    } else {
      store_.Invalidate();
      return NoStatic(ll->Push(CallNode::make(f->dynamic, x_dyn, op->attrs, op->type_args)));
    }
  }

  struct TimeFrame {
    PartialEvaluator* pe_;
    FuncId fid_;
    std::vector<Time> old_time;
    bool has_old_time;
    TimeFrame(PartialEvaluator* pe,
              FuncId fid,
              const std::vector<Time>& args_time) : pe_(pe), fid_(fid) {
      has_old_time = pe_->time_map_.count(fid_) > 0;
      old_time = pe_->time_map_[fid_];
      pe_->time_map_[fid_] = args_time;
    }
    ~TimeFrame() {
      if (has_old_time) {
        pe_->time_map_[fid_] = old_time;
      } else {
        pe_->time_map_.erase(fid_);
      }
    }
  };

  Func VisitFuncStatic(const Function& func, const Expr& var) {
    CHECK(IsAtomic(var));
    if (func->IsPrimitive()) {
      return ConstEvaluateFunc(func);
    }
    std::vector<std::pair<Var, PStatic> > free_vars;
    for (const auto& v : FreeVars(func)) {
      free_vars.push_back(std::pair<Var, PStatic>(v, env_.Lookup(v)));
    }
    return [=](const std::vector<PStatic>& pv,
               const Attrs& attrs,
               const tvm::Array<Type>& type_args,
               LetList* ll) {
      return env_.Extend<PStatic>([&]() {
          CHECK_EQ(pv.size(), func->params.size());
          for (size_t i = 0; i < pv.size(); ++i) {
            env_.Insert(func->params[i], pv[i]);
          }
          for (const auto& p : free_vars) {
            env_.Insert(p.first, p.second);
          }
          tvm::Map<TypeVar, Type> subst;
          for (size_t i = 0; i < type_args.size(); ++i) {
            subst.Set(func->type_params[i], type_args[i]);
          }
          for (size_t i = type_args.size(); i < func->type_params.size(); ++i) {
            subst.Set(func->type_params[i], IncompleteTypeNode::make(kType));
          }
          std::vector<Time> args_time;
          for (const auto& v : pv) {
            args_time.push_back(v->created_time);
          }
          CHECK_GT(func_map_.count(func), 0);
          FuncId fid = func_map_.at(func);
          auto recurse = [&]() {
            TimeFrame tf(this, fid, args_time);
            return VisitExpr(RegisterFuncId(TypeSubst(AnnotateFuncId(func->body), subst)), ll);
          };
          if (time_map_.count(fid) == 0) {
            return recurse();
          } else {
            /* We check to see that at least one argument decrease
             * with respect to all previous invocation.
             * The depth of the recursion is bounded by
             * the sum of the time of all argument at the first call.
             */
            bool can_recurse = false;
            std::vector<Time>& min_time = time_map_.at(fid);
            CHECK_EQ(args_time.size(), min_time.size());
            for (size_t i = 0; i < args_time.size(); ++i) {
              if (args_time[i] < min_time[i]) {
                can_recurse = true;
              }
              args_time[i] = std::min(args_time[i], min_time[i]);
            }
            if (can_recurse) {
              return recurse();
            } else {
              std::vector<Expr> dyn;
              for (const auto& v : pv) {
                dyn.push_back(v->dynamic);
              }
              return NoStatic(ll->Push(CallNode::make(var, dyn, attrs, type_args)));
            }
          }
        });
    };
  }

  Expr VisitFuncDynamic(const Function& func, const Func& f) {
    return store_.Extend<Expr>([&]() {
      store_.Invalidate();
      return FunctionNode::make(func->params,
                                LetList::With([&](LetList* ll) {
        std::vector<PStatic> pv;
        for (const auto& v : func->params) {
          pv.push_back(NoStatic(v));
        }
        tvm::Array<Type> type_args;
        for (const auto& tp : func->type_params) {
          type_args.push_back(tp);
        }
        return f(pv, Attrs(), type_args, ll)->dynamic;
      }), func->ret_type, func->type_params, func->attrs);
    });
  }

  PStatic VisitFunc(const Function& func, LetList* ll) {
    Var v = VarNode::make("x", Type());
    Func f = VisitFuncStatic(func, v);
    Function u_func = AsFunc(RegisterFuncId(DeDup(AnnotateFuncId(func))));
    // TODO(@M.K.): we seems to reduce landin knot into letrec.
    // restore letrec support across whole relay.
    return HasStatic(MkSFunc(f),
                     ll->Push(v, VisitFuncDynamic(u_func, f)));
  }

  PStatic VisitExpr_(const FunctionNode* op, LetList* ll) final {
    return VisitFunc(GetRef<Function>(op), ll);
  }

  Expr Reflect(const PStatic& st) {
    if (const STensorNode* op = st->pstatic.as<STensorNode>()) {
      return ConstantNode::make(op->data);
    } else if (const STupleNode* op = st->pstatic.as<STupleNode>()) {
      tvm::Array<Expr> fields;
      for (const PStatic& field : op->fields) {
        fields.push_back(Reflect(field));
      }
      return TupleNode::make(fields);
    } else {
      LOG(FATAL) << "Unknown case";
      throw;
    }
  }

  PStatic Reify(const Value& v, LetList* ll) const {
    if (const TensorValueNode* op = v.as<TensorValueNode>()) {
      return HasStatic(MkSTensor(op->data), ll->Push(ConstantNode::make(op->data)));
    } else if (const TupleValueNode* op = v.as<TupleValueNode>()) {
      std::vector<PStatic> fields;
      tvm::Array<Expr> fields_dyn;
      for (const Value& field : op->fields) {
        PStatic ps = Reify(field, ll);
        fields.push_back(ps);
        fields_dyn.push_back(ps->dynamic);
      }
      return HasStatic(MkSTuple(fields), ll->Push(TupleNode::make(fields_dyn)));
    } else {
      LOG(FATAL) << "Unknown case";
      throw;
    }
  }

  // Constant evaluate a expression.
  PStatic ConstEvaluate(const Expr& expr, LetList* ll) {
    std::vector<transform::Pass> passes = {transform::FuseOps(0),
                                           transform::InferType()};
    auto mod = ModuleNode::FromExpr(expr);
    auto seq = transform::Sequential(passes);
    mod = seq(mod);
    auto entry_func = mod->Lookup("main");
    auto fused_infered =
        expr.as<FunctionNode>() == nullptr ? entry_func->body : entry_func;
    return Reify(executor_(fused_infered), ll);
  }

  Func ConstEvaluateFunc(const Expr& expr) {
    CHECK_EQ(FreeVars(expr).size(), 0);
    return [=](const std::vector<PStatic>& pv,
               const Attrs& attrs,
               const tvm::Array<Type>& type_args,
               LetList* ll) {
      tvm::Array<Expr> ns_args;
      for (const PStatic& ps : pv) {
        ns_args.push_back(ps->dynamic);
      }
      PStatic ns = NoStatic(ll->Push(CallNode::make(expr, ns_args, attrs, type_args)));
      if (StatefulOp(expr)) {
        return ns;
      }
      tvm::Array<Expr> args;
      for (const PStatic& ps : pv) {
        if (ps->pstatic.defined()) {
          args.push_back(Reflect(ps));
        } else {
          return ns;
        }
      }
      return ConstEvaluate(CallNode::make(expr, args, attrs, type_args), ll);
    };
  }

  PStatic VisitExpr_(const OpNode* op, LetList* ll) final {
    return HasStatic(MkSFunc(ConstEvaluateFunc(GetRef<Expr>(op))), GetRef<Expr>(op));
  }

  PStatic VisitExpr_(const ConstructorNode* op, LetList* ll) final {
    Constructor c = GetRef<Constructor>(op);
    Func f = [=](const std::vector<PStatic>& pv,
                 const Attrs& attrs,
                 const tvm::Array<Type>& type_args,
                 LetList* ll) {
      tvm::Array<Expr> dyn;
      for (const PStatic& ps : pv) {
        dyn.push_back(ps->dynamic);
      }
      return HasStatic(MkSConstructor(c, pv), ll->Push(CallNode::make(c, dyn)));
    };
    return HasStatic(MkSFunc(f), GetRef<Expr>(op));
  }

  PStatic VisitExpr_(const MatchNode* op, LetList* ll) final {
    PStatic ps = VisitExpr(op->data, ll);
    return env_.Extend<PStatic>([&]() {
        for (const Clause& c : op->clauses) {
          switch (VisitPattern(c->lhs, ps)) {
          case MatchStatus::Match:
            return VisitExpr(c->rhs, ll);
          case MatchStatus::NoMatch:
            continue;
          case MatchStatus::Unknown:
            tvm::Array<Clause> clauses;
            for (const Clause& c : op->clauses) {
              Expr expr = store_.Extend<Expr>([&]() {
                  return LetList::With([&](LetList* ll) {
                      for (const Var& v : BoundVars(c->lhs)) {
                        env_.Insert(v, NoStatic(v));
                      }
                      return VisitExpr(c->rhs, ll)->dynamic;
                    });
                });
              clauses.push_back(ClauseNode::make(c->lhs, expr));
            }
            store_.Invalidate();
            return NoStatic(ll->Push(MatchNode::make(ps->dynamic, clauses)));
          }
        }
        LOG(FATAL) << "No case Match";
        throw;
      });
  }

  MatchStatus VisitPattern_(const PatternWildcardNode* op, const PStatic& ps) final {
    return MatchStatus::Match;
  }

  MatchStatus VisitPattern_(const PatternVarNode* op, const PStatic& ps) final {
    env_.Insert(op->var, ps);
    return MatchStatus::Match;
  }

  MatchStatus VisitPattern_(const PatternConstructorNode* op, const PStatic& ps) final {
    if (ps->pstatic.defined()) {
      SConstructor scn = Downcast<SConstructor>(ps->pstatic);
      CHECK_NE(op->constructor->tag, -1);
      CHECK_NE(scn->constructor->tag, -1);
      if (op->constructor->tag == scn->constructor->tag) {
        CHECK_EQ(op->patterns.size(), scn->fields.size());
        MatchStatus current_match_status = MatchStatus::Match;
        for (size_t i = 0; i < op->patterns.size(); ++i) {
          MatchStatus ms = VisitPattern(op->patterns[i], scn->fields[i]);
          switch (ms) {
          case MatchStatus::Match:
            continue;
          case MatchStatus::NoMatch:
            return MatchStatus::NoMatch;
          case MatchStatus::Unknown:
            current_match_status = MatchStatus::Unknown;
          }
        }
        return current_match_status;
      }
      return MatchStatus::NoMatch;
    } else {
      return MatchStatus::Unknown;
    }
  }

  void InitializeFuncId(const Expr& e) {
    struct InitializeFuncIdVisitor : ExprVisitor, PatternVisitor {
      PartialEvaluator* pe;
      explicit InitializeFuncIdVisitor(PartialEvaluator* pe) : pe(pe) { }

      void VisitExpr_(const FunctionNode* op) final {
        Function f = GetRef<Function>(op);
        CHECK_EQ(pe->func_map_.count(f), 0);
        pe->func_map_.insert({f, pe->func_map_.size()});
        VisitExpr(f->body);
      }

      void VisitPattern(const Pattern& p) final {
        PatternVisitor::VisitPattern(p);
      }
    };
    InitializeFuncIdVisitor(this).VisitExpr(e);
  }

  Expr RegisterFuncId(const Expr& e) {
    struct RegisterFuncIdVisitor : ExprVisitor, PatternVisitor {
      PartialEvaluator* pe;
      explicit RegisterFuncIdVisitor(PartialEvaluator* pe) : pe(pe) { }

      void VisitExpr_(const CallNode* op) final {
        if (op->op.same_as(WithFuncIdOp())) {
          CHECK_EQ(op->args.size(), 1);
          CHECK(op->attrs.defined());
          CHECK(op->attrs.as<WithFuncIdAttrs>());
          Function f = AsFunc(op->args[0]);
          FuncId fid = op->attrs.as<WithFuncIdAttrs>()->fid;
          if (pe->func_map_.count(f) != 0) {
            CHECK_EQ(pe->func_map_.at(f), fid);
          }
          pe->func_map_.insert({f, fid});
        }
        ExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const FunctionNode* op) final {
        Function f = GetRef<Function>(op);
        CHECK_GT(pe->func_map_.count(f), 0);
        ExprVisitor::VisitExpr_(op);
      }

      void VisitPattern(const Pattern& p) final {
        PatternVisitor::VisitPattern(p);
      }
    };
    RegisterFuncIdVisitor(this).VisitExpr(e);
    return e;
  }

  Expr AnnotateFuncId(const Expr& e) {
    struct AnnotateFuncIdMutator : ExprMutator, PatternMutator {
      PartialEvaluator* pe;
      explicit AnnotateFuncIdMutator(PartialEvaluator* pe) : pe(pe) { }

      Expr VisitExpr_(const FunctionNode* op) final {
        Function f = GetRef<Function>(op);
        CHECK_GT(pe->func_map_.count(f), 0);
        return MkWithFuncId(ExprMutator::VisitExpr_(op), pe->func_map_.at(f));
      }

      Pattern VisitPattern(const Pattern& p) final {
        return PatternMutator::VisitPattern(p);
      }

      Var VisitVar(const Var& v) final {
        return v;
      }
    };
    return AnnotateFuncIdMutator(this).VisitExpr(e);
  }

 private:
  Environment env_;
  Module mod_;
  std::unordered_map<GlobalVar, PStatic, NodeHash, NodeEqual> gv_map_;
  /*! Termination checking is done as follows:
   *  We have finitely many FunctionIds.
   *  Each FunctionId maps to a class of semantically equivalent function (ignoring type),
   *  as both TypeSubst and DeDup create semantically equivalent function.
   *  We partially map each FunctionId to a std::vector<Time>,
   *  denoting the minimal TimeFrame of each argument of the function.
   *  Every time we try to inline a Function,
   *  we make sure it either does not have a vector<Time>, which means this is the initial call,
   *  or some argument has a lesser time, which means some earlier argument is passed in.
   *  In any case, we remap the mapping to a minimal vector<Time> across all previous invocations
   *  when we PE inside the Function body.
   *  Termination is guaranteed because the creation time of at least one argument will decrease every call.
   */
  std::unordered_map<Function, FuncId, NodeHash, NodeEqual> func_map_;
  std::unordered_map<FuncId, std::vector<Time> > time_map_;
  Store store_;
  DLContext context_ = CPUContext();
  FInterpreter executor_ = CPUInterpreter();
};

/*! \brief Remap multiple Var sharing the same Id into the same Var. */
Expr Remap(const Expr& e) {
  class RemapMutator : public ExprMutator, public PatternMutator {
    Expr VisitExpr_(const VarNode* op) final {
      Var v = GetRef<Var>(op);
      if (remap_.count(v) == 0) {
        remap_.insert({v, v});
      }
      return remap_.at(v);
    }

    Var VisitVar(const Var& v) final {
      return Downcast<Var>(VisitExpr(v));
    }

   private:
    std::unordered_map<Var, Var, VarHash, VarEqual> remap_;
  };
  return RemapMutator().VisitExpr(e);
}

Expr StripWithFuncId(const Expr& e) {
  struct StripWithFuncIdMutator : ExprMutator, PatternMutator {
    Expr VisitExpr_(const CallNode* op) final {
      if (op->op.same_as(WithFuncIdOp())) {
        CHECK_EQ(op->args.size(), 1);
        return VisitExpr(op->args[0]);
      } else {
        return ExprMutator::VisitExpr_(op);
      }
    }

    Pattern VisitPattern(const Pattern& p) final {
      return PatternMutator::VisitPattern(p);
    }

    Var VisitVar(const Var& v) final {
      return v;
    }
  };
  return StripWithFuncIdMutator().VisitExpr(e);
}

Expr PostProcess(const Expr& e) {
  return StripWithFuncId(DeDup(Remap(e)));
}

}  // namespace partial_eval

Module PartialEval(const Module& m) {
  relay::partial_eval::PartialEvaluator pe(m);
  std::vector<GlobalVar> gvs;
  for (const auto& p : m->functions) {
    gvs.push_back(p.first);
  }
  for (const auto& gv : gvs) {
    pe.VisitGlobalVar(gv);
  }
  return m;
}

namespace transform {

Pass PartialEval() {
  runtime::TypedPackedFunc<Module(Module, PassContext)> pass_func =
    [=](Module m, PassContext pc) {
    return PartialEval(m);
  };
  return CreateModulePass(pass_func, 1, "PartialEvaluate", {});
}

TVM_REGISTER_API("relay._transform.PartialEvaluate")
.set_body_typed(PartialEval);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
