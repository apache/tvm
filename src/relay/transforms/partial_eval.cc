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
 * While it is permitted, most pass use ObjectPtrHash for Var,
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
#include <tvm/ir/type_functor.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/feature.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relay/transform.h>

#include "let_list.h"
#include "pass_utils.h"

namespace tvm {
namespace relay {
namespace partial_eval {

using namespace runtime;

/*! \brief Hash Var by it's id.
 * Different VarNode might has same vid, and they are considered to be the same var in such case.
 * Use VarHash to hash Var by id.
 */
struct VarHash {
  size_t operator()(const Var& v) const { return ObjectPtrHash()(v->vid); }
};

/*! \brief Compare Var by it's id.
 * Different VarNode might has same vid, and they are considered to be the same var in such case.
 * Use VarEqual to compare Var by id.
 */
struct VarEqual {
  bool operator()(const Var& l, const Var& r) const { return l->vid.get() == r->vid.get(); }
};

Expr PostProcess(const Expr&);

/*! \brief A StaticNode contains some static data that the Partial Evaluator can use. */
class StaticNode : public RelayNode {
 public:
  static constexpr const char* _type_key = "relay.Static";
  TVM_DECLARE_BASE_OBJECT_INFO(StaticNode, RelayNode);
};

class Static : public ObjectRef {
 public:
  Static() {}
  explicit Static(ObjectPtr<Object> n) : ObjectRef(n) {}
  const StaticNode* operator->() const { return static_cast<const StaticNode*>(get()); }

  using ContainerType = StaticNode;
};

using Time = size_t;

struct PStaticNode : Object {
  static Time time() {
    static Time time_ = 0;
    Time ret = time_;
    time_++;
    return ret;
  }
  Static pstatic;  // may be null
  Expr dynamic;
  Time created_time;
  PStaticNode(const Static& pstatic, const Expr& dynamic)
      : pstatic(pstatic), dynamic(dynamic), created_time(time()) {}
  explicit PStaticNode(const Expr& dynamic) : PStaticNode(Static(), dynamic) {}
  static constexpr const char* _type_key = "relay.PStatic";
  TVM_DECLARE_FINAL_OBJECT_INFO(PStaticNode, Object);
};

class PStatic : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(PStatic, ObjectRef, PStaticNode);
};

struct STupleNode : StaticNode {
  std::vector<PStatic> fields;
  explicit STupleNode(const std::vector<PStatic>& fields) : fields(fields) {}
  static constexpr const char* _type_key = "relay.STuple";
  TVM_DECLARE_FINAL_OBJECT_INFO(STupleNode, StaticNode);
};

class STuple : public Static {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(STuple, Static, STupleNode);
};

Static MkSTuple(const std::vector<PStatic>& fields) {
  return Static(make_object<STupleNode>(fields));
}

struct STensorNode : StaticNode {
  runtime::NDArray data;
  explicit STensorNode(const NDArray& data) : data(data) {}
  static constexpr const char* _type_key = "relay.STensor";
  TVM_DECLARE_FINAL_OBJECT_INFO(STensorNode, StaticNode);
};

class STensor : public Static {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(STensor, Static, STensorNode);
};

Static MkSTensor(const NDArray& data) { return Static(make_object<STensorNode>(data)); }

struct SConstructorNode : StaticNode {
  Constructor constructor;
  std::vector<PStatic> fields;
  SConstructorNode(const Constructor& constructor, const std::vector<PStatic>& fields)
      : constructor(constructor), fields(fields) {}
  static constexpr const char* _type_key = "relay.SConstructor";
  TVM_DECLARE_FINAL_OBJECT_INFO(SConstructorNode, StaticNode);
};

class SConstructor : public Static {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(SConstructor, Static, SConstructorNode);
};

Static MkSConstructor(const Constructor& constructor, const std::vector<PStatic>& fields) {
  return Static(make_object<SConstructorNode>(constructor, fields));
}

struct SRefNode : StaticNode {
  static constexpr const char* _type_key = "relay.SRef";
  // we will use the address as the guid for hashing
  TVM_DECLARE_FINAL_OBJECT_INFO(SRefNode, StaticNode);
};

class SRef : public Static {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(SRef, Static, SRefNode);
};

Static MkSRef() { return Static(make_object<SRefNode>()); }

using Func = std::function<PStatic(const PStatic&, const std::vector<PStatic>&, const Attrs&,
                                   const Array<Type>&, LetList*)>;

struct SFuncNode : StaticNode {
  Func func;
  explicit SFuncNode(const Func& func) : func(func) {}
  static constexpr const char* _type_key = "relay.SFunc";
  TVM_DECLARE_FINAL_OBJECT_INFO(SFuncNode, StaticNode);
};

class SFunc : public Static {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(SFunc, Static, SFuncNode);
};

Static MkSFunc(const Func& func) { return Static(make_object<SFuncNode>(func)); }

class FuelNode;
/*! \brief A meet-semilattice with finite descending chain.
 * It means that we can meet two element to get an element,
 * and for every element, there is only a finite amount of meet before getting back the same
 * element.
 *
 * Every time we recurse, we do a meet and require that progress must be made.
 * This ensures we do not recurse infinitely in the Partial Evaluator.
 */
class Fuel : public ObjectRef {
 public:
  Fuel() {}
  explicit Fuel(ObjectPtr<Object> n) : ObjectRef(n) {}
  const FuelNode* operator->() const;

  using ContainerType = FuelNode;
};

class FuelNode : public RelayNode {
 public:
  virtual ~FuelNode() {}
  // Please implement one of the following function or there will be infinite loop.
  /*! \brief return the new Fuel, and whether progress is made.
   *
   * Note that progress is not symmetric - it only measure progress for (*this).
   *
   * Thus, if the generated is smaller then the argument of Meet,
   * and the generated is not smaller then (*this),
   * progress should be false.
   */
  virtual std::tuple<Fuel, bool> Meet(const Fuel& f) const {
    bool progress = false;
    auto ret = Meet(f, &progress);
    return std::make_tuple(ret, progress);
  }
  /*! \brief return the new Fuel, and write (*progress | is progress made) to *progress. */
  virtual Fuel Meet(const Fuel& f, bool* progress) const {
    ICHECK(progress);
    auto ret = Meet(f);
    *progress |= std::get<1>(ret);
    return std::get<0>(ret);
  }
  static constexpr const char* _type_key = "relay.Fuel";
  TVM_DECLARE_BASE_OBJECT_INFO(FuelNode, RelayNode);
};

const FuelNode* Fuel::operator->() const { return static_cast<const FuelNode*>(get()); }

Fuel MkFSeq(const std::vector<Fuel>& fuels);
struct FSeqNode : FuelNode {
  std::vector<Fuel> fuels;
  Fuel Meet(const Fuel& f, bool* progress) const final {
    auto x = f.as<FSeqNode>();
    ICHECK(x);
    ICHECK_EQ(fuels.size(), x->fuels.size());
    std::vector<Fuel> new_fuels;
    for (size_t i = 0; i < fuels.size(); ++i) {
      new_fuels.push_back(fuels[i]->Meet(x->fuels[i], progress));
    }
    return MkFSeq(new_fuels);
  }
  explicit FSeqNode(const std::vector<Fuel>& fuels) : fuels(fuels) {}
  static constexpr const char* _type_key = "relay.FSeq";
  TVM_DECLARE_FINAL_OBJECT_INFO(FSeqNode, FuelNode);
};

class FSeq : public Fuel {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(FSeq, Fuel, FSeqNode);
};

Fuel MkFSeq(const std::vector<Fuel>& fuels) { return Fuel(make_object<FSeqNode>(fuels)); }

Fuel MkFTime(Time time);
struct FTimeNode : FuelNode {
  Time time;
  std::tuple<Fuel, bool> Meet(const Fuel& f) const final {
    auto x = f.as<FTimeNode>();
    ICHECK(x);
    Time new_time = std::min(time, x->time);
    return std::make_tuple(MkFTime(new_time), new_time < time);
  }
  explicit FTimeNode(Time time) : time(time) {}
  static constexpr const char* _type_key = "relay.FTime";
  TVM_DECLARE_FINAL_OBJECT_INFO(FTimeNode, FuelNode);
};

class FTime : public Fuel {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(FTime, Fuel, FTimeNode);
};

Fuel MkFTime(Time time) { return Fuel(make_object<FTimeNode>(time)); }

Fuel MkFTValue(size_t tvalue);
/*! \brief If the pstatic is hold a positive integer scalar, that number, else 0. */
struct FTValueNode : FuelNode {
  size_t tvalue;
  std::tuple<Fuel, bool> Meet(const Fuel& f) const final {
    auto x = f.as<FTValueNode>();
    ICHECK(x);
    size_t new_tvalue = std::min(tvalue, x->tvalue);
    return std::make_tuple(MkFTValue(new_tvalue), new_tvalue < tvalue);
  }
  explicit FTValueNode(size_t tvalue) : tvalue(tvalue) {}
  static constexpr const char* _type_key = "relay.FTValue";
  TVM_DECLARE_FINAL_OBJECT_INFO(FTValueNode, FuelNode);
};

class FTValue : public Fuel {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(FTValue, Fuel, FTValueNode);
};

Fuel MkFTValue(size_t tvalue) { return Fuel(make_object<FTValueNode>(tvalue)); }

/*! \brief Initially every element has Fuel of FTop. It is the largest element.
 *
 * Note that it is illegal to has FTop inside some other Fuel -
 * doing so break the finite descending chain property.
 */
struct FTopNode : FuelNode {
  std::tuple<Fuel, bool> Meet(const Fuel& f) const final {
    return std::make_tuple(f, !f.as<FTopNode>());
  }
  static constexpr const char* _type_key = "relay.FTop";
  TVM_DECLARE_FINAL_OBJECT_INFO(FTopNode, FuelNode);
};

class FTop : public Fuel {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(FTop, Fuel, FTopNode);
};

Fuel MkFTop() { return Fuel(make_object<FTopNode>()); }

/*!
 * \brief A stack frame in the Relay interpreter.
 *
 * Contains a mapping from relay::Var to relay::Object.
 */
struct Frame {
  /*! \brief The set of local variables and arguments for the frame. */
  std::unordered_map<Var, PStatic, VarHash, VarEqual> locals;
  Frame() = default;
};

class Environment {
 public:
  Environment() : env_({Frame()}) {}
  Environment(const Environment&) = delete;

  template <typename T>
  T Extend(const std::function<T()>& body) {
    FrameContext fc(this);
    return body();
  }

  void Insert(const Var& v, const PStatic& ps) {
    ICHECK(ps.defined());
    ICHECK_GT(env_.size(), 0);
    ICHECK_EQ(env_.back().locals.count(v), 0);
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
    explicit FrameContext(Environment* env) : env_(env) { env_->env_.push_back(Frame()); }
    ~FrameContext() { env_->env_.pop_back(); }
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
  explicit StoreFrame(const std::unordered_map<const SRefNode*, PStatic>& store) : store(store) {}
  StoreFrame() = default;
};

class Store {
 public:
  Store() : store_({StoreFrame()}) {}
  Store(const Store&) = delete;

  template <typename T>
  T Extend(const std::function<T()>& body) {
    StoreFrameContext sfc(this);
    return body();
  }

  void Insert(const SRefNode* r, const PStatic& ps) {
    ICHECK(r);
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
  ICHECK(stat.defined());
  return PStatic(make_object<PStaticNode>(stat, dynamic));
}

PStatic NoStatic(const Expr& dynamic) { return PStatic(make_object<PStaticNode>(dynamic)); }

enum struct MatchStatus { Match, NoMatch, Unknown };

bool StatefulOp(const Expr& e) {
  static auto op_stateful = Op::GetAttrMap<TOpIsStateful>("TOpIsStateful");
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

using FInterpreter = runtime::TypedPackedFunc<ObjectRef(Expr)>;

Target CPUTarget() { return Target("llvm"); }

Device CPUDevice() {
  Device dev;
  dev.device_type = kDLCPU;
  dev.device_id = 0;
  return dev;
}

using FuncId = int;

/*!
 * \brief Annotate a function with a FuncId.
 */
struct WithFuncIdAttrs : public tvm::AttrsNode<WithFuncIdAttrs> {
  FuncId fid;

  TVM_DECLARE_ATTRS(WithFuncIdAttrs, "relay.attrs.WithFuncIdAttrs") {
    TVM_ATTR_FIELD(fid).describe("The FuncId that an function is annotated with.").set_default(-1);
  }
};

TVM_REGISTER_NODE_TYPE(WithFuncIdAttrs);

RELAY_REGISTER_OP("annotation.with_funcid")
    .describe(R"code(Annotate a function with a funcid.)code" TVM_ADD_FILELINE)
    .set_num_inputs(1)
    .add_argument("func", "Function", "The input data.");

// Cache with_funcid op to reduce lookup overhead during traversal.
static const Op& with_funcid_op = Op::Get("annotation.with_funcid");

Expr MkWithFuncId(const Expr& expr, FuncId fid) {
  auto attrs = make_object<WithFuncIdAttrs>();
  attrs->fid = fid;
  return Call(with_funcid_op, {expr}, Attrs(attrs), {});
}

Expr StripWithFuncId(const Expr& e);

Function AsFunc(const Expr& e) {
  if (e.as<FunctionNode>()) {
    return Downcast<Function>(e);
  } else if (const CallNode* c = e.as<CallNode>()) {
    ICHECK(c->op == with_funcid_op);
    ICHECK_EQ(c->args.size(), 1);
    return AsFunc(c->args[0]);
  } else {
    LOG(FATAL) << "Unknown case";
    throw;
  }
}

class PartialEvaluator : public ExprFunctor<PStatic(const Expr& e, LetList* ll)>,
                         public PatternFunctor<MatchStatus(const Pattern&, const PStatic&)> {
 public:
  PartialEvaluator(const IRModule& mod) : mod_(mod) {}

  PStatic VisitExpr(const Expr& e, LetList* ll) final {
    PStatic ret = ExprFunctor<PStatic(const Expr&, LetList*)>::VisitExpr(e, ll);
    ICHECK(IsAtomic(ret->dynamic)) << ret->dynamic;
    return ret;
  }

  PStatic VisitExpr(const Expr& e, LetList* ll, const Var& name) {
    if (const CallNode* c = e.as<CallNode>()) {
      if (c->op == with_funcid_op) {
        ICHECK_EQ(c->args.size(), 1);
        return VisitExpr(c->args[0], ll, name);
      }
    }
    PStatic ret =
        e.as<FunctionNode>() ? VisitFunc(Downcast<Function>(e), ll, name) : VisitExpr(e, ll);
    ICHECK(IsAtomic(ret->dynamic)) << ret->dynamic;
    return ret;
  }

  PStatic VisitExpr_(const ConstantNode* op, LetList* ll) final {
    return HasStatic(MkSTensor(op->data.CopyTo(device_)), ll->Push(GetRef<Expr>(op)));
  }

  PStatic VisitExpr_(const TupleNode* op, LetList* ll) final {
    std::vector<PStatic> value;
    tvm::Array<Expr> expr;
    for (const Expr& e : op->fields) {
      PStatic ps = VisitExpr(e, ll);
      value.push_back(ps);
      expr.push_back(ps->dynamic);
    }
    // Note: The partial evaluator seems to do some weird stuff with sharing. Changing Tuple(expr)
    // to WithFields(op, expr) causes failures in the partial evaluator tests.
    return HasStatic(MkSTuple(value), ll->Push(Tuple(expr)));
  }

  PStatic VisitExpr_(const TupleGetItemNode* op, LetList* ll) final {
    PStatic ps = VisitExpr(op->tuple, ll);
    if (ps->pstatic.defined()) {
      return Downcast<STuple>(ps->pstatic)->fields[op->index];
    } else {
      return NoStatic(ll->Push(TupleGetItem(ps->dynamic, op->index)));
    }
  }

  PStatic VisitExpr_(const VarNode* op, LetList* ll) final { return env_.Lookup(GetRef<Var>(op)); }

  PStatic VisitGlobalVar(const GlobalVar& gv) {
    ICHECK(mod_.defined());
    if (gv_map_.count(gv) == 0) {
      BaseFunc base_func = mod_->Lookup(gv);
      if (auto opt = base_func.as<Function>()) {
        auto func = opt.value();
        InitializeFuncId(func);
        Func f = VisitFuncStatic(func, gv);
        gv_map_.insert({gv, HasStatic(MkSFunc(f), gv)});
        func = AsFunc(PostProcess(VisitFuncDynamic(func, f, gv)));
        mod_->Update(gv, func);
        return gv_map_.at(gv);
      } else {
        return NoStatic(gv);
      }
    }
    return gv_map_.at(gv);
  }

  PStatic VisitExpr_(const GlobalVarNode* op, LetList* ll) final {
    return VisitGlobalVar(GetRef<GlobalVar>(op));
  }

  PStatic VisitExpr_(const LetNode* op, LetList* ll) final {
    env_.Insert(op->var, VisitExpr(op->value, ll, op->var));
    return VisitExpr(op->body, ll);
  }

  PStatic VisitExpr_(const IfNode* op, LetList* ll) final {
    PStatic c = VisitExpr(op->cond, ll);
    if (c->pstatic.defined()) {
      NDArray cpu_array = Downcast<STensor>(c->pstatic)->data.CopyTo(CPUDevice());
      ICHECK_EQ(DataType(cpu_array->dtype), DataType::Bool());
      if (reinterpret_cast<uint8_t*>(cpu_array->data)[0]) {
        return VisitExpr(op->true_branch, ll);
      } else {
        return VisitExpr(op->false_branch, ll);
      }
    } else {
      Expr t = store_.Extend<Expr>([&]() {
        return LetList::With([&](LetList* ll) { return VisitExpr(op->true_branch, ll)->dynamic; });
      });
      Expr f = store_.Extend<Expr>([&]() {
        return LetList::With([&](LetList* ll) { return VisitExpr(op->false_branch, ll)->dynamic; });
      });
      store_.Invalidate();
      return NoStatic(ll->Push(If(c->dynamic, t, f)));
    }
  }

  PStatic VisitExpr_(const RefCreateNode* op, LetList* ll) final {
    PStatic ps = VisitExpr(op->value, ll);
    Static r = MkSRef();
    store_.Insert(r.as<SRefNode>(), ps);
    return HasStatic(r, ll->Push(RefCreate(ps->dynamic)));
  }

  PStatic VisitExpr_(const RefWriteNode* op, LetList* ll) final {
    PStatic r = VisitExpr(op->ref, ll);
    PStatic v = VisitExpr(op->value, ll);
    if (r->pstatic.defined()) {
      store_.Insert(r->pstatic.as<SRefNode>(), v);
    } else {
      store_.Invalidate();
    }
    return HasStatic(MkSTuple({}), ll->Push(RefWrite(r->dynamic, v->dynamic)));
  }

  PStatic VisitExpr_(const RefReadNode* op, LetList* ll) final {
    PStatic r = VisitExpr(op->ref, ll);
    if (r->pstatic.defined()) {
      PStatic ret = store_.Lookup(r->pstatic.as<SRefNode>());
      if (ret.defined()) {
        return ret;
      }
    }
    return NoStatic(ll->Push(RefRead(r->dynamic)));
  }

  PStatic VisitExpr_(const CallNode* op, LetList* ll) final {
    if (op->op == with_funcid_op) {
      ICHECK_EQ(op->args.size(), 1);
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
      return Downcast<SFunc>(f->pstatic)->func(f, x, op->attrs, op->type_args, ll);
    } else {
      store_.Invalidate();
      return NoStatic(ll->Push(Call(f->dynamic, x_dyn, op->attrs, op->type_args)));
    }
  }

  struct FuelFrame {
    PartialEvaluator* pe_;
    FuncId fid_;
    Fuel old_fuel;
    FuelFrame(PartialEvaluator* pe, FuncId fid, const Fuel& new_fuel) : pe_(pe), fid_(fid) {
      ICHECK_GT(pe_->fuel_map_.count(fid_), 0);
      old_fuel = pe_->fuel_map_[fid_];
      pe_->fuel_map_[fid_] = new_fuel;
    }
    ~FuelFrame() { pe_->fuel_map_[fid_] = old_fuel; }
  };

  size_t GetFTValue(const PStatic& ps) {
    if (ps->pstatic.defined()) {
      if (auto* st = ps->pstatic.as<STensorNode>()) {
        if (st->data.Shape().empty()) {
          NDArray cpu_array = st->data.CopyTo(CPUDevice());
          DataType dtype = DataType(cpu_array->dtype);
          if (dtype == DataType::Int(32)) {
            return std::max<int32_t>(0, *static_cast<const int32_t*>(cpu_array->data));
          } else if (dtype == DataType::Int(64)) {
            return std::max<int64_t>(0, *static_cast<const int64_t*>(cpu_array->data));
          }
        }
      }
    }
    return 0;
  }

  Fuel GetFuel(const PStatic& ps) {
    std::vector<Fuel> fuels;
    fuels.push_back(MkFTime(ps->created_time));
    fuels.push_back(MkFTValue(GetFTValue(ps)));
    return MkFSeq(fuels);
  }

  Func VisitFuncStatic(const Function& func, const Expr& var) {
    ICHECK(IsAtomic(var));
    if (func->HasNonzeroAttr(attr::kPrimitive)) {
      return ConstEvaluateFunc(func);
    }
    std::vector<std::pair<Var, PStatic>> free_vars;
    for (const auto& v : FreeVars(func)) {
      if (v != var) {
        free_vars.push_back(std::pair<Var, PStatic>(v, env_.Lookup(v)));
      }
    }
    return [=](const PStatic& self, const std::vector<PStatic>& pv, const Attrs& attrs,
               const tvm::Array<Type>& type_args, LetList* ll) {
      return env_.Extend<PStatic>([&]() {
        ICHECK_EQ(pv.size(), func->params.size());
        ICHECK_GT(func_map_.count(func), 0);
        FuncId fid = func_map_.at(func);
        if (fuel_map_.count(fid) == 0) {
          fuel_map_.insert({fid, MkFTop()});
        }
        std::vector<Fuel> args_fuel;
        for (const auto& v : pv) {
          args_fuel.push_back(GetFuel(v));
        }
        auto meet_res = fuel_map_[fid]->Meet(MkFSeq(args_fuel));
        if (std::get<1>(meet_res)) {
          FuelFrame tf(this, fid, std::get<0>(meet_res));
          Expr dedup_func = RegisterFuncId(DeDup(AnnotateFuncId(func)));
          Function func = AsFunc(dedup_func);
          if (var.as<VarNode>()) {
            env_.Insert(Downcast<Var>(var), self);
          }
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
            subst.Set(func->type_params[i], IncompleteType(kType));
          }
          return VisitExpr(RegisterFuncId(TypeSubst(AnnotateFuncId(func->body), subst)), ll);
        } else {
          std::vector<Expr> dyn;
          for (const auto& v : pv) {
            dyn.push_back(v->dynamic);
          }
          return NoStatic(ll->Push(Call(var, dyn, attrs, type_args)));
        }
      });
    };
  }

  Expr VisitFuncDynamic(const Function& func, const Func& f, const Expr& self) {
    return store_.Extend<Expr>([&]() {
      store_.Invalidate();
      return WithFields(
          func, func->params, LetList::With([&](LetList* ll) {
            std::vector<PStatic> pv;
            for (const auto& v : func->params) {
              pv.push_back(NoStatic(v));
            }
            tvm::Array<Type> type_args;
            for (const auto& tp : func->type_params) {
              type_args.push_back(tp);
            }
            return f(HasStatic(MkSFunc(f), self), pv, Attrs(), type_args, ll)->dynamic;
          }));
    });
  }

  PStatic VisitFunc(const Function& func, LetList* ll, const Var& name) {
    Func f = VisitFuncStatic(func, name);
    Function u_func = AsFunc(RegisterFuncId(DeDup(AnnotateFuncId(func))));
    // TODO(@M.K.): we seems to reduce landin knot into letrec.
    // restore letrec support across whole relay.
    return HasStatic(MkSFunc(f), ll->Push(name, VisitFuncDynamic(u_func, f, name)));
  }

  PStatic VisitExpr_(const FunctionNode* op, LetList* ll) final {
    return VisitFunc(GetRef<Function>(op), ll, Var::GenSym());
  }

  struct ReflectError : Error {
    ReflectError() : Error("static value not found") {}
  };

  Expr Reflect(const PStatic& st) {
    if (!st->pstatic.defined()) {
      throw ReflectError();
    } else if (const STensorNode* op = st->pstatic.as<STensorNode>()) {
      return Constant(op->data);
    } else if (const STupleNode* op = st->pstatic.as<STupleNode>()) {
      tvm::Array<Expr> fields;
      for (const PStatic& field : op->fields) {
        fields.push_back(Reflect(field));
      }
      return Tuple(fields);
    } else {
      LOG(FATAL) << "Unknown case: " << st->dynamic;
      throw;
    }
  }

  PStatic Reify(const ObjectRef& v, LetList* ll) const {
    if (v->IsInstance<runtime::NDArray::ContainerType>()) {
      auto nd_array = Downcast<runtime::NDArray>(v);
      return HasStatic(MkSTensor(nd_array), ll->Push(Constant(nd_array)));
    } else if (auto opt = v.as<runtime::ADT>()) {
      std::vector<PStatic> fields;
      tvm::Array<Expr> fields_dyn;
      auto adt = opt.value();
      for (size_t i = 0; i < adt.size(); ++i) {
        PStatic ps = Reify(adt[i], ll);
        fields.push_back(ps);
        fields_dyn.push_back(ps->dynamic);
      }
      return HasStatic(MkSTuple(fields), ll->Push(Tuple(fields_dyn)));
    } else {
      LOG(FATAL) << "Unknown case";
      throw;
    }
  }

  // Constant evaluate an expression.
  PStatic ConstEvaluate(const Expr& expr, LetList* ll) {
    // use a fresh build context in case we are already in a build context.
    With<transform::PassContext> fresh_build_ctx(transform::PassContext::Create());
    return Reify(Eval(expr, mod_->type_definitions, mod_->Imports(), CPUDevice(), CPUTarget()), ll);
  }

  Func ConstEvaluateFunc(const Expr& expr) {
    ICHECK_EQ(FreeVars(expr).size(), 0);
    return [=](const PStatic& self, const std::vector<PStatic>& pv, const Attrs& attrs,
               const tvm::Array<Type>& type_args, LetList* ll) {
      tvm::Array<Expr> ns_args;
      for (const PStatic& ps : pv) {
        ns_args.push_back(ps->dynamic);
      }
      auto ns = [&]() { return NoStatic(ll->Push(Call(expr, ns_args, attrs, type_args))); };
      if (StatefulOp(expr)) {
        return ns();
      }
      try {
        tvm::Array<Expr> args;
        for (const PStatic& ps : pv) {
          args.push_back(Reflect(ps));
        }
        return ConstEvaluate(Call(expr, args, attrs, type_args), ll);
      } catch (const ReflectError&) {
        return ns();
      }
    };
  }

  PStatic VisitExpr_(const OpNode* op, LetList* ll) final {
    return HasStatic(MkSFunc(ConstEvaluateFunc(GetRef<Expr>(op))), GetRef<Expr>(op));
  }

  PStatic VisitExpr_(const ConstructorNode* op, LetList* ll) final {
    Constructor c = GetRef<Constructor>(op);
    Func f = [=](const PStatic& self, const std::vector<PStatic>& pv, const Attrs& attrs,
                 const tvm::Array<Type>& type_args, LetList* ll) {
      tvm::Array<Expr> dyn;
      for (const PStatic& ps : pv) {
        dyn.push_back(ps->dynamic);
      }
      return HasStatic(MkSConstructor(c, pv), ll->Push(Call(c, dyn)));
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
            return [&]() {
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
                clauses.push_back(Clause(c->lhs, expr));
              }
              store_.Invalidate();
              return NoStatic(ll->Push(Match(ps->dynamic, clauses, op->complete)));
            }();
          default:
            LOG(FATAL) << "Unknown MatchStatus";
            throw;
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
      ICHECK_NE(op->constructor->tag, -1);
      ICHECK_NE(scn->constructor->tag, -1);
      if (op->constructor->tag == scn->constructor->tag) {
        ICHECK_EQ(op->patterns.size(), scn->fields.size());
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

  MatchStatus VisitPattern_(const PatternTupleNode* op, const PStatic& ps) final {
    if (ps->pstatic.defined()) {
      STuple stn = Downcast<STuple>(ps->pstatic);
      ICHECK_EQ(op->patterns.size(), stn->fields.size());
      MatchStatus current_match_status = MatchStatus::Match;
      for (size_t i = 0; i < op->patterns.size(); ++i) {
        MatchStatus ms = VisitPattern(op->patterns[i], stn->fields[i]);
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
    } else {
      return MatchStatus::Unknown;
    }
  }

  void InitializeFuncId(const Expr& e) {
    struct InitializeFuncIdVisitor : ExprVisitor, PatternVisitor {
      PartialEvaluator* pe;
      explicit InitializeFuncIdVisitor(PartialEvaluator* pe) : pe(pe) {}

      void VisitExpr_(const FunctionNode* op) final {
        Function f = GetRef<Function>(op);
        ICHECK_EQ(pe->func_map_.count(f), 0);
        pe->func_map_.insert({f, pe->func_map_.size()});
        VisitExpr(f->body);
      }

      void VisitPattern(const Pattern& p) final { PatternVisitor::VisitPattern(p); }
    };
    InitializeFuncIdVisitor(this).VisitExpr(e);
  }

  Expr RegisterFuncId(const Expr& e) {
    struct RegisterFuncIdVisitor : ExprVisitor, PatternVisitor {
      PartialEvaluator* pe;
      explicit RegisterFuncIdVisitor(PartialEvaluator* pe) : pe(pe) {}

      void VisitExpr_(const CallNode* op) final {
        if (op->op == with_funcid_op) {
          ICHECK_EQ(op->args.size(), 1);
          ICHECK(op->attrs.defined());
          ICHECK(op->attrs.as<WithFuncIdAttrs>());
          Function f = AsFunc(op->args[0]);
          FuncId fid = op->attrs.as<WithFuncIdAttrs>()->fid;
          if (pe->func_map_.count(f) != 0) {
            ICHECK_EQ(pe->func_map_.at(f), fid);
          }
          pe->func_map_.insert({f, fid});
        }
        ExprVisitor::VisitExpr_(op);
      }

      void VisitExpr_(const FunctionNode* op) final {
        Function f = GetRef<Function>(op);
        ICHECK_GT(pe->func_map_.count(f), 0);
        ExprVisitor::VisitExpr_(op);
      }

      void VisitPattern(const Pattern& p) final { PatternVisitor::VisitPattern(p); }
    };
    RegisterFuncIdVisitor(this).VisitExpr(e);
    return e;
  }

  Expr AnnotateFuncId(const Expr& e) {
    struct AnnotateFuncIdMutator : ExprMutator, PatternMutator {
      PartialEvaluator* pe;
      explicit AnnotateFuncIdMutator(PartialEvaluator* pe) : pe(pe) {}

      Expr VisitExpr_(const FunctionNode* op) final {
        Function f = GetRef<Function>(op);
        ICHECK_GT(pe->func_map_.count(f), 0);
        return MkWithFuncId(ExprMutator::VisitExpr_(op), pe->func_map_.at(f));
      }

      Pattern VisitPattern(const Pattern& p) final { return PatternMutator::VisitPattern(p); }

      Var VisitVar(const Var& v) final { return v; }
    };
    return AnnotateFuncIdMutator(this).VisitExpr(e);
  }

 private:
  Environment env_;
  IRModule mod_;
  std::unordered_map<GlobalVar, PStatic, ObjectPtrHash, ObjectPtrEqual> gv_map_;
  /*! Termination checking is done as follows:
   *  We have finitely many FunctionIds.
   *  Each FunctionId maps to a class of semantically equivalent function (ignoring type),
   *  as both TypeSubst and DeDup create semantically equivalent function.
   *  We partially map each FunctionId to a Fuel.
   *  Every time we try to inline a Function,
   *  we make sure it either does not have a Fuel,
   *  or we meet the existing fuel with the fuel calculated from the argument.
   *  If no progress is made, we do not inline.
   *  In both case, we remap the mapping to the new Fuel
   *  when we PE inside the Function body.
   *  Termination is guaranteed because Fuel is finitely descending - there can only be so many
   * meet.
   */
  std::unordered_map<Function, FuncId, ObjectPtrHash, ObjectPtrEqual> func_map_;
  std::unordered_map<FuncId, Fuel> fuel_map_;
  Store store_;
  Device device_ = CPUDevice();
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

    Var VisitVar(const Var& v) final { return Downcast<Var>(VisitExpr(v)); }

   private:
    std::unordered_map<Var, Var, VarHash, VarEqual> remap_;
  };
  return RemapMutator().VisitExpr(e);
}

Expr StripWithFuncId(const Expr& e) {
  struct StripWithFuncIdMutator : ExprMutator, PatternMutator {
    Expr VisitExpr_(const CallNode* op) final {
      if (op->op == with_funcid_op) {
        ICHECK_EQ(op->args.size(), 1);
        return VisitExpr(op->args[0]);
      } else {
        return ExprMutator::VisitExpr_(op);
      }
    }

    Pattern VisitPattern(const Pattern& p) final { return PatternMutator::VisitPattern(p); }

    Var VisitVar(const Var& v) final { return v; }
  };
  return StripWithFuncIdMutator().VisitExpr(e);
}

Expr PostProcess(const Expr& e) { return StripWithFuncId(DeDup(Remap(e))); }

}  // namespace partial_eval

IRModule PartialEval(const IRModule& m) {
  CheckFeature(m, FeatureSet::All() - fGraph);
  relay::partial_eval::PartialEvaluator pe(m);
  std::vector<GlobalVar> gvs;
  for (const auto& p : m->functions) {
    gvs.push_back(p.first);
  }
  for (const auto& gv : gvs) {
    pe.VisitGlobalVar(gv);
  }
  CheckFeature(m, FeatureSet::All() - fGraph);
  return m;
}

namespace transform {

Pass PartialEval() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func =
      [=](IRModule m, PassContext pc) { return relay::PartialEval(m); };
  return CreateModulePass(pass_func, 1, "PartialEval", {});
}

TVM_REGISTER_GLOBAL("relay._transform.PartialEvaluate").set_body_typed(PartialEval);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
