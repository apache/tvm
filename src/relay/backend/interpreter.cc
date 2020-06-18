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
 * \file src/relay/interpreter.cc
 * \brief An interpreter for the Relay IR.
 */
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/object.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/debug.h>
#include <tvm/relay/feature.h>
#include <tvm/driver/driver_api.h>

#include "compile_engine.h"

namespace tvm {
namespace relay {

using namespace runtime;

InterpreterClosure::InterpreterClosure(tvm::Map<Var, ObjectRef> env,
                                       Function func) {
  ObjectPtr<InterpreterClosureObj> n = make_object<InterpreterClosureObj>();
  n->env = std::move(env);
  n->func = std::move(func);
  data_ = std::move(n);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<InterpreterClosureObj >([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const InterpreterClosureObj*>(ref.get());
  p->stream << "InterpreterClosureNode(" << node->func << ", " << node->env << ")";
});

inline const PackedFunc& GetPackedFunc(const std::string& name) {
  const PackedFunc* pf = tvm::runtime::Registry::Get(name);
  CHECK(pf != nullptr) << "Cannot find function " << name << " in registry";
  return *pf;
}

// TODO(@jroesch): this doesn't support mutual letrec
/* Object Implementation */
RecClosure::RecClosure(InterpreterClosure clos, Var bind) {
  ObjectPtr<RecClosureObj> n = make_object<RecClosureObj>();
  n->clos = std::move(clos);
  n->bind = std::move(bind);
  data_ = std::move(n);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<RecClosureObj>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const RecClosureObj*>(ref.get());
    p->stream << "RecClosureObj(" << node->clos << ")";
  });

RefValue::RefValue(ObjectRef value) {
  ObjectPtr<RefValueObj> n = make_object<RefValueObj>();
  n->value = value;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relay._make.RefValue")
.set_body_typed([](ObjectRef value){
  return RefValue(value);
});

TVM_REGISTER_NODE_TYPE(RefValueObj);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<RefValueObj>([](const ObjectRef& ref, ReprPrinter* p) {
    auto* node = static_cast<const RefValueObj*>(ref.get());
    p->stream << "RefValueObj(" << node->value << ")";
  });

ConstructorValue::ConstructorValue(int32_t tag,
                                   tvm::Array<ObjectRef> fields,
                                   Constructor constructor) {
  ObjectPtr<ConstructorValueObj> n = make_object<ConstructorValueObj>();
  n->tag = tag;
  n->fields = fields;
  n->constructor = constructor;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relay._make.ConstructorValue")
.set_body_typed([](int32_t tag, tvm::Array<ObjectRef> fields,
                   Constructor constructor) {
  return ConstructorValue(tag, fields, constructor);
});

TVM_REGISTER_NODE_TYPE(ConstructorValueObj);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<ConstructorValueObj>([](const ObjectRef& ref, ReprPrinter* p) {
  auto* node = static_cast<const ConstructorValueObj*>(ref.get());
  p->stream << "ConstructorValueObj(" << node->tag << ","
            << node->fields << ")";
});

/*!
 * \brief A stack frame in the Relay interpreter.
 *
 * Contains a mapping from relay::Var to relay::ObjectRef.
 */
struct Frame {
  /*! \brief The set of local variables and arguments for the frame. */
  tvm::Map<Var, ObjectRef> locals;

  explicit Frame(tvm::Map<Var, ObjectRef> locals) : locals(locals) {}
};

/*!
 * \brief The call stack in the Relay interpreter.
 *
 * Contains a stack of frames; each corresponding to
 * a function call.
 */
struct Stack {
  /*! \brief The stack frames. */
  std::vector<Frame> frames;
  Stack() : frames() { frames.push_back(Frame({})); }

  Frame& current_frame() { return frames.back(); }

  ObjectRef Lookup(const Var& local) {
    for (auto frame = frames.rbegin(); frame != frames.rend(); frame++) {
      auto elem = frame->locals.find(local);
      if (elem != frame->locals.end()) {
        return (*elem).second;
      }
    }

    LOG(FATAL) << "could not find variable binding for " << local
               << "address= " << local.operator->();
    return ObjectRef();
  }
  /*!
   * A wrapper around Frame to add RAII semantics to pushing and popping
   * stack frames.
   */
  struct LocalFrame {
    Stack& st;
    explicit LocalFrame(Stack& st, const Frame& fr) : st(st) {
      st.frames.push_back(fr);
    }
    ~LocalFrame() { st.frames.pop_back(); }
  };
};

/*! \brief A representation of the interpreter state which can be passed back to Python. */
class InterpreterState;

/*! \brief A container capturing the state of the interpreter. */
class InterpreterStateObj : public Object {
 public:
  using Frame = tvm::Map<Var, ObjectRef>;
  using Stack = tvm::Array<Frame>;

  /*! \brief The current expression under evaluation. */
  Expr current_expr;

  /*! \brief The call stack of the interpreter. */
  Stack stack;

  void VisitAttrs(tvm::AttrVisitor* v) {
    v->Visit("current_expr", &current_expr);
    v->Visit("stack", &stack);
  }

  static InterpreterState make(Expr current_expr, Stack stack);

  static constexpr const char* _type_key = "relay.InterpreterState";
  TVM_DECLARE_FINAL_OBJECT_INFO(InterpreterStateObj, Object);
};

class InterpreterState : public ObjectRef {
 public:
  TVM_DEFINE_OBJECT_REF_METHODS(InterpreterState, ObjectRef, InterpreterStateObj);
};

InterpreterState InterpreterStateObj::make(Expr current_expr, Stack stack) {
  ObjectPtr<InterpreterStateObj> n = make_object<InterpreterStateObj>();
  n->current_expr = std::move(current_expr);
  n->stack = std::move(stack);
  return InterpreterState(n);
}

// NOTE: the current interpreter assumes A-normal form.
// which is better for execution.
//
// It will run duplicated computations when taking program that
// contains DAG in dataflow-form.
//
// Conversion to ANF is recommended before running the interpretation.
class Interpreter :
      public ExprFunctor<ObjectRef(const Expr& n)>,
             PatternFunctor<bool(const Pattern& p, const ObjectRef& v)> {
 public:
  Interpreter(IRModule mod, DLContext context, Target target)
      : mod_(mod),
        context_(context),
        target_(target),
        debug_op_(Op::Get("debug")),
        shape_of_op_(Op::Get("shape_of")) {
    engine_ = CompileEngine::Global();
  }

  template <typename T>
  T WithFrame(const Frame& fr, const std::function<T()>& f) {
    Stack::LocalFrame lf(stack_, fr);
    return f();
  }

  void extend(const Var& id, ObjectRef v) {
    stack_.current_frame().locals.Set(id, v);
  }

  ObjectRef Lookup(const Var& local) {
    return stack_.Lookup(local);
  }

  ObjectRef Eval(const Expr& expr) {
    return VisitExpr(expr);
  }

  ObjectRef VisitExpr_(const VarNode* var_node) final {
    return Lookup(GetRef<Var>(var_node));
  }

  ObjectRef VisitExpr_(const GlobalVarNode* op) final {
    return Eval(mod_->Lookup(GetRef<GlobalVar>(op)));
  }

  ObjectRef VisitExpr_(const OpNode* id) override {
    // TODO(@jroesch): Eta-expand and return in this case.
    LOG(FATAL) << "internal error, need to wrap intrinsic into call synthetic call node "
               << "in "
               << "this case, eta expand";
    return ObjectRef();
  }

  ObjectRef VisitExpr_(const ConstantNode* op) final {
    return op->data.CopyTo(context_);
  }

  ObjectRef VisitExpr_(const TupleNode* op) final {
    std::vector<ObjectRef> values;

    for (const auto& field : op->fields) {
      ObjectRef field_value = Eval(field);
      values.push_back(field_value);
    }

    return ADT::Tuple(values);
  }

  ObjectRef MakeClosure(const Function& func, Var letrec_name = Var()) {
    tvm::Map<Var, ObjectRef> captured_mod;
    Array<Var> free_vars = FreeVars(func);

    for (const auto& var : free_vars) {
      // Evaluate the free var (which could be a function call) if it hasn't
      // shown up in a letting binding that has invoked the function.
      if (letrec_name.defined() && letrec_name == var) {
        continue;
      }

      captured_mod.Set(var, Eval(var));
    }

    // We must use mutation here to build a self referential closure.
    InterpreterClosure closure(captured_mod, func);
    if (letrec_name.defined()) {
      return RecClosure(closure, letrec_name);
    }
    return std::move(closure);
  }

  ObjectRef VisitExpr_(const FunctionNode* func_node) final {
    auto func = GetRef<Function>(func_node);
    return MakeClosure(func);
  }

  Array<Shape> ComputeDynamicShape(const Function& func,
                                   const Array<ObjectRef>& args) {
    CCacheKey key(func, Target::Create("llvm"));
    auto cfunc = engine_->LowerShapeFunc(key);
    size_t arity = cfunc->inputs.size() + cfunc->outputs.size();

    std::vector<TVMValue> values(arity);
    std::vector<int> codes(arity);
    TVMArgsSetter setter(values.data(), codes.data());
    std::vector<NDArray> inputs(cfunc->inputs.size());
    std::vector<NDArray> outputs(cfunc->outputs.size());

    DLContext cpu_ctx;
    cpu_ctx.device_type = kDLCPU;
    cpu_ctx.device_id = 0;

    auto fset_input = [&](size_t i, ObjectRef val, bool need_shape) {
        auto nd_array = Downcast<NDArray>(val);
        if (need_shape) {
          int64_t ndim = nd_array.Shape().size();
          NDArray shape_arr;
          if (ndim == 0) {
            shape_arr = NDArray::Empty({}, DataType::Int(64), cpu_ctx);
          } else {
            shape_arr = NDArray::Empty({ndim}, DataType::Int(64), cpu_ctx);
            int64_t* data = reinterpret_cast<int64_t*>(shape_arr->data);
            for (auto j = 0; j < ndim; ++j) {
              data[j] = nd_array.Shape()[j];
            }
          }
          inputs[i] = shape_arr;
          setter(i, shape_arr);
        } else {
          auto arr = nd_array.CopyTo(cpu_ctx);
          inputs[i] = arr;
          setter(i, arr);
        }
    };

    size_t arg_counter = 0;
    for (size_t i = 0; i < args.size(); ++i) {
      auto arg = args[i];
      auto param = func->params[i];
      int state = cfunc->shape_func_param_states[i]->value;
      if (arg->IsInstance<runtime::NDArray::ContainerType>()) {
        if (state & kNeedInputData) {
          fset_input(arg_counter++, arg, false);
        }
        if (state & kNeedInputShape) {
          fset_input(arg_counter++, arg, true);
        }
      } else {
        const ADT adt = Downcast<ADT>(arg);
        if (state & kNeedInputData) {
          for (size_t i = 0; i < adt.size(); ++i) {
            fset_input(arg_counter++, adt[i], false);
          }
        }
        if (state & kNeedInputShape) {
          for (size_t i = 0; i < adt.size(); ++i) {
            fset_input(arg_counter++, adt[i], true);
          }
        }
      }
    }
    CHECK_EQ(arg_counter, cfunc->inputs.size())
      << "Shape function input sizes mismatch";

    auto fset_shape_output = [&](size_t i, Type val_type) {
        // TODO(@icemelon): allow recursive tuple
        const TensorTypeNode* rtype = val_type.as<TensorTypeNode>();
        CHECK(rtype != nullptr);
        int64_t ndim = rtype->shape.size();
        auto arr = NDArray::Empty({ndim}, DataType::Int(64), cpu_ctx);
        outputs[i] = arr;
        setter(arg_counter + i, arr);
    };

    auto ret_type = func->body->checked_type();
    size_t out_cnt = 0;
    if (auto rtype = ret_type.as<TupleTypeNode>()) {
      out_cnt = rtype->fields.size();
      for (size_t i = 0; i < out_cnt; ++i) {
        fset_shape_output(i, rtype->fields[i]);
      }
    } else {
      out_cnt = 1;
      auto tt = Downcast<TensorType>(ret_type);
      fset_shape_output(0, tt);
    }
    CHECK_EQ(cfunc->outputs.size(), out_cnt)
      << "Shape function output sizes mismatch";

    PackedFunc shape_func;
    Module m;
    TVMRetValue rv;
    if (const auto* f = runtime::Registry::Get("relay.backend.build")) {
      m = (*f)(cfunc->funcs, cfunc->target);
    } else {
      m = build(cfunc->funcs, cfunc->target, Target(nullptr), BuildConfig::Current());
    }
    shape_func = m.GetFunction(cfunc->func_name);
    shape_func.CallPacked(TVMArgs(values.data(), codes.data(), arity), &rv);

    // Get output shapes
    Array<Shape> out_shapes;
    for (auto out_tensor : outputs) {
      int64_t* shape_data = reinterpret_cast<int64_t*>(out_tensor->data);
      Shape out_shape;
      for (int i = 0; i < out_tensor->shape[0]; ++i) {
        out_shape.push_back(tvm::Integer(shape_data[i]));
      }
      out_shapes.push_back(out_shape);
    }
    return out_shapes;
  }

  ObjectRef InvokePrimitiveOp(const Function& func,
                          const Array<ObjectRef>& args) {
    const auto* call_node = func->body.as<CallNode>();

    if (call_node && call_node->op == debug_op_) {
      auto dattrs = call_node->attrs.as<DebugAttrs>();
      auto interp_state = this->get_state(call_node->args[0]);

      if (dattrs->debug_func.defined()) {
        dattrs->debug_func(interp_state);
      } else {
        RELAY_DEBUG_INTERP(interp_state);
      }

      return args[0];
    }

    // Marshal the arguments.
    // Handle adt input/output by flattening them.
    size_t arg_len = 0;
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i]->IsInstance<NDArray::ContainerType>()) {
        ++arg_len;
      } else {
        auto adt = Downcast<ADT>(args[i]);
        arg_len += adt.size();
      }
    }
    size_t num_inputs = arg_len;
    if (const auto* tuple_type = func->body->checked_type().as<TupleTypeNode>()) {
      arg_len += tuple_type->fields.size();
    } else {
      CHECK(func->body->checked_type().as<TensorTypeNode>())
        << func->body->checked_type();
      arg_len += 1;
    }
    std::vector<TVMValue> values(arg_len);
    std::vector<int> codes(arg_len);
    TVMArgsSetter setter(values.data(), codes.data());

    auto fset_input = [&](size_t i, ObjectRef val) {
      const auto nd_array = Downcast<NDArray>(val);
      setter(i, nd_array);
      DLContext arg_ctx = nd_array->ctx;
      CHECK(arg_ctx.device_type ==  context_.device_type &&
            arg_ctx.device_id == context_.device_id)
        << "Interpreter expect context to be "
        << context_ << ", but get " << arg_ctx;
    };

    int arg_counter = 0;
    for (ObjectRef arg : args) {
      if (arg->IsInstance<NDArray::ContainerType>()) {
        fset_input(arg_counter++,  arg);
      } else {
        auto adt = Downcast<ADT>(arg);
        for (size_t i = 0; i < adt.size(); ++i) {
          fset_input(arg_counter++, adt[i]);
        }
      }
    }

    // TVM's calling convention is that the final argument is the output
    // buffer. To preserve the illusion of being a functional language
    // we need to allocate space for the output buffer based on the
    // return type.
    auto fset_output = [&](size_t i, Type val_type) {
      const TensorTypeNode* rtype = val_type.as<TensorTypeNode>();
      CHECK(rtype != nullptr);
      // Allocate output tensor.
      std::vector<int64_t> shape;
      for (auto dim : rtype->shape) {
        const auto* ivalue = tir::as_const_int(dim);
        CHECK(ivalue) << "expected concrete dimensions";
        shape.push_back(ivalue[0]);
      }
      DLDataType dtype = rtype->dtype;
      NDArray nd_array = NDArray::Empty(shape, dtype, context_);
      setter(num_inputs + i, nd_array);
      return nd_array;
    };

    Array<Shape> out_shapes;
    auto ret_type = func->body->checked_type();
    bool is_dyn = IsDynamic(func->checked_type());
    if (call_node->op == shape_of_op_) {
      // The output shape of shape_of must be static since Relay doesn't support
      // dynamic rank tensors.
      is_dyn = false;
    }

    if (is_dyn) {
      CHECK(func->HasNonzeroAttr(attr::kPrimitive));
      out_shapes = ComputeDynamicShape(func, args);
    }

    PackedFunc packed_func = engine_->JIT(CCacheKey(func, target_));
    TVMRetValue rv;
    if (const TupleTypeNode* rtype = func->body->checked_type().as<TupleTypeNode>()) {
      CHECK(!is_dyn || out_shapes.size() == rtype->fields.size());
      std::vector<ObjectRef> fields;
      for (size_t i = 0; i < rtype->fields.size(); ++i) {
        if (is_dyn) {
          auto sh = out_shapes[i];
          auto tt = Downcast<TensorType>(rtype->fields[i]);
          fields.push_back(fset_output(i, TensorType(sh, tt->dtype)));
        } else {
          fields.push_back(fset_output(i, rtype->fields[i]));
        }
      }
      packed_func.CallPacked(TVMArgs(values.data(), codes.data(), arg_len), &rv);
      return ADT::Tuple(fields);
    } else {
      ObjectRef out_tensor;
      if (is_dyn) {
        CHECK_EQ(out_shapes.size(), 1);
        auto sh = out_shapes[0];
        auto tt = Downcast<TensorType>(ret_type);
        out_tensor = fset_output(0, TensorType(sh, tt->dtype));
      } else {
        out_tensor = fset_output(0, ret_type);
      }
      packed_func.CallPacked(TVMArgs(values.data(), codes.data(), arg_len), &rv);
      return out_tensor;
    }
  }

  // Invoke the closure
  ObjectRef Invoke(const InterpreterClosure& closure,
                   const tvm::Array<ObjectRef>& args,
                   const Var& bind = Var()) {
    // Get a reference to the function inside the closure.
    if (closure->func->HasNonzeroAttr(attr::kPrimitive)) {
      return InvokePrimitiveOp(closure->func, args);
    }
    auto func = closure->func;
    // Allocate a frame with the parameters and free variables.
    tvm::Map<Var, ObjectRef> locals;

    CHECK_EQ(func->params.size(), args.size());

    for (size_t i = 0; i < func->params.size(); i++) {
      CHECK_EQ(locals.count(func->params[i]), 0);
      locals.Set(func->params[i], args[i]);
    }

    // Add the var to value mappings from the Closure's environment.
    for (auto it = closure->env.begin(); it != closure->env.end(); ++it) {
      CHECK_EQ(locals.count((*it).first), 0);
      locals.Set((*it).first, (*it).second);
    }

    if (bind.defined()) {
      locals.Set(bind, RecClosure(closure, bind));
    }

    return WithFrame<ObjectRef>(Frame(locals), [&]() { return Eval(func->body); });
  }

  ObjectRef VisitExpr_(const CallNode* call) final {
    tvm::Array<ObjectRef> args;
    for (auto arg : call->args) {
      args.push_back(Eval(arg));
    }
    // We should not find operators after running fusion,
    // and operator lowering.
    //
    // We have some functions cotaining chunks of operators
    // which will be loaded into operator map.
    if (const auto* op_node = call->op.as<OpNode>()) {
      LOG(FATAL) << "found " << op_node->name
                 << "; operators should be removed by future passes; try "
                    "fusing and lowering";
    }
    if (auto con = call->op.as<ConstructorNode>()) {
      return ConstructorValue(con->tag, args, GetRef<Constructor>(con));
    }
    // Now we just evaluate and expect to find a closure.
    ObjectRef fn_val = Eval(call->op);
    if (const InterpreterClosureObj* closure_node = fn_val.as<InterpreterClosureObj>()) {
      auto closure = GetRef<InterpreterClosure>(closure_node);
      return this->Invoke(closure, args);
    } else if (const RecClosureObj* closure_node = fn_val.as<RecClosureObj>()) {
      return this->Invoke(closure_node->clos, args, closure_node->bind);
    } else {
      LOG(FATAL) << "internal error: type error, expected function value in the call "
                 << "position";
      return ObjectRef();
    }
  }

  ObjectRef VisitExpr_(const LetNode* let) final {
    if (auto func = let->value.as<FunctionNode>()) {
      auto clo = MakeClosure(GetRef<Function>(func), let->var);
      this->extend(let->var, clo);
    } else {
      auto value = Eval(let->value);
      this->extend(let->var, value);
    }

    return Eval(let->body);
  }

  ObjectRef VisitExpr_(const TupleGetItemNode* op) final {
    ObjectRef val = Eval(op->tuple);
    const auto* adt_obj = val.as<ADTObj>();
    CHECK(adt_obj)
      << "interal error: when evaluating TupleGetItem expected an ADT value";
    auto adt = GetRef<ADT>(adt_obj);
    CHECK_LT(static_cast<size_t>(op->index), adt.size())
        << "internal error: index out of bounds";
    return adt[op->index];
  }

  ObjectRef VisitExpr_(const IfNode* op) final {
    ObjectRef v = Eval(op->cond);
    if (v->IsInstance<NDArray::ContainerType>()) {
      auto nd_array = Downcast<NDArray>(v);
      DLContext cpu_ctx;
      cpu_ctx.device_type = kDLCPU;
      cpu_ctx.device_id = 0;
      NDArray cpu_array = nd_array.CopyTo(cpu_ctx);
      CHECK_EQ(DataType(cpu_array->dtype), DataType::Bool());
      // TODO(@jroesch, @MK): Refactor code into helper from DCE.
      if (reinterpret_cast<uint8_t*>(cpu_array->data)[0]) {
        return Eval(op->true_branch);
      } else {
        return Eval(op->false_branch);
      }
    } else {
      LOG(FATAL) << "type error, type system should have caught this";
      return ObjectRef();
    }
  }

  ObjectRef VisitExpr_(const RefWriteNode* op) final {
    ObjectRef r = Eval(op->ref);
    if (const RefValueObj* rv = r.as<RefValueObj>()) {
      rv->value = Eval(op->value);
      return ADT::Tuple(std::vector<ObjectRef>());
    } else {
      LOG(FATAL) << "type error, type system should have caught this";
      return ObjectRef();
    }
  }

  ObjectRef VisitExpr_(const RefCreateNode* op) final {
    return RefValue(Eval(op->value));
  }

  ObjectRef VisitExpr_(const RefReadNode* op) final {
    ObjectRef r = Eval(op->ref);
    if (const RefValueObj* rv = r.as<RefValueObj>()) {
      return rv->value;
    } else {
      LOG(FATAL) << "type error, type system should have caught this";
      return ObjectRef();
    }
  }

  ObjectRef VisitExpr_(const MatchNode* op) final {
    ObjectRef v = Eval(op->data);
    for (const Clause& c : op->clauses) {
      if (VisitPattern(c->lhs, v)) {
        return VisitExpr(c->rhs);
      }
    }
    LOG(FATAL) << "did not find any match";
    return ObjectRef();
  }

  bool VisitPattern_(const PatternConstructorNode* op, const ObjectRef& v) final {
    const ConstructorValueObj* cvn = v.as<ConstructorValueObj>();
    CHECK(cvn) << "need to be a constructor for match";
    CHECK_NE(op->constructor->tag, -1);
    CHECK_NE(cvn->tag, -1);
    if (op->constructor->tag == cvn->tag) {
      CHECK_EQ(op->patterns.size(), cvn->fields.size());
      for (size_t i = 0; i < op->patterns.size(); ++i) {
        if (!VisitPattern(op->patterns[i], cvn->fields[i])) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  bool VisitPattern_(const PatternTupleNode* op, const ObjectRef& v) final {
    auto adt = Downcast<ADT>(v);
    CHECK_EQ(op->patterns.size(), adt.size());
    for (size_t i = 0; i < op->patterns.size(); ++i) {
      if (!VisitPattern(op->patterns[i], adt[i])) {
        return false;
      }
    }
    return true;
  }

  bool VisitPattern_(const PatternWildcardNode* op, const ObjectRef& v) final {
    return true;
  }

  bool VisitPattern_(const PatternVarNode* op, const ObjectRef& v) final {
    extend(op->var, v);
    return true;
  }

  InterpreterState get_state(Expr e = Expr()) const {
    InterpreterStateObj::Stack stack;
    for (auto fr : this->stack_.frames) {
      InterpreterStateObj::Frame frame = fr.locals;
      stack.push_back(frame);
    }
    auto state = InterpreterStateObj::make(e, stack);
    return state;
  }

 private:
  // Module
  IRModule mod_;
  // For simplicity we only run the interpreter on a single context.
  // Context to run the interpreter on.
  DLContext context_;
  // Target parameter being used by the interpreter.
  Target target_;
  // Object stack.
  Stack stack_;
  // Backend compile engine.
  CompileEngine engine_;
  // Cache ops that need to be frequently used later to reduce lookup overhead.
  const Op& debug_op_;
  const Op& shape_of_op_;
};


TypedPackedFunc<ObjectRef(Expr)>
CreateInterpreter(
    IRModule mod,
    DLContext context,
    Target target) {
  if (mod.defined()) {
    // eta expand to support constructors in argument position
    transform::Sequential seq({
        transform::EtaExpand(
            /* expand_constructor */ true, /* expand_global_var */ false)});
    transform::PassContext pass_ctx = transform::PassContext::Current();
    tvm::With<transform::PassContext> ctx(pass_ctx);
    mod = seq(mod);
  }

  auto intrp = std::make_shared<Interpreter>(mod, context, target);
  auto packed = [intrp](Expr expr) {
    auto f = DetectFeature(expr);
    CHECK(f.is_subset_of(FeatureSet::All() - fGraph));
    return intrp->Eval(expr);
  };
  return TypedPackedFunc<ObjectRef(Expr)>(packed);
}

TVM_REGISTER_GLOBAL("relay.backend.CreateInterpreter")
.set_body_typed(CreateInterpreter);

}  // namespace relay
}  // namespace tvm
