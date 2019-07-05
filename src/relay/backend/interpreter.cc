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
 * \file src/tvm/relay/interpreter.cc
 * \brief An interpreter for the Relay IR.
 */
#include <tvm/packed_func_ext.h>
#include <tvm/runtime/device_api.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/debug.h>
#include "compile_engine.h"

namespace tvm {
namespace relay {

using namespace runtime;

inline const PackedFunc& GetPackedFunc(const std::string& name) {
  const PackedFunc* pf = tvm::runtime::Registry::Get(name);
  CHECK(pf != nullptr) << "Cannot find function " << name << " in registry";
  return *pf;
}

/* Value Implementation */
Closure ClosureNode::make(tvm::Map<Var, Value> env, Function func) {
  NodePtr<ClosureNode> n = make_node<ClosureNode>();
  n->env = std::move(env);
  n->func = std::move(func);
  return Closure(n);
}

TVM_REGISTER_API("relay._make.Closure")
.set_body_typed(ClosureNode::make);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<ClosureNode>([](const ClosureNode* node, tvm::IRPrinter* p) {
    p->stream << "ClosureNode(" << node->func << ")";
  });

TupleValue TupleValueNode::make(tvm::Array<Value> value) {
  NodePtr<TupleValueNode> n = make_node<TupleValueNode>();
  n->fields = value;
  return TupleValue(n);
}

TVM_REGISTER_API("relay._make.TupleValue")
.set_body_typed(TupleValueNode::make);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<TupleValueNode>([](const TupleValueNode* node, tvm::IRPrinter* p) {
    p->stream << "TupleValueNode(" << node->fields << ")";
  });

TensorValue TensorValueNode::make(runtime::NDArray data) {
  NodePtr<TensorValueNode> n = make_node<TensorValueNode>();
  n->data = std::move(data);
  return TensorValue(n);
}

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<TensorValueNode>([](const TensorValueNode* node, tvm::IRPrinter* p) {
    auto to_str = GetPackedFunc("relay._tensor_value_repr");
    std::string data_str = to_str(GetRef<TensorValue>(node));
    p->stream << "TensorValueNode(" << data_str << ")";
  });

TVM_REGISTER_API("relay._make.TensorValue")
.set_body_typed(TensorValueNode::make);

RefValue RefValueNode::make(Value value) {
  NodePtr<RefValueNode> n = make_node<RefValueNode>();
  n->value = value;
  return RefValue(n);
}

TVM_REGISTER_API("relay._make.RefValue")
.set_body_typed(RefValueNode::make);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<RefValueNode>([](const RefValueNode* node,
                               tvm::IRPrinter* p) {
                              p->stream << "RefValueNode(" << node->value << ")";
                            });

ConstructorValue ConstructorValueNode::make(int32_t tag,
                                            tvm::Array<Value> fields,
                                            Constructor constructor) {
  NodePtr<ConstructorValueNode> n = make_node<ConstructorValueNode>();
  n->tag = tag;
  n->fields = fields;
  n->constructor = constructor;
  return ConstructorValue(n);
}

TVM_REGISTER_API("relay._make.ConstructorValue")
.set_body_typed(ConstructorValueNode::make);

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
.set_dispatch<ConstructorValueNode>([](const ConstructorValueNode* node,
                                       tvm::IRPrinter* p) {
  p->stream << "ConstructorValueNode(" << node->tag << ","
            << node->fields << ")";
});

/*!
 * \brief A stack frame in the Relay interpreter.
 *
 * Contains a mapping from relay::Var to relay::Value.
 */
struct Frame {
  /*! \brief The set of local variables and arguments for the frame. */
  tvm::Map<Var, Value> locals;

  explicit Frame(tvm::Map<Var, Value> locals) : locals(locals) {}
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

  Value Lookup(const Var& local) {
    for (auto frame = frames.rbegin(); frame != frames.rend(); frame++) {
      auto elem = frame->locals.find(local);
      if (elem != frame->locals.end()) {
        return (*elem).second;
      }
    }

    LOG(FATAL) << "could not find variable binding for " << local
               << "address= " << local.operator->();
    return Value();
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
class InterpreterStateNode : public Node {
 public:
  using Frame = tvm::Map<Var, Value>;
  using Stack = tvm::Array<Frame>;

  /*! \brief The current expression under evaluation. */
  Expr current_expr;

  /*! \brief The call stack of the interpreter. */
  Stack stack;

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("current_expr", &current_expr);
    v->Visit("stack", &stack);
  }

  static InterpreterState make(Expr current_expr, Stack stack);

  static constexpr const char* _type_key = "relay.InterpreterState";
  TVM_DECLARE_NODE_TYPE_INFO(InterpreterStateNode, Node);
};

RELAY_DEFINE_NODE_REF(InterpreterState, InterpreterStateNode, NodeRef);

InterpreterState InterpreterStateNode::make(Expr current_expr, Stack stack) {
  NodePtr<InterpreterStateNode> n = make_node<InterpreterStateNode>();
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
      public ExprFunctor<Value(const Expr& n)>,
             PatternFunctor<bool(const Pattern& p, const Value& v)> {
 public:
  Interpreter(Module mod,
              DLContext context,
              Target target)
      : mod_(mod), context_(context), target_(target) {
    engine_ = CompileEngine::Global();
  }

  template <typename T>
  T WithFrame(const Frame& fr, const std::function<T()>& f) {
    Stack::LocalFrame lf(stack_, fr);
    return f();
  }

  void extend(const Var& id, Value v) {
    stack_.current_frame().locals.Set(id, v);
  }

  inline Value Lookup(const Var& local) {
    return stack_.Lookup(local);
  }

  Value Eval(const Expr& expr) {
    return VisitExpr(expr);
  }

  Value VisitExpr(const Expr& expr) final {
    auto ret = ExprFunctor<Value(const Expr& n)>::VisitExpr(expr);
    return ret;
  }

  Value VisitExpr_(const VarNode* var_node) final {
    return Lookup(GetRef<Var>(var_node));
  }

  Value VisitExpr_(const GlobalVarNode* op) final {
    return Eval(mod_->Lookup(GetRef<GlobalVar>(op)));
  }

  Value VisitExpr_(const OpNode* id) override {
    // TODO(@jroesch): Eta-expand and return in this case.
    LOG(FATAL) << "internal error, need to wrap intrinsic into call synthetic call node "
               << "in "
               << "this case, eta expand";
    return Value();
  }

  Value VisitExpr_(const ConstantNode* op) final {
    return TensorValueNode::make(op->data.CopyTo(context_));
  }

  Value VisitExpr_(const TupleNode* op) final {
    std::vector<Value> values;

    for (const auto& field : op->fields) {
      Value field_value = Eval(field);
      values.push_back(field_value);
    }

    return TupleValueNode::make(values);
  }

  // TODO(@jroesch): this doesn't support mututal letrec
  inline Value MakeClosure(const Function& func, Var letrec_name = Var()) {
    tvm::Map<Var, Value> captured_mod;
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
    auto closure = ClosureNode::make(captured_mod, func);
    auto mut_closure =
        static_cast<ClosureNode*>(const_cast<Node*>(closure.get()));
    if (letrec_name.defined()) {
      mut_closure->env.Set(letrec_name, closure);
    }
    return std::move(closure);
  }

  Value VisitExpr_(const FunctionNode* func_node) final {
    auto func = GetRef<Function>(func_node);
    return MakeClosure(func);
  }

  Value InvokePrimitiveOp(Function func,
                          const Array<Value>& args) {
    auto call_node = func->body.as<CallNode>();

    if (call_node && call_node->op == Op::Get("debug")) {
      auto dattrs = call_node->attrs.as<DebugAttrs>();
      auto interp_state = this->get_state(call_node->args[0]);

      if (dattrs->debug_func.defined()) {
        dattrs->debug_func(interp_state);
      } else {
        RELAY_DEBUG(interp_state);
      }

      return args[0];
    }

    // Marshal the arguments.
    // Handle tuple input/output by flattening them.
    size_t arg_len = 0;
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i].as<TensorValueNode>()) {
        ++arg_len;
      } else {
        const auto* tvalue = args[i].as<TupleValueNode>();
        arg_len += tvalue->fields.size();
      }
    }
    size_t num_inputs = arg_len;
    if (const auto* tuple_type = func->body->checked_type().as<TupleTypeNode>()) {
      arg_len += tuple_type->fields.size();
    } else {
      CHECK(func->body->checked_type().as<TensorTypeNode>());
      arg_len += 1;
    }
    std::vector<TVMValue> values(arg_len);
    std::vector<int> codes(arg_len);
    TVMArgsSetter setter(values.data(), codes.data());

    auto fset_input = [&](size_t i, Value val) {
      const TensorValueNode* tv = val.as<TensorValueNode>();
      CHECK(tv != nullptr) << "expect Tensor argument";
      setter(i, tv->data);
      DLContext arg_ctx = tv->data->ctx;
      CHECK(arg_ctx.device_type ==  context_.device_type &&
            arg_ctx.device_id == context_.device_id)
        << "Interpreter expect context to be "
        << context_ << ", but get " << arg_ctx;
    };

    int arg_counter = 0;
    for (Value arg : args) {
      if (arg.as<TensorValueNode>()) {
        fset_input(arg_counter++,  arg);
      } else {
        const TupleValueNode* tuple = arg.as<TupleValueNode>();
        CHECK(tuple != nullptr);
        for (size_t i = 0; i < tuple->fields.size(); ++i) {
          fset_input(arg_counter++, tuple->fields[i]);
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
        const auto* ivalue = as_const_int(dim);
        CHECK(ivalue) << "expected concrete dimensions";
        shape.push_back(ivalue[0]);
      }
      DLDataType dtype = Type2TVMType(rtype->dtype);
      auto out_tensor = TensorValueNode::make(
          NDArray::Empty(shape, dtype, context_));
      setter(num_inputs + i, out_tensor->data);
      return out_tensor;
    };

    PackedFunc packed_func = engine_->JIT(CCacheKeyNode::make(func, target_));
    TVMRetValue rv;
    if (const TupleTypeNode* rtype = func->body->checked_type().as<TupleTypeNode>()) {
      Array<Value> fields;
      for (size_t i = 0; i < rtype->fields.size(); ++i) {
        fields.push_back(fset_output(i, rtype->fields[i]));
      }
      packed_func.CallPacked(TVMArgs(values.data(), codes.data(), arg_len), &rv);
      return TupleValueNode::make(fields);
    } else {
      Value out_tensor = fset_output(0, func->body->checked_type());
      packed_func.CallPacked(TVMArgs(values.data(), codes.data(), arg_len), &rv);
      return out_tensor;
    }
  }

  // Invoke the closure
  Value Invoke(const Closure& closure, const tvm::Array<Value>& args) {
    // Get a reference to the function inside the closure.
    if (closure->func->IsPrimitive()) {
      return InvokePrimitiveOp(closure->func, args);
    }
    auto func = closure->func;
    // Allocate a frame with the parameters and free variables.
    tvm::Map<Var, Value> locals;

    CHECK_EQ(func->params.size(), args.size());

    for (size_t i = 0; i < func->params.size(); i++) {
      CHECK_EQ(locals.count(func->params[i]), 0);
      locals.Set(func->params[i], args[i]);
    }

    // Add the var to value mappings from the Closure's modironment.
    for (auto it = closure->env.begin(); it != closure->env.end(); ++it) {
      CHECK_EQ(locals.count((*it).first), 0);
      locals.Set((*it).first, (*it).second);
    }

    return WithFrame<Value>(Frame(locals), [&]() { return Eval(func->body); });
  }

  Value VisitExpr_(const CallNode* call) final {
    tvm::Array<Value> args;
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
      return ConstructorValueNode::make(con->tag, args, GetRef<Constructor>(con));
    }
    // Now we just evaluate and expect to find a closure.
    Value fn_val = Eval(call->op);
    if (const ClosureNode* closure_node = fn_val.as<ClosureNode>()) {
      auto closure = GetRef<Closure>(closure_node);
      return this->Invoke(closure, args);
    } else {
      LOG(FATAL) << "internal error: type error, expected function value in the call "
                 << "position";
      return Value();
    }
  }

  Value VisitExpr_(const LetNode* let) final {
    if (auto func = let->value.as<FunctionNode>()) {
      auto clo = MakeClosure(GetRef<Function>(func), let->var);
      this->extend(let->var, clo);
    } else {
      auto value = Eval(let->value);
      this->extend(let->var, value);
    }

    return Eval(let->body);
  }

  Value VisitExpr_(const TupleGetItemNode* op) final {
    Value val = Eval(op->tuple);
    auto product_node = val.as<TupleValueNode>();
    CHECK(product_node)
      << "interal error: when evaluating TupleGetItem expected a tuple value";
    CHECK_LT(static_cast<size_t>(op->index), product_node->fields.size())
        << "internal error: index out of bounds";
    return product_node->fields[op->index];
  }

  Value VisitExpr_(const IfNode* op) final {
    Value v = Eval(op->cond);
    if (const TensorValueNode* bv = v.as<TensorValueNode>()) {
      DLContext cpu_ctx;
      cpu_ctx.device_type = kDLCPU;
      cpu_ctx.device_id = 0;
      NDArray cpu_array = bv->data.CopyTo(cpu_ctx);
      CHECK_EQ(TVMType2Type(cpu_array->dtype), Bool());
      // TODO(@jroesch, @MK): Refactor code into helper from DCE.
      if (reinterpret_cast<uint8_t*>(cpu_array->data)[0]) {
        return Eval(op->true_branch);
      } else {
        return Eval(op->false_branch);
      }
    } else {
      LOG(FATAL) << "type error, type system should have caught this";
      return Value();
    }
  }

  Value VisitExpr_(const RefWriteNode* op) final {
    Value r = Eval(op->ref);
    if (const RefValueNode* rv = r.as<RefValueNode>()) {
      rv->value = Eval(op->value);
      return TupleValueNode::make({});
    } else {
      LOG(FATAL) << "type error, type system should have caught this";
      return Value();
    }
  }

  Value VisitExpr_(const RefCreateNode* op) final {
    return RefValueNode::make(Eval(op->value));
  }

  Value VisitExpr_(const RefReadNode* op) final {
    Value r = Eval(op->ref);
    if (const RefValueNode* rv = r.as<RefValueNode>()) {
      return rv->value;
    } else {
      LOG(FATAL) << "type error, type system should have caught this";
      return Value();
    }
  }

  Value VisitExpr_(const MatchNode* op) final {
    Value v = Eval(op->data);
    for (const Clause& c : op->clauses) {
      if (VisitPattern(c->lhs, v)) {
        return VisitExpr(c->rhs);
      }
    }
    LOG(FATAL) << "did not find any match";
    return Value();
  }

  bool VisitPattern_(const PatternConstructorNode* op, const Value& v) final {
    const ConstructorValueNode* cvn = v.as<ConstructorValueNode>();
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

  bool VisitPattern_(const PatternWildcardNode* op, const Value& v) final {
    return true;
  }

  bool VisitPattern_(const PatternVarNode* op, const Value& v) final {
    extend(op->var, v);
    return true;
  }

  InterpreterState get_state(Expr e = Expr()) const {
    InterpreterStateNode::Stack stack;
    for (auto fr : this->stack_.frames) {
      InterpreterStateNode::Frame frame = fr.locals;
      stack.push_back(frame);
    }
    auto state = InterpreterStateNode::make(e, stack);
    return state;
  }

 private:
  // Module
  Module mod_;
  // For simplicity we only run the interpreter on a single context.
  // Context to run the interpreter on.
  DLContext context_;
  // Target parameter being used by the interpreter.
  Target target_;
  // Value stack.
  Stack stack_;
  // Backend compile engine.
  CompileEngine engine_;
};


TypedPackedFunc<Value(Expr)>
CreateInterpreter(
    Module mod,
    DLContext context,
    Target target) {
  auto intrp = std::make_shared<Interpreter>(mod, context, target);
  auto packed = [intrp](Expr expr) {
    return intrp->Eval(expr);
  };
  return TypedPackedFunc<Value(Expr)>(packed);
}

TVM_REGISTER_API("relay.backend.CreateInterpreter")
.set_body_typed(CreateInterpreter);

TVM_REGISTER_NODE_TYPE(ClosureNode);
TVM_REGISTER_NODE_TYPE(TupleValueNode);
TVM_REGISTER_NODE_TYPE(TensorValueNode);

}  // namespace relay
}  // namespace tvm
