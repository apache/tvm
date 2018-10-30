/*!
 *  Copyright (c) 2018 by Contributors
 * \file src/tvm/relay/interpreter.cc
 * \brief An interpreter for the Relay IR.
 */

#include <tvm/codegen.h>
#include <tvm/packed_func_ext.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/logging.h>
#include <tvm/relay/pass.h>
#include "./ir/type_functor.h"

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
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      *ret = ClosureNode::make(args[0], args[1]);
    });

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
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      *ret = TupleValueNode::make(args[0]);
    });

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
    .set_dispatch<TupleValueNode>([](const TupleValueNode* node,
                                     tvm::IRPrinter* p) {
      p->stream << "TupleValueNode(" << node->fields << ")";
    });

TensorValue TensorValueNode::make(runtime::NDArray data) {
  NodePtr<TensorValueNode> n = make_node<TensorValueNode>();
  n->data = std::move(data);
  return TensorValue(n);
}

TVM_STATIC_IR_FUNCTOR_REGISTER(IRPrinter, vtable)
    .set_dispatch<TensorValueNode>([](const TensorValueNode* node,
                                      tvm::IRPrinter* p) {
      auto to_str = GetPackedFunc("relay._tensor_value_repr");
      std::string data_str = to_str(GetRef<TensorValue>(node));
      p->stream << "TensorValueNode(" << data_str << ")";
    });

TensorValue TensorValueNode::FromType(const Type& t) {
  if (auto tt_node = t.as<TensorTypeNode>()) {
    std::vector<int64_t> dims;

    for (auto dim : tt_node->shape) {
      auto int_node = dim.as<tvm::ir::IntImm>();
      CHECK(int_node) << "expected concrete dimensions";
      dims.push_back(int_node->value);
    }

    DLDataType dtype;
    DLContext context;

    switch (tt_node->dtype.code()) {
      case halideir_type_int:
        dtype.code = kDLInt;
        break;
      case halideir_type_uint:
        dtype.code = kDLUInt;
        break;
      case halideir_type_float:
        dtype.code = kDLFloat;
        break;
      default:
        throw dmlc::Error("can not convert HalideIR type into DLTensor dtype");
    }

    dtype.bits = tt_node->dtype.bits();
    dtype.lanes = tt_node->dtype.lanes();

    // TODO(@jroesch): Is this the right place to place the tensor?
    context.device_type = DLDeviceType::kDLCPU;
    context.device_id = 0;
    runtime::NDArray data = NDArray::Empty(dims, dtype, context);
    return TensorValueNode::make(data);
  } else {
    LOG(FATAL) << "expected a tensor type";
    return TensorValue();
  }
}

TVM_REGISTER_API("relay._make.TensorValue")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      runtime::NDArray data = args[0];
      *ret = TensorValueNode::make(data);
    });

/* Evaluator Implementation. */
struct EvalError : dmlc::Error {
  explicit EvalError(const std::string& msg) : Error(msg) {}
};

struct Frame {
  tvm::Map<Var, Value> locals;

  explicit Frame(tvm::Map<Var, Value> locals) : locals(locals) {}
};

struct Stack {
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

  struct LocalFrame {
    Stack& st;
    explicit LocalFrame(Stack& st, const Frame& fr) : st(st) {
      st.frames.push_back(fr);
    }
    ~LocalFrame() { st.frames.pop_back(); }
  };
};

struct Interpreter : ExprFunctor<Value(const Expr& n)> {
  Environment env;
  Stack stack;
  using JitKey = Function;

  using OpMap = std::unordered_map<JitKey, PackedFunc, ExprHash, ExprEqual>;

  OpMap operator_map_;

  template <typename T>
  T with_frame(const Frame& fr, const std::function<T()>& f) {
    Stack::LocalFrame lf(stack, fr);
    return f();
  }

  Interpreter(Environment env) : env(env), operator_map_() {}
  Interpreter(Environment env, OpMap operator_map) : env(env), operator_map_(operator_map) {}

  void extend(const Var& id, Value v) {
    this->stack.current_frame().locals.Set(id, v);
  }

  inline Value Lookup(const Var& local) {
    return this->stack.Lookup(local);
  }

  Value Eval(const Expr& expr) {
    return (*this)(expr);
  }

  Value VisitExpr(const Expr& expr) override {
    RELAY_LOG(INFO) << "VisitExpr: " << expr << std::endl;
    auto ret = ExprFunctor<Value(const Expr& n)>::VisitExpr(expr);
    return ret;
  }

  Value VisitExpr_(const VarNode* var_node) override {
    return Lookup(GetRef<Var>(var_node));
  }

  Value VisitExpr_(const GlobalVarNode* op) override {
    return Eval(this->env->Lookup(GetRef<GlobalVar>(op)));
  }

  Value VisitExpr_(const OpNode* id) override {
    // TODO(@jroesch): Eta-expand and return in this case.
    throw EvalError(
        "internal error, need to wrap intrinsic into call synthetic call node "
        "in "
        "this case, eta expand");
  }

  Value VisitExpr_(const ConstantNode* op) override {
    return TensorValueNode::make(op->data);
  }

  Value VisitExpr_(const TupleNode* op) override {
    std::vector<Value> values;

    for (const auto& field : op->fields) {
      Value field_value = Eval(field);
      values.push_back(field_value);
    }

    return TupleValueNode::make(values);
  }

  Value VisitExpr_(const FunctionNode* func_node) override {
    auto func = GetRef<Function>(func_node);
    tvm::Map<Var, Value> captured_env;
    Array<Var> free_vars = FreeVars(func);

    for (const auto& var : free_vars) {
      captured_env.Set(var, Eval(var));
    }

    return ClosureNode::make(captured_env, func);
  }

  inline Value InvokeCompiledOp(PackedFunc func, const Array<Value>& args,
                                Type ret_type) {
    // Marshal the arguments.
    auto arg_len = args.size() + 1;
    std::vector<TVMValue> values(arg_len);
    std::vector<int> codes(arg_len);
    TVMArgsSetter setter(values.data(), codes.data());
    TVMRetValue ret;

    // We need real type information to properly allocate the structure.
    for (size_t i = 0; i < args.size(); i++) {
      if (const TensorValueNode* tv = args[i].as<TensorValueNode>()) {
        setter(i, tv->data);
      }
    }

    // TVM's calling convention is that the final argument is the output
    // buffer. To preserve the illusion of being a functional language
    // we need to allocate space for the output buffer based on the
    // return type.
    CHECK(ret_type.as<TensorTypeNode>());

    auto out_tensor = TensorValueNode::FromType(ret_type);

    setter(arg_len - 1, out_tensor->data);
    func.CallPacked(TVMArgs(values.data(), codes.data(), arg_len), &ret);
    return out_tensor;
  }

  Value Invoke(const Closure& closure, const tvm::Array<Value>& args) {
    // Get a reference to the function inside the closure.
    auto func = closure->func;
    auto compiled = operator_map_.find(func);
    tvm::Array<Function> funcs;
    for (auto op : operator_map_) {
      funcs.push_back(op.first);
    }

    // This case we know we have precompiled the operator.
    if (compiled != operator_map_.end()) {
      auto func_ty = func->func_type_annotation();
      return InvokeCompiledOp(compiled->second, args, func_ty->ret_type);
    }

    // Allocate a frame with the parameters and free variables.
    tvm::Map<Var, Value> locals;

    CHECK(func->params.size() == args.size());

    for (size_t i = 0; i < func->params.size(); i++) {
      CHECK_EQ(locals.count(func->params[i]), 0);
      locals.Set(func->params[i], args[i]);
    }

    // Add the var to value mappings from the Closure's environment.
    for (auto it = closure->env.begin(); it != closure->env.end(); ++it) {
      CHECK_EQ(locals.count((*it).first), 0);
      locals.Set((*it).first, (*it).second);
    }

    return with_frame<Value>(Frame(locals), [&]() { return Eval(func->body); });
  }

  Value VisitExpr_(const CallNode* call) override {
    tvm::Array<Value> args;
    for (auto arg : call->args) {
      args.push_back(Eval(arg));
    }

    // We should not find operators after running fusion,
    // and operator lowering.
    //
    // We have some functions cotaining chunks of operators
    // which will be loaded into operator map.
    if (auto op_node = call->op.as<OpNode>()) {
      LOG(FATAL) << "found " << op_node->name
                 << "; operators should be removed by future passes; try "
                    "fusing and lowering";
    }

    // Now we just evaluate and expect to find a closure.
    Value fn_val = Eval(call->op);
    if (const ClosureNode* closure_node = fn_val.as<ClosureNode>()) {
      auto closure = GetRef<Closure>(closure_node);
      return this->Invoke(closure, args);
    } else {
      throw EvalError(
          "internal error: type error, expected function value in the call "
          "position");
    }
  }

  Value VisitExpr_(const LetNode* op) override {
    auto value = Eval(op->value);
    this->extend(op->var, value);
    return Eval(op->body);
  }

  Value VisitExpr_(const TupleGetItemNode* op) override {
    Value val = Eval(op->tuple);
    if (auto product_node = val.as<TupleValueNode>()) {
      CHECK(op->index < product_node->fields.size());
      return product_node->fields[op->index];
    } else {
      throw EvalError("not a product");
    }
  }

  Value VisitExpr_(const IfNode* op) override {
    Value v = Eval(op->cond);
    if (const TensorValueNode* bv = v.as<TensorValueNode>()) {
      // TODO(@jroesch, @MK): Refactor code into helper from DCE.
      if (reinterpret_cast<uint8_t*>(bv->data->data)[0]) {
        return Eval(op->true_branch);
      } else {
        return Eval(op->false_branch);
      }
    } else {
      throw EvalError("type error, type system should have caught this");
    }
  }
};

Interpreter::OpMap CompileOperators(const Environment& env, const Expr& e) {
  Interpreter::OpMap op_map;
  auto lowered_ops = LowerOps(env, e);
  RELAY_LOG(INFO) << "LoweredFuncs: " << lowered_ops << std::endl;
  if (lowered_ops.size()) {
    const PackedFunc* fbuild_ptr = Registry::Get("relay.op.compiler._build");
    CHECK(fbuild_ptr);
    auto fbuild = *fbuild_ptr;

    // Collect the set of lowered functions to build a module.
    Array<LoweredFunc> lowered_funcs;
    for (auto lop : lowered_ops) {
      lowered_funcs.push_back(lop->lowered_func);
    }

    Module module = fbuild(lowered_funcs);

    // Loop over the lowered operations to map them into the operator map.
    for (auto lop : lowered_ops) {
      Function func = lop->func;
      LoweredFunc lf = lop->lowered_func;

      RELAY_LOG(INFO) << "LoweredFunc: " << lf->name << std::endl;
      auto op_impl = module.GetFunction(lf->name);
      op_map.insert({func, op_impl});
    }
  }

  return op_map;
}

Value Evaluate(Environment env, Expr e) {
  auto op_map = CompileOperators(env, e);
  Interpreter interp(env, op_map);
  return interp.Eval(e);
}

TVM_REGISTER_API("relay._interpreter.evaluate")
    .set_body([](TVMArgs args, TVMRetValue* ret) {
      Environment env = args[0];
      Expr expr = args[1];
      *ret = Evaluate(env, expr);
    });

}  // namespace relay
}  // namespace tvm
