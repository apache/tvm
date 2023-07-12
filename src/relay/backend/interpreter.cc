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

#include <tvm/driver/driver_api.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/attrs/call.h>
#include <tvm/relay/attrs/debug.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/feature.h>
#include <tvm/relay/interpreter.h>
#include <tvm/relay/pattern_functor.h>
#include <tvm/relay/qnn/transform.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/container/map.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/object.h>
#include <tvm/target/compilation_config.h>

#include "../op/annotation/annotation.h"
#include "../op/call/call.h"
#include "../op/memory/device_copy.h"
#include "../transforms/pass_utils.h"
#include "te_compiler.h"

namespace tvm {
namespace relay {

using runtime::ADT;
using runtime::ADTObj;
using runtime::NDArray;
using runtime::TVMArgsSetter;
using runtime::operator<<;

namespace {
// TODO(mbs): Centralize.
struct PairHash {
  template <typename T1, typename T2>
  std::size_t operator()(const std::pair<T1, T2>& k) const {
    return dmlc::HashCombine(std::hash<T1>()(k.first), std::hash<T2>()(k.second));
  }
  template <typename T2>
  std::size_t operator()(const std::pair<Target, T2>& k) const {
    return dmlc::HashCombine(ObjectHash()(k.first), std::hash<T2>()(k.second));
  }
};

// Analogue of FlattenTupleType for runtime ADT vs NDArray values.
// TODO(mbs): Hoist somewhere sensible, maybe op/memory.h?
void FlattenADTAux(const ObjectRef& object_ref, std::vector<NDArray>* out) {
  if (auto ndarray = object_ref.as<NDArray>()) {
    out->push_back(ndarray.value());
  } else if (const ADTObj* adt = object_ref.as<ADTObj>()) {
    for (size_t i = 0; i < adt->size; ++i) {
      FlattenADTAux((*adt)[i], out);
    }
  } else {
    LOG(FATAL) << "unsupported " << object_ref;
  }
}

std::vector<NDArray> FlattenADT(const ObjectRef& object_ref) {
  std::vector<NDArray> out;
  FlattenADTAux(object_ref, &out);
  return out;
}

std::vector<NDArray> FlattenADTs(const std::vector<ObjectRef>& object_refs) {
  std::vector<NDArray> out;
  for (const auto& object_ref : object_refs) {
    FlattenADTAux(object_ref, &out);
  }
  return out;
}

// Analogue of ToTupleType for runtime ADT vs NDArray values.
// TODO(mbs): Hoist somewhere sensible, maybe op/memory.h?
void ToADTOrNDArrayAux(const Type& type, const std::vector<NDArray>& nd_arrays, int* index,
                       std::vector<ObjectRef>* out) {
  if (type.as<TensorTypeNode>()) {
    out->push_back(nd_arrays[*index]);
    *index += 1;
  } else if (const TupleTypeNode* ttn = type.as<TupleTypeNode>()) {
    std::vector<ObjectRef> tuple_out;
    for (size_t i = 0; i < ttn->fields.size(); i++) {
      ToADTOrNDArrayAux(ttn->fields[i], nd_arrays, index, &tuple_out);
    }
    out->push_back(ADT::Tuple(tuple_out));
  } else {
    LOG(FATAL) << "unsupported " << type;
  }
}

ObjectRef ToADTOrNDArray(const Type& type, const std::vector<NDArray>& nd_arrays) {
  if (type.as<TensorTypeNode>() && nd_arrays.size() == 1) {
    return nd_arrays[0];
  } else {
    std::vector<ObjectRef> out;
    int index = 0;
    ToADTOrNDArrayAux(type, nd_arrays, &index, &out);
    return out[0];
  }
}

}  // namespace

InterpreterClosure::InterpreterClosure(Map<Var, ObjectRef> env, Function func) {
  ObjectPtr<InterpreterClosureObj> n = make_object<InterpreterClosureObj>();
  n->env = std::move(env);
  n->func = std::move(func);
  data_ = std::move(n);
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<InterpreterClosureObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const InterpreterClosureObj*>(ref.get());
      p->stream << "InterpreterClosureNode(" << node->func << ", " << node->env << ")";
    });

inline const PackedFunc& GetPackedFunc(const std::string& name) {
  const PackedFunc* pf = runtime::Registry::Get(name);
  ICHECK(pf != nullptr) << "Cannot find function " << name << " in registry";
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

TVM_REGISTER_GLOBAL("relay._make.RefValue").set_body_typed([](ObjectRef value) {
  return RefValue(value);
});

TVM_REGISTER_NODE_TYPE(RefValueObj);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<RefValueObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const RefValueObj*>(ref.get());
      p->stream << "RefValueObj(" << node->value << ")";
    });

ConstructorValue::ConstructorValue(int32_t tag, Array<ObjectRef> fields, Constructor constructor) {
  ObjectPtr<ConstructorValueObj> n = make_object<ConstructorValueObj>();
  n->tag = tag;
  n->fields = fields;
  n->constructor = constructor;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("relay._make.ConstructorValue")
    .set_body_typed([](int32_t tag, Array<ObjectRef> fields, Constructor constructor) {
      return ConstructorValue(tag, fields, constructor);
    });

TVM_REGISTER_NODE_TYPE(ConstructorValueObj);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<ConstructorValueObj>([](const ObjectRef& ref, ReprPrinter* p) {
      auto* node = static_cast<const ConstructorValueObj*>(ref.get());
      p->stream << "ConstructorValueObj(" << node->tag << "," << node->fields << ")";
    });

/*!
 * \brief A stack frame in the Relay interpreter.
 *
 * Contains a mapping from relay::Var to relay::ObjectRef.
 */
struct Frame {
  /*! \brief The set of local variables and arguments for the frame. */
  Map<Var, ObjectRef> locals;

  explicit Frame(Map<Var, ObjectRef> locals) : locals(locals) {}
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
    explicit LocalFrame(Stack& st, const Frame& fr) : st(st) { st.frames.push_back(fr); }
    ~LocalFrame() { st.frames.pop_back(); }
  };
};

/*! \brief A representation of the interpreter state which can be passed back to Python. */
class InterpreterState;

/*! \brief A container capturing the state of the interpreter. */
class InterpreterStateObj : public Object {
 public:
  using Frame = Map<Var, ObjectRef>;
  using Stack = Array<Frame>;

  /*! \brief The current expression under evaluation. */
  Expr current_expr;

  /*! \brief The call stack of the interpreter. */
  Stack stack;

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("current_expr", &current_expr);
    v->Visit("stack", &stack);
  }

  static constexpr const char* _type_key = "relay.InterpreterState";
  TVM_DECLARE_FINAL_OBJECT_INFO(InterpreterStateObj, Object);
};

class InterpreterState : public ObjectRef {
 public:
  using Frame = Map<Var, ObjectRef>;
  using Stack = Array<Frame>;

  InterpreterState(Expr current_expr, Stack stack);

  TVM_DEFINE_OBJECT_REF_METHODS(InterpreterState, ObjectRef, InterpreterStateObj);
};

InterpreterState::InterpreterState(Expr current_expr, InterpreterState::Stack stack) {
  ObjectPtr<InterpreterStateObj> n = make_object<InterpreterStateObj>();
  n->current_expr = std::move(current_expr);
  n->stack = std::move(stack);
  data_ = std::move(n);
}

// NOTE: the current interpreter assumes A-normal form.
// which is better for execution.
//
// It will run duplicated computations when taking program that
// contains DAG in dataflow-form.
//
// Conversion to ANF is recommended before running the interpretation.
class Interpreter : public ExprFunctor<ObjectRef(const Expr& n)>,
                    PatternFunctor<bool(const Pattern& p, const ObjectRef& v)> {
 public:
  Interpreter(IRModule unified_mod, CompilationConfig config, Device device)
      : unified_mod_(unified_mod),
        config_(std::move(config)),
        device_(device),
        debug_op_(Op::Get("debug")) {}

  template <typename T>
  T WithFrame(const Frame& fr, const std::function<T()>& f) {
    Stack::LocalFrame lf(stack_, fr);
    return f();
  }

  void extend(const Var& id, ObjectRef v) { stack_.current_frame().locals.Set(id, v); }

  ObjectRef Lookup(const Var& local) { return stack_.Lookup(local); }

  ObjectRef Eval(const Expr& expr) { return VisitExpr(expr); }

  ObjectRef VisitExpr_(const VarNode* var_node) final { return Lookup(GetRef<Var>(var_node)); }

  ObjectRef VisitExpr_(const GlobalVarNode* op) final {
    return Eval(unified_mod_->Lookup(GetRef<GlobalVar>(op)));
  }

  ObjectRef VisitExpr_(const OpNode* id) override {
    // TODO(@jroesch): Eta-expand and return in this case.
    LOG(FATAL) << "internal error, need to wrap intrinsic into call synthetic call node "
               << "in this case, eta expand";
    return ObjectRef();
  }

  ObjectRef VisitExpr_(const ConstantNode* op) final { return op->data.CopyTo(device_); }

  ObjectRef VisitExpr_(const TupleNode* op) final {
    std::vector<ObjectRef> values;

    for (const auto& field : op->fields) {
      ObjectRef field_value = Eval(field);
      values.push_back(field_value);
    }

    return ADT::Tuple(values);
  }

  ObjectRef MakeClosure(const Function& func, Var letrec_name = Var()) {
    Map<Var, ObjectRef> captured_mod;
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

  /*!
   * \brief Returns the packed function implementing the TIR function bound to \p tir_fn_var.
   *
   * \param tir_fn_var Global var for the already lowered TIR function.
   * \param all_tir_fn_vars Global vars for all lowered TIR functions the above
   * may reference, plus \p tir_fn_var itself.
   * \param target Target for which the TIR function should be compiled. For primitives this
   * will be the interpreter's target_. However for shape functions this will be the generic
   * 'cpu' target, since shape functions are always executed on the host cpu.
   */
  PackedFunc TIRToPackedFunc(const GlobalVar& tir_fn_var, const Array<GlobalVar>& all_tir_fn_vars,
                             Target target) {
    std::pair<Target, std::string> packed_func_key(target, tir_fn_var->name_hint);
    auto packed_itr = compiled_packed_funcs_.find(packed_func_key);
    if (packed_itr != compiled_packed_funcs_.end()) {
      // Already compiled.
      return packed_itr->second;
    }

    // Project out just the function(s) we need.
    IRModule lowered_projected_mod;
    Map<Target, IRModule> per_target_module = tec::GetPerTargetModules(unified_mod_);
    std::unordered_map<Target, IRModule, backend::TargetStrHash, backend::TargetStrEqual>
        per_target_module_std_map = backend::TargetModuleMapToTargetStrModuleMap(per_target_module);
    auto mod_itr = per_target_module_std_map.find(target);
    ICHECK(mod_itr != per_target_module_std_map.end())
        << "No target module for target " << target->ToDebugString();
    const IRModule& target_module = (*mod_itr).second;
    for (const auto& var : all_tir_fn_vars) {
      ICHECK(target_module->ContainGlobalVar(var->name_hint))
          << "No global var for '" << var->name_hint << "' in module for target "
          << target->ToDebugString();
      lowered_projected_mod->Add(var, target_module->Lookup(var->name_hint));
    }

    // Compile (aka 'build') the projected module into a runtime module of packed functions.
    runtime::Module runtime_module;
    if (const auto* f = runtime::Registry::Get("relay.backend.build")) {
      // TODO(mbs): Cleanup hooks.
      runtime_module = (*f)(lowered_projected_mod, target);
    } else {
      runtime_module = build(lowered_projected_mod, target, /*target_host=*/Target(nullptr));
    }

    // Extract all the packed functions.
    for (const auto& var : all_tir_fn_vars) {
      PackedFunc packed_func = runtime_module.GetFunction(var->name_hint);
      ICHECK(packed_func != nullptr)
          << "No packed function for global var '" << var->name_hint
          << "' in compiled module for target " << target->ToDebugString();
      compiled_packed_funcs_.emplace(std::make_pair(target, var->name_hint), packed_func);
    }

    // Return just what we need for this call.
    packed_itr = compiled_packed_funcs_.find(packed_func_key);
    ICHECK(packed_itr != compiled_packed_funcs_.end()) << " " << tir_fn_var->name_hint;
    ICHECK_NOTNULL(packed_itr->second);
    return packed_itr->second;
  }

  /*!
   * \brief Call the dynamic shape function bound to \p prim_shape_fn_var passing the
   * shapes of args, and return the resulting shapes.
   *
   * \param prim_shape_fn_var Global var bound to lowered shape function.
   * \param all_prim_shape_fn_vars All the global vars needed to build the above, including
   * the shape function itself.
   * \param prim_shape_fn_states For each primitive arg, indicate whether the primitive shape
   * function requires the shape of the argument and/or the actual argument tensor.
   * \param num_shape_inputs The number of inputs, after accounting for both shapes vs data
   * inputs and unfolding of tuple types.
   * \param num_shape_outputs The number of outputs, after accounting for flattening of
   * tuple types.
   * \param args Arguments to the primitive this shape function is for.
   * \return Expected shapes of the underlying primitive's flattened outputs.
   */
  Array<Shape> ComputeDynamicShape(const GlobalVar& prim_shape_fn_var,
                                   const Array<GlobalVar>& all_prim_shape_fn_vars,
                                   const Array<Integer>& prim_shape_fn_states,
                                   size_t num_shape_inputs, size_t num_shape_outputs,
                                   Target prim_shape_target, const std::vector<ObjectRef>& args) {
    VLOG_CONTEXT << "ComputeDynamicShape";
    ICHECK(prim_shape_fn_var.defined());
    ICHECK(prim_shape_fn_var->checked_type().defined());
    VLOG(1) << "prim_shape_fn_var:" << std::endl << PrettyPrint(prim_shape_fn_var);
    ICHECK(prim_shape_fn_states.defined());
    for (size_t i = 0; i < prim_shape_fn_states.size(); ++i) {
      VLOG(1) << "prim_shape_fn_states[" << i << "]: " << prim_shape_fn_states[i];
    }
    VLOG(1) << "num_shape_inputs: " << num_shape_inputs;
    VLOG(1) << "num_shape_outputs: " << num_shape_outputs;
    VLOG(1) << "args.size(): " << args.size();
    VLOG(1) << "prim_shape_target: " << prim_shape_target->ToDebugString();

    // The function type is that of the shape function rather than the original primitive the shape
    // function is for.
    const auto* func_type_node = prim_shape_fn_var->checked_type().as<FuncTypeNode>();
    ICHECK(func_type_node);
    // The shape function states are w.r.t. the original primitive's arguments in
    // non-flattened form.
    // TODO(mbs): Clean this up so we don't mix flattened vs original conventions.
    ICHECK_EQ(args.size(), prim_shape_fn_states.size());

    // num_shape_inputs will account for which primitive function arguments are dynamic,
    // whether the shape and or data needs to be passed, and flattening of tuples.
    // Similarly, num_shape_outputs will account for flattening of tuples.

    // TODO(mbs): Take this from the host_virtual_device.
    Device shape_device;
    shape_device.device_type = static_cast<DLDeviceType>(prim_shape_target->GetTargetDeviceType());
    shape_device.device_id = 0;

    // 'Compile' the TIR shape function to appropriate callable form.
    PackedFunc packed_shape_func =
        TIRToPackedFunc(prim_shape_fn_var, all_prim_shape_fn_vars, prim_shape_target);

    size_t arity = num_shape_inputs + num_shape_outputs;
    std::vector<TVMValue> values(arity);
    std::vector<int> codes(arity);
    TVMArgsSetter setter(values.data(), codes.data());
    std::vector<NDArray> inputs(num_shape_inputs);
    std::vector<NDArray> outputs(num_shape_outputs);

    // Collect the shapes and/or data needed by the shape function from
    // the primitive's arguments.
    size_t arg_counter = 0;
    for (size_t i = 0; i < args.size(); ++i) {
      // TODO(mbs): The same need data/need shape arg state applies to everything in the
      // flattened form of this arg. Does that match what lowering actually does?
      int64_t state = prim_shape_fn_states[i]->value;
      for (const auto& nd_array : FlattenADT(args[i])) {
        if (state & tec::kNeedInputData) {
          auto arr = nd_array.CopyTo(shape_device);
          inputs[arg_counter] = arr;
          setter(arg_counter, arr);
          ++arg_counter;
        }
        if (state & tec::kNeedInputShape) {
          int64_t ndim = nd_array.Shape().size();
          NDArray shape_arr;
          if (ndim == 0) {
            shape_arr = NDArray::Empty({}, DataType::Int(64), shape_device);
          } else {
            shape_arr = NDArray::Empty({ndim}, DataType::Int(64), shape_device);
            int64_t* data = reinterpret_cast<int64_t*>(shape_arr->data);
            for (auto j = 0; j < ndim; ++j) {
              data[j] = nd_array.Shape()[j];
            }
          }
          inputs[arg_counter] = shape_arr;
          setter(arg_counter, shape_arr);
          ++arg_counter;
        }
      }
    }
    ICHECK_EQ(arg_counter, num_shape_inputs) << "Shape function input sizes mismatch";

    // Prepare NDArrays to hold the output shapes.
    size_t out_cnt = 0;
    for (const auto& ttype : FlattenTupleType(func_type_node->ret_type)) {
      ICHECK(out_cnt < num_shape_outputs);
      std::vector<int64_t> concrete_shape;
      for (const auto& dim : ttype->shape) {
        const auto* ivalue = tir::as_const_int(dim);
        ICHECK(ivalue) << "expected concrete dimensions";
        concrete_shape.push_back(ivalue[0]);
      }
      auto arr = NDArray::Empty(concrete_shape, ttype->dtype, shape_device);
      outputs[out_cnt] = arr;
      setter(arg_counter + out_cnt, arr);
      ++out_cnt;
    }
    ICHECK_EQ(out_cnt, num_shape_outputs) << "Shape function output sizes mismatch";

    // Call the dynamic shape function.
    TVMRetValue rv;  // ignored
    packed_shape_func.CallPacked(TVMArgs(values.data(), codes.data(), arity), &rv);

    // Convert result tensors back to shapes.
    Array<Shape> out_shapes;
    for (auto out_tensor : outputs) {
      int64_t* shape_data = reinterpret_cast<int64_t*>(out_tensor->data);
      Shape out_shape;
      for (int i = 0; i < out_tensor->shape[0]; ++i) {
        out_shape.push_back(Integer(shape_data[i]));
      }
      out_shapes.push_back(out_shape);
    }
    return out_shapes;
  }

  /*!
   * \brief Call primitive op bound to \p prim_fn_var with \p args. If necessary, evaluate dynamic
   * shape function bound to \p prim_shape_fn_var to calculate shapes of result tensors.
   *
   * @param prim_fn_var Global bound to lowered primitive.
   * @param all_prim_fn_vars  All globals references by lowered primitive, plus prim_fn_var itself.
   * @param prim_shape_fn_var Global bound to lowered shape function for primitive, if needed.
   * @param all_prim_shape_fn_vars All globals references by lowered shape function, plus
   * prim_shape_fn_var itself.
   * @param prim_shape_fn_states Records whether shape and/or data is needed by the dynamic
   * shape function (if any) for each (flattened) argument.
   * @param num_shape_inputs Number of arguments to the dynamic shape function (if any).
   * @param num_shape_outputs Number of outputs from the dynamic shape function (if any).
   * @param args Already evaluated arguments to primitive.
   * @return Result of primitive.
   */
  ObjectRef InvokePrimitiveOp(const GlobalVar& prim_fn_var, const Array<GlobalVar> all_prim_fn_vars,
                              Target prim_target, const GlobalVar& prim_shape_fn_var,
                              const Array<GlobalVar>& all_prim_shape_fn_vars,
                              const Array<Integer>& prim_shape_fn_states, size_t num_shape_inputs,
                              size_t num_shape_outputs, Target prim_shape_target,
                              const std::vector<ObjectRef>& args) {
    ICHECK(prim_fn_var->checked_type().defined());
    const FuncTypeNode* ftn = prim_fn_var->checked_type().as<FuncTypeNode>();
    ICHECK(ftn);

    // 'Compile' the TIR primitive to appropriate callable form (on the desired target).
    PackedFunc packed_func = TIRToPackedFunc(prim_fn_var, all_prim_fn_vars, prim_target);

    // Argument tuples are flattened.
    std::vector<NDArray> arg_nd_arrays = FlattenADTs(args);
    const size_t num_inputs = arg_nd_arrays.size();
    // num_inputs should equal size(concat(map(FlattenTupleType, function arg types)))

    // TVM's primitive calling convention is for the final arguments to be for output
    // buffers. We must allocate space for those buffers based on the return type.
    std::vector<TensorType> result_tensor_types = FlattenTupleType(ftn->ret_type);
    const size_t arg_len = num_inputs + result_tensor_types.size();

    std::vector<TVMValue> values(arg_len);
    std::vector<int> codes(arg_len);
    TVMArgsSetter setter(values.data(), codes.data());

    // Marshall the call's arguments in flattened form.
    int arg_counter = 0;
    for (const auto& nd_array : arg_nd_arrays) {
      setter(arg_counter++, nd_array);
      Device arg_dev = nd_array->device;
      ICHECK(arg_dev.device_type == device_.device_type && arg_dev.device_id == device_.device_id)
          << "Interpreter expect device to be " << device_ << ", but got " << arg_dev;
    }

    // If necessary, retrieve concrete shapes for outputs from shape function rather
    // than relying on TensorType shapes.
    Array<Shape> runtime_shapes;
    bool is_dyn = IsDynamic(ftn->ret_type);
    if (is_dyn) {
      ICHECK(prim_shape_fn_var.defined());
      ICHECK(prim_shape_fn_states.defined());
      runtime_shapes =
          ComputeDynamicShape(prim_shape_fn_var, all_prim_shape_fn_vars, prim_shape_fn_states,
                              num_shape_inputs, num_shape_outputs, prim_shape_target, args);
      ICHECK_EQ(runtime_shapes.size(), result_tensor_types.size());
    }

    // Prepare the result tensors for the call.
    TVMRetValue rv;  // ignored
    std::vector<NDArray> result_nd_arrays;
    for (size_t i = 0; i < result_tensor_types.size(); ++i) {
      const auto& ttype = result_tensor_types[i];
      const Shape& shape = is_dyn ? runtime_shapes[i] : ttype->shape;
      // Allocate output tensor of appropriate shape.
      std::vector<int64_t> concrete_shape;
      for (const auto& dim : shape) {
        const auto* ivalue = tir::as_const_int(dim);
        ICHECK(ivalue) << "expected concrete dimensions";
        concrete_shape.push_back(ivalue[0]);
      }
      NDArray nd_array = NDArray::Empty(concrete_shape, ttype->dtype, device_);
      setter(num_inputs + i, nd_array);
      result_nd_arrays.emplace_back(nd_array);
    }

    // Call the primitive.
    packed_func.CallPacked(TVMArgs(values.data(), codes.data(), static_cast<int>(arg_len)), &rv);

    // Unflatten the results.
    return ToADTOrNDArray(ftn->ret_type, result_nd_arrays);
  }

  /*!
   * \brief Invoke \p closure with \p args. If \p bind is defined then this is a recursive
   * closure and \p bind should refer to itself.
   */
  ObjectRef Invoke(const InterpreterClosure& closure, const Array<ObjectRef>& args,
                   const Var& bind = Var()) {
    // Get a reference to the function inside the closure.
    Function func = closure->func;
    ICHECK_EQ(func->params.size(), args.size());

    if (func->HasNonzeroAttr(attr::kPrimitive)) {
      if (const CallNode* call_node = closure->func->body.as<CallNode>()) {
        if (call_node->op == debug_op_) {
          // Special case: Calling the debug tracing function.
          auto dattrs = call_node->attrs.as<DebugAttrs>();
          auto interp_state = get_state(call_node->args[0]);

          if (dattrs->debug_func.defined()) {
            dattrs->debug_func(interp_state);
          } else {
            RELAY_DEBUG_INTERP(interp_state);
          }

          return args[0];
        }
      }
    }

    ICHECK(!func->HasNonzeroAttr(attr::kPrimitive))
        << "Calls to primitive functions should have been removed by lowering";

    // Allocate a frame with the parameters and free variables.
    Map<Var, ObjectRef> locals;
    for (size_t i = 0; i < func->params.size(); i++) {
      ICHECK_EQ(locals.count(func->params[i]), 0);
      locals.Set(func->params[i], args[i]);
    }

    // Add the var to value mappings from the Closure's environment.
    for (auto it = closure->env.begin(); it != closure->env.end(); ++it) {
      ICHECK_EQ(locals.count((*it).first), 0);
      locals.Set((*it).first, (*it).second);
    }

    if (bind.defined()) {
      locals.Set(bind, RecClosure(closure, bind));
    }

    return WithFrame<ObjectRef>(Frame(locals), [&]() { return Eval(func->body); });
  }

  ObjectRef VisitExpr_(const CallNode* call_node) final {
    DeviceCopyProps device_copy_props = GetDeviceCopyProps(call_node);
    CallLoweredProps call_lowered_props = GetCallLoweredProps(call_node);

    if (device_copy_props.body.defined()) {
      // TODO(mbs): device_copy cleanup
      LOG(FATAL) << "The interpreter does not support device_copy";
    } else if (call_lowered_props.lowered_func.defined()) {
      // Special case: Call a lowered TIR function.

      // Evaluate only function args
      std::vector<ObjectRef> args;
      for (auto arg : call_lowered_props.arguments) {
        args.push_back(Eval(arg));
      }

      // TODO(mbs): Make calling convention first-class in Relay.
      Array<GlobalVar> all_prim_fn_vars;
      if (call_lowered_props.attrs.metadata.count("all_prim_fn_vars")) {
        all_prim_fn_vars =
            Downcast<Array<GlobalVar>>(call_lowered_props.attrs.metadata.at("all_prim_fn_vars"));
      }
      GlobalVar prim_shape_fn_var;
      if (call_lowered_props.attrs.metadata.count("prim_shape_fn_var")) {
        prim_shape_fn_var =
            Downcast<GlobalVar>(call_lowered_props.attrs.metadata.at("prim_shape_fn_var"));
      }
      Array<GlobalVar> all_prim_shape_fn_vars;
      if (call_lowered_props.attrs.metadata.count("all_prim_shape_fn_vars")) {
        all_prim_shape_fn_vars = Downcast<Array<GlobalVar>>(
            call_lowered_props.attrs.metadata.at("all_prim_shape_fn_vars"));
      }
      Array<Integer> prim_shape_fn_states;
      if (call_lowered_props.attrs.metadata.count("prim_shape_fn_states")) {
        prim_shape_fn_states =
            Downcast<Array<Integer>>(call_lowered_props.attrs.metadata.at("prim_shape_fn_states"));
      }

      size_t num_shape_inputs = 0;
      if (call_lowered_props.attrs.metadata.count("prim_shape_fn_num_inputs")) {
        num_shape_inputs = static_cast<size_t>(
            Downcast<Integer>(call_lowered_props.attrs.metadata.at("prim_shape_fn_num_inputs"))
                ->value);
      }
      size_t num_shape_outputs = 0;
      if (call_lowered_props.attrs.metadata.count("prim_shape_fn_num_outputs")) {
        num_shape_outputs = static_cast<size_t>(
            Downcast<Integer>(call_lowered_props.attrs.metadata.at("prim_shape_fn_num_outputs"))
                ->value);
      }
      ICHECK(config_->optional_homogeneous_target.defined());
      return InvokePrimitiveOp(call_lowered_props.lowered_func, all_prim_fn_vars,
                               config_->optional_homogeneous_target, prim_shape_fn_var,
                               all_prim_shape_fn_vars, prim_shape_fn_states, num_shape_inputs,
                               num_shape_outputs, config_->host_virtual_device->target, args);
    } else {  // All other calls
      // Evaluate all arguments
      std::vector<ObjectRef> args;
      for (auto arg : call_node->args) {
        args.push_back(Eval(arg));
      }

      if (call_node->op == OnDeviceOp()) {
        // Special case: The call 'on_device(expr)' denotes that expr should be executed on
        // a particular device. We can ignore this during interpretation.
        ICHECK_EQ(call_node->args.size(), 1UL);
        return args[0];
      }
      if (const ConstructorNode* con = call_node->op.as<ConstructorNode>()) {
        // Special case: ADT constructor

        return ConstructorValue(con->tag, args, GetRef<Constructor>(con));
      }

      if (const OpNode* op_node = call_node->op.as<OpNode>()) {
        // Except for call_lowered and on_device, we should not find calls to operators after
        // running fusion and lowering.
        LOG(FATAL) << "found " << op_node->name
                   << "; operators should have been removed by previous passes; try "
                      "fusing and lowering";
      }

      // Now we just evaluate and expect to find a closure.
      // TODO(@electriclilies): How should call_lowered behave with closures?
      ObjectRef fn_val = Eval(call_node->op);
      if (auto closure = fn_val.as<InterpreterClosure>()) {
        return Invoke(closure.value(), args);
      } else if (const RecClosureObj* closure_node = fn_val.as<RecClosureObj>()) {
        return Invoke(closure_node->clos, args, closure_node->bind);
      } else {
        LOG(FATAL) << "internal error: type error, expected function value in the call "
                   << "position";
        return ObjectRef();
      }
    }
  }

  ObjectRef VisitExpr_(const LetNode* let) final {
    if (auto func = let->value.as<Function>()) {
      auto clo = MakeClosure(func.value(), let->var);
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
    ICHECK(adt_obj) << "internal error: when evaluating TupleGetItem expected an ADT value";
    auto adt = GetRef<ADT>(adt_obj);
    ICHECK_LT(static_cast<size_t>(op->index), adt.size()) << "internal error: index out of bounds";
    return adt[op->index];
  }

  ObjectRef VisitExpr_(const IfNode* op) final {
    ObjectRef v = Eval(op->cond);
    if (v->IsInstance<NDArray::ContainerType>()) {
      auto nd_array = Downcast<NDArray>(v);
      Device cpu_dev;
      cpu_dev.device_type = kDLCPU;
      cpu_dev.device_id = 0;
      NDArray cpu_array = nd_array.CopyTo(cpu_dev);
      ICHECK_EQ(DataType(cpu_array->dtype), DataType::Bool());
      // TODO(@jroesch, @MK): Refactor code into helper from DCE.
      if (reinterpret_cast<uint8_t*>(cpu_array->data)[0]) {
        return Eval(op->true_branch);
      } else {
        return Eval(op->false_branch);
      }
    } else {
      LOG(FATAL) << "type error, type system should have caught this";
    }
  }

  ObjectRef VisitExpr_(const RefWriteNode* op) final {
    ObjectRef r = Eval(op->ref);
    if (const RefValueObj* rv = r.as<RefValueObj>()) {
      rv->value = Eval(op->value);
      return ADT::Tuple(std::vector<ObjectRef>());
    } else {
      LOG(FATAL) << "type error, type system should have caught this";
    }
  }

  ObjectRef VisitExpr_(const RefCreateNode* op) final { return RefValue(Eval(op->value)); }

  ObjectRef VisitExpr_(const RefReadNode* op) final {
    ObjectRef r = Eval(op->ref);
    if (const RefValueObj* rv = r.as<RefValueObj>()) {
      return rv->value;
    } else {
      LOG(FATAL) << "type error, type system should have caught this";
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
  }

  bool VisitPattern_(const PatternConstructorNode* op, const ObjectRef& v) final {
    const ConstructorValueObj* cvn = v.as<ConstructorValueObj>();
    ICHECK(cvn) << "need to be a constructor for match";
    ICHECK_NE(op->constructor->tag, -1);
    ICHECK_NE(cvn->tag, -1);
    if (op->constructor->tag == cvn->tag) {
      ICHECK_EQ(op->patterns.size(), cvn->fields.size());
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
    ICHECK_EQ(op->patterns.size(), adt.size());
    for (size_t i = 0; i < op->patterns.size(); ++i) {
      if (!VisitPattern(op->patterns[i], adt[i])) {
        return false;
      }
    }
    return true;
  }

  bool VisitPattern_(const PatternWildcardNode* op, const ObjectRef& v) final { return true; }

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
    auto state = InterpreterState(e, stack);
    return state;
  }

 private:
  // Unified module. Functions are annotated with their target.
  // All expressions are eval'ed w.r.t. the definitions in this module.
  // This module contains functions that used to be in main_module and the per_target_module (TIR
  // functions) in one module.
  IRModule unified_mod_;
  // Cached packed functions for the primitives and shape functions, keyed by target and
  // global var name.
  std::unordered_map<std::pair<Target, std::string>, PackedFunc, PairHash> compiled_packed_funcs_;
  /*! \brief Compilation config describing the available targets. */
  CompilationConfig config_;
  // Unique device on which primitives (but not shape functions) will be executed.
  // (For simplicity we only run the interpreter on a single device.)
  Device device_;
  // Call stack.
  Stack stack_;
  // The distinguished 'debug' operator, which is handled specially.
  const Op& debug_op_;
};

/*!
 * Lowers all calls to primitives in \p mod appropriate for \p config. Returns the
 * rewritten \p mod and target-specific modules containing bindings for all TIR primitive
 * functions needed by the rewritten module.
 */
IRModule Prepare(IRModule mod, const CompilationConfig& config) {
  // Run minimal transforms on module to establish invariants needed by interpreter.
  transform::Sequential seq(
      {transform::SimplifyInference(), qnn::transform::Legalize(),
       // Figure out which devices should be used to execute.
       // TODO(mbs): Should ignore all existing annotations when constant folding
       transform::PlanDevices(config),
       // FuseOps will mark wrapped calls to prim-ops with the 'Primitive'
       // attribute.
       transform::FuseOps(/*fuse_opt_level=*/0),
       // Use ANF to reduce number of cases to handle.
       transform::ToANormalForm(),
       // eta expand to support constructors in argument position.
       transform::EtaExpand(
           /*expand_constructor=*/true, /*expand_global_var=*/false),
       transform::InferType(), tec::LowerTE(/*module_name=*/"intrp", config)});

  transform::PassContext pass_ctx = transform::PassContext::Current();
  With<transform::PassContext> ctx(pass_ctx);
  mod = seq(mod);

  return mod;
}

/*! \brief Check if an expression could be changed by \p Prepare.
 *
 * If not we can evaluate it directly and don't need to bind it into a fresh module.
 */
class NeedsPreparationVisitor : public ExprVisitor {
 public:
  bool needs_preparation = false;

 private:
  void VisitExpr_(const VarNode* vn) override {
    // Could be prim.
    needs_preparation = true;
  }
  // ConstantNode ok
  // GlobalVarNode ok
  void VisitExpr_(const OpNode* op) override {
    // Could be prim.
    needs_preparation = true;
  }
  // TupleNode recurse
  void VisitExpr_(const FunctionNode* op) override {
    // Could be prim.
    needs_preparation = true;
  }
  // CallNode recurse
  void VisitExpr_(const LetNode* ln) override {
    // May bind prim.
    needs_preparation = true;
  }
  // IfNode recurse
  // TupleGetItemNode recurse
  // RefCreateNode recurse
  // RefReadNode recurse
  // RefWriteNode recurse
  // ConstructorNode ok
  void VisitExpr_(const MatchNode* op) override {
    // Needs eta-expansion.
    needs_preparation = true;
  }
};

TypedPackedFunc<ObjectRef(Array<Expr>)> EvalFunction(IRModule mod, Expr expr, Device device,
                                                     Target target) {
  VLOG_CONTEXT << "EvalFunction";
  VLOG(1) << "evaling module:" << std::endl
          << PrettyPrint(mod) << "and expression:" << std::endl
          << PrettyPrint(expr);

  ICHECK_EQ(device.device_type, target->GetTargetDeviceType());
  Array<Target> raw_targets = {target};
  CompilationConfig config(transform::PassContext::Current(), raw_targets);

  //
  // Step 1: Prepare mod.
  //

  // If expr is simple enough we can avoid binding it into the module and
  // just eval it directly.
  NeedsPreparationVisitor visitor;
  visitor.VisitExpr(expr);

  Expr expr_to_eval;
  IRModule mod_with_expr;  // default empty
  if (visitor.needs_preparation) {
    GlobalVar main;
    // Bind expr to a new zero-argument function so it can be prepared along with the module
    // (if any).
    std::pair<IRModule, GlobalVar> mod_and_global;
    if (mod.defined()) {
      // TODO(mbs): Type inference currently assumes all global functions in modules have
      // known result types, and so each global function has it's body types inferred independently
      // and in arbitrary order. However, the interpreter may be called with an expression relative
      // to a 'main' which has no result type annotation, and that expressions will be bound into a
      // fresh global below. Type inference then fails since 'main' has unknown type. We should
      // allow inference on mutually recursive global functions. To workaround, infer the type
      // of mod now. Obviously that won't work if 'main' itself calls other global functions of
      // partial type, but it at least maintains legacy behavior.
      transform::PassContext pass_ctx = transform::PassContext::Current();
      With<transform::PassContext> ctx(pass_ctx);
      mod = transform::InferType()(mod);
      mod_and_global =
          IRModule::FromExprInContext(expr, mod->functions, mod->type_definitions, mod->Imports());
    } else {
      mod_and_global = IRModule::FromExprInContext(expr);
    }
    mod_with_expr = mod_and_global.first;
    expr_to_eval = mod_and_global.second;
  } else {
    if (mod.defined()) {
      mod_with_expr = mod;
    }
    // Prepare won't change expr, so we don't need to worry about binding it into a module
    // and can just eval it directly.
    expr_to_eval = expr;
  }
  IRModule lowered_mod = Prepare(mod_with_expr, config);

  std::shared_ptr<Interpreter> intrp = std::make_shared<Interpreter>(lowered_mod, config, device);

  //
  // Step 2: Evaluate target function to a closure.
  //
  ObjectRef object_ref = intrp->Eval(expr_to_eval);
  if (auto opt = object_ref.as<InterpreterClosure>()) {
    InterpreterClosure closure = opt.value();
    ICHECK(closure->func.defined());

    return TypedPackedFunc<ObjectRef(Array<Expr>)>([intrp, closure](Array<Expr> args) {
      VLOG_CONTEXT << "EvalFunction::Apply";
      VLOG(1) << "evaling closure with " << args.size() << " arguments";
      //
      // Step 3: Apply closure to arguments.
      //
      ICHECK_NOTNULL(intrp);
      ICHECK(closure.defined());
      ICHECK(closure->func.defined());
      Array<ObjectRef> evaled_args;
      for (auto arg : args) {
        NeedsPreparationVisitor visitor;
        visitor.VisitExpr(arg);
        ICHECK(!visitor.needs_preparation)
            << "attempting to apply closure to expression which needs preparation: "
            << PrettyPrint(arg);
        evaled_args.push_back(intrp->Eval(arg));
      }
      return intrp->Invoke(closure, evaled_args);
    });
  } else {
    LOG(FATAL) << "expecting expression to have function type and evaluate to a closure";
  }
}

ObjectRef Eval(Expr expr, Map<GlobalTypeVar, TypeData> type_definitions,
               std::unordered_set<String> import_set, Device device, Target target,
               Map<String, ObjectRef> attrs) {
  ICHECK_EQ(device.device_type, target->GetTargetDeviceType());
  Array<Target> raw_targets = {target};
  CompilationConfig config(transform::PassContext::Current(), raw_targets);

  std::pair<IRModule, GlobalVar> mod_and_global =
      IRModule::FromExprInContext(expr, /*global_funcs=*/{}, type_definitions, import_set);

  IRModule mod = Prepare(WithAttrs(mod_and_global.first, {attrs}), config);

  Interpreter intrp(mod, config, device);
  Expr expr_to_eval = mod->GetGlobalVar(mod_and_global.second->name_hint);
  if (expr.as<BaseFuncNode>() == nullptr) {
    // TODO(mbs): IRModule::FromExpr will implicitly close over the free vars of expr
    // unless it is a function, so we must reverse that in the expression to eval.
    // This should done more systematically.
    expr_to_eval = Call(expr_to_eval, {});
  }
  return intrp.Eval(expr_to_eval);
}

TVM_REGISTER_GLOBAL("relay.backend.EvalFunction").set_body_typed(EvalFunction);

}  // namespace relay
}  // namespace tvm
