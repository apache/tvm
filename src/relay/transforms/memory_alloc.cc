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
 * \file src/relay/transforms/memory_alloc.cc
 * \brief A pass for manifesting explicit memory allocations.
 */

#include <tvm/node/structural_equal.h>
#include <tvm/node/structural_hash.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/attrs/call.h>
#include <tvm/relay/attrs/device_copy.h>
#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/op.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/target.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_set>
#include <vector>

#include "../backend/te_compiler.h"
#include "../backend/te_compiler_cache.h"
#include "../op/annotation/annotation.h"
#include "../op/call/call.h"
#include "../op/memory/device_copy.h"
#include "../op/memory/memory.h"
#include "../op/vm/vm.h"
#include "./device_aware_visitors.h"
#include "./let_list.h"
#include "./pass_utils.h"
#include "./pattern_utils.h"

using namespace tvm::runtime;

namespace tvm {
namespace relay {

class DialectRewriter : public transform::DeviceAwareExprMutator {
 public:
  DialectRewriter(IRModule mod, VirtualDevice host_virtual_device)
      : transform::DeviceAwareExprMutator(mod),
        mod_(std::move(mod)),
        host_virtual_device_(std::move(host_virtual_device)) {}

  Function Rewrite(const Function& expr) { return Downcast<Function>(Mutate(expr)); }

 private:
  using ExprMutator::VisitExpr_;

  Expr VisitExpr_(const TupleNode* tuple_node) final {
    LetList& scope = scopes_.back();
    Array<Expr> new_fields;
    new_fields.reserve(tuple_node->fields.size());

    for (auto field : tuple_node->fields) {
      auto new_field = Mutate(field);
      if (const auto* op = new_field.as<ConstantNode>()) {
        DataType dtype(op->data->dtype);
        bool is_simple_const = (dtype == DataType::Int(32) || dtype == DataType::Int(64) ||
                                dtype == DataType::Float(32) || dtype == DataType::Float(64) ||
                                dtype == DataType::Bool());
        if (!op->is_scalar() || !is_simple_const) {
          VirtualDevice virtual_device = GetVirtualDevice(field);
          ICHECK(!virtual_device->IsFullyUnconstrained());
          Var const_var("const", Type(nullptr));
          new_field = scope.Push(const_var, MaybeOnDeviceFixed(new_field, virtual_device));
        }
      }
      new_fields.push_back(new_field);
    }
    return WithFields(GetRef<Tuple>(tuple_node), new_fields);
  }

  void PreVisitLetBlock_(const LetNode* let_node) final { scopes_.emplace_back(); }

  std::pair<Var, Expr> PreVisitLetBinding_(const Var& var, const Expr& value) final {
    Expr new_value = Mutate(value);
    VirtualDevice virtual_device = GetVirtualDevice(value);
    ICHECK(!virtual_device->IsFullyUnconstrained());
    scopes_.back().Push(var, MaybeOnDeviceFixed(new_value, virtual_device));
    // Since we always need a let block on which to bind sub-expressions the rewritten bindings
    // are tracked in the current scopes. But return the rewritten binding anyway.
    return {var, new_value};
  }

  Expr PostVisitLetBlock_(const LetNode* pre_let_node, const LetNode* post_let_node) final {
    // The current scope has captured all the rewritten let-binding, as well as any additional
    // bindings we needed to add. All we need is the rewritted body.
    Expr new_body = post_let_node->body;
    while (const auto* inner_let_node = new_body.as<LetNode>()) {
      new_body = inner_let_node->body;
    }
    auto ret = scopes_.back().Get(new_body);
    scopes_.pop_back();
    return ret;
  }

  Expr DeviceAwareVisitExpr_(const CallNode* call_node) final {
    DeviceCopyProps device_copy_props = GetDeviceCopyProps(call_node);
    CallLoweredProps call_lowered_props = GetCallLoweredProps(call_node);

    if (device_copy_props.body.defined()) {
      // Special case: device_copy calls remain in their original (and functional) form.
      // TODO(mbs): device_copy cleanup.
      return transform::DeviceAwareExprMutator::DeviceAwareVisitExpr_(call_node);
    }

    if (!call_lowered_props.lowered_func.defined()) {
      // This is a call to a user-defined Relay functinon, which will be handled directly by
      // the VM and does not need conversion to DPS.
      return transform::DeviceAwareExprMutator::DeviceAwareVisitExpr_(call_node);
    }

    Call call = GetRef<Call>(call_node);
    VLOG(1) << "converting lowered call to DPS:" << std::endl << PrettyPrint(call);

    VirtualDevice virtual_device = GetVirtualDevice(call);
    ICHECK(!virtual_device->IsFullyUnconstrained());
    ICHECK(!scopes_.empty())
        << "Calls out of a let block are not supported, do you forget to transform "
        << "with ToANormalForm or set opt_level >= 1 in the pass context?";
    LetList& scope = scopes_.back();

    std::vector<Expr> new_args;
    for (const auto& arg : call_lowered_props.arguments) {
      new_args.push_back(Mutate(arg));
    }
    Tuple ins(new_args);
    Type ret_type = call_node->checked_type_;
    std::vector<TensorType> out_types = FlattenTupleType(ret_type);

    // Handle reshape.
    // Original:
    //   reshape(body, <ReshapeAttrs>)
    //   dyn.reshape(body, shape, <ReshapeAttrs>)
    // After FuseOps:
    //   let %f = fn(x, primitive=1, relay.reshape_only=1) { reshape(x, <ReshapeAttrs>) }
    //   %f(body)
    // After LowerTEPass:
    //   call_lowered(@xxx_reshape, (body), <LoweredCallAttrs with
    //   relay_attrs|->dict[relay.reshape_only] = 1)
    //    -OR-
    //   call_lowered(@xxx_dyn_reshape, (body, shape), <LoweredCallAttrs with same>)
    //   where @reshape_xxx is bound as a PrimFunc.
    //   (the name is irrelevant, only the relay.reshape_only attribute matters)
    // After this pass:
    //   vm.reshape_tensor(body, shape, <TIRCallAttrs>)
    if (IsReshapeOnly(call_lowered_props)) {
      return EmitReshapeTensor(&scope, ins, call_lowered_props.attrs, ret_type);
    }

    // At this point we could be calling a PrimFunc or an 'external' and already compiled primitive.
    // The calling conventions are identical.

    // Handle 'dynamic' calls, ie to PrimFuncs whose result shape must be first computed
    // by a companion shape function.
    if (IsDynamic(ret_type)) {
      return DynamicInvoke(&scope, call_lowered_props.lowered_func, ins, call_lowered_props.attrs,
                           out_types, ret_type, virtual_device);
    }

    // Handle ordinary primitive calls.
    Array<Expr> outputs;
    for (size_t i = 0; i < out_types.size(); ++i) {
      outputs.push_back(
          MakeStaticAllocation(&scope, out_types[i], virtual_device, std::to_string(i)));
    }
    Tuple outs(outputs);
    Expr invoke =
        InvokeTVMOp(call_lowered_props.lowered_func, ins, outs,
                    Downcast<DictAttrs>(call_lowered_props.attrs.metadata.at("relay_attrs")));
    scope.Push(MaybeOnDeviceFixed(invoke, virtual_device));
    return ToTupleType(ret_type, std::vector<Expr>(outputs.begin(), outputs.end()));
  }

  /*!
   * \brief Returns the Relay Constant representing the 1d tensor with \p value.
   *
   * CAUTION: Make sure the constant ends up on the correct device.
   */
  inline Constant MakeConstant(const std::vector<int64_t>& value) {
    return MakeConstantTensor(DataType::Int(64), {static_cast<int64_t>(value.size())}, value);
  }

  /*! Returns an \p alloc_tensor call for a tensor of \p shape and \p dtype over \p storage. */
  inline Expr AllocTensor(const Expr& storage, tvm::relay::Expr shape, DataType dtype,
                          Array<IndexExpr> assert_shape) {
    Expr offset =
        MaybeOnDeviceFixed(MakeConstantScalar(DataType::Int(64), 0), host_virtual_device_);
    return tvm::relay::AllocTensor(storage, std::move(offset), std::move(shape), dtype,
                                   assert_shape);
  }

  Expr ComputeAlignment(const DataType& dtype) const {
    int64_t align = dtype.bits() / 8 * dtype.lanes();
    if (align < 64) {
      align = 64;
    }
    return MakeConstantScalar(DataType::Int(64), align);
  }

  Expr ComputeStorageInRelay(const Expr& shape, const TensorType& type) const {
    auto dtype = DataType(type->dtype);
    Expr els = Prod(shape, Array<Integer>(nullptr), false, false);
    Expr num = MakeConstantScalar(DataType::Int(64), dtype.bits() * dtype.lanes());
    Expr add = Add(num, MakeConstantScalar(DataType::Int(64), 7));
    Expr div = MakeConstantScalar(DataType::Int(64), 8);
    Expr ret = Multiply(els, Divide(add, div));
    return std::move(ret);
  }

  Expr ComputeStorage(const TensorType& type) {
    int64_t size = 1;
    for (auto it : type->shape) {
      auto val = it.as<IntImmNode>();
      CHECK(val);
      size *= val->value;
    }
    size *= (type->dtype.bits() * type->dtype.lanes() + 7) / 8;
    return std::move(MakeConstantScalar(DataType::Int(64), size));
  }

  // Allocate a tensor with a statically known shape.
  Var MakeStaticAllocation(LetList* scope, const TensorType& type,
                           const VirtualDevice& virtual_device, String name_hint) {
    std::vector<int64_t> int_shape;
    for (auto it : type->shape) {
      const auto* imm = it.as<IntImmNode>();
      CHECK(imm) << "expect static int shape";
      int_shape.push_back(imm->value);
    }
    Expr shape = MaybeOnDeviceFixed(MakeConstant(int_shape), host_virtual_device_);
    Expr size = MaybeOnDeviceFixed(ComputeStorage(type), host_virtual_device_);
    // Alignment is directly captured in the instruction rather than calculated, so we
    // don't want to wrap it with an "on_device".
    Expr alignment = ComputeAlignment(type->dtype);
    // Run type inference later to get the correct type.
    Var var("storage_" + name_hint, Type(nullptr));
    Expr value = AllocStorage(size, shape, alignment, virtual_device, type->dtype);
    auto sto = scope->Push(var, MaybeOnDeviceFixed(value, virtual_device));

    // TODO(@jroesch): There is a bug with typing based on the constant shape.
    auto tensor = AllocTensor(sto, shape, type->dtype, /*assert_shape=*/type->shape);
    Var tensor_var("tensor_" + name_hint, Type(nullptr));
    return scope->Push(tensor_var, MaybeOnDeviceFixed(tensor, virtual_device));
  }

  /*!
   * \brief Appends to \p scope the computation necessary to call the shape function given
   * in \p tir_call_attrs and bind the resulting result shapes into \p scope. The result
   * shapes are for a call to a primitive with \p ins arguments. Some combinationn of the
   * data and/or shapes of \p ins will be needed by the shape function.
   */
  Array<Expr> EmitShapeFunc(LetList* scope, const Tuple& ins, const CallLoweredAttrs& attrs) {
    ICHECK(attrs.metadata.count("prim_shape_fn_states"));
    Array<Integer> input_states =
        Downcast<Array<Integer>>(attrs.metadata.at("prim_shape_fn_states"));
    ICHECK(attrs.metadata.count("prim_shape_fn_var"));
    auto prim_fn_var = Downcast<GlobalVar>(attrs.metadata.at("prim_shape_fn_var"));

    const auto* func_type_node = prim_fn_var->checked_type().as<FuncTypeNode>();
    ICHECK(func_type_node);

    // Establish the arguments to the shape function.
    Array<Expr> shape_func_ins;
    int input_pos = 0;
    ICHECK_EQ(ins->fields.size(), input_states.size());
    for (size_t i = 0; i < ins->fields.size(); ++i) {
      const Expr& arg = ins->fields[i];
      Type ty;
      if (const auto* vn = arg.as<VarNode>()) {
        ty = vn->type_annotation;
      } else {
        ty = arg->checked_type();
      }
      int64_t state = input_states[i]->value;
      // Pass Shapes
      if (state == tec::kNeedInputShape) {
        std::vector<Expr> exprs = FromTupleType(ty, arg);
        for (size_t j = 0; j < exprs.size(); ++j) {
          Expr sh_of = Mutate(ShapeOf(exprs[j]));
          Var in_shape_var("in_shape_" + std::to_string(input_pos + j), Type(nullptr));
          shape_func_ins.push_back(
              scope->Push(in_shape_var, MaybeOnDeviceFixed(sh_of, host_virtual_device_)));
          input_pos++;
        }
      } else if (state == tec::kNeedInputData) {
        auto new_arg = Mutate(arg);  // already accounts for device
        VirtualDevice arg_virtual_device = GetVirtualDevice(arg);
        ICHECK(!arg_virtual_device->IsFullyUnconstrained());
        // The dynamic shape function is expecting its data on the host/CPU, so insert a
        // device_copy otherwise. (We'll need to fuse & lower these copies in the same way
        // we fuse & lower other operators we insert for, eg, dynamic tensor size calculation.)
        new_arg = MaybeDeviceCopy(MaybeOnDeviceFixed(new_arg, arg_virtual_device),
                                  arg_virtual_device, host_virtual_device_);
        Var in_shape_var("in_shape_" + std::to_string(input_pos), Type(nullptr));
        shape_func_ins.push_back(
            scope->Push(in_shape_var, MaybeOnDeviceFixed(new_arg, host_virtual_device_)));
        input_pos++;
      } else {
        // TODO(@jroesch): handle kNeedBoth
        LOG(FATAL) << "unsupported shape function input state";
      }
    }
    ICHECK_EQ(shape_func_ins.size(), func_type_node->arg_types.size());

    // Establish the result shapes.
    const auto* res_tuple_node = func_type_node->ret_type.as<TupleTypeNode>();
    ICHECK(res_tuple_node);

    Array<Expr> out_shapes;
    for (size_t i = 0; i < res_tuple_node->fields.size(); ++i) {
      const auto* tensor_type_node = res_tuple_node->fields[i].as<TensorTypeNode>();
      ICHECK(tensor_type_node);
      // Put the shape func on the host. This also ensures that everything between
      // shape_of and shape_func is similarly on the host.
      Var alloc = MakeStaticAllocation(scope, GetRef<TensorType>(tensor_type_node),
                                       host_virtual_device_, "out_shape_" + std::to_string(i));
      out_shapes.push_back(alloc);
    }

    // Represent the call in DPS form.
    auto shape_call = InvokeTVMOp(prim_fn_var, Tuple(shape_func_ins), Tuple(out_shapes),
                                  Downcast<DictAttrs>(attrs.metadata.at("relay_attrs")));
    Var shape_func_var("shape_func", Type(nullptr));
    scope->Push(shape_func_var, MaybeOnDeviceFixed(shape_call, host_virtual_device_));
    return out_shapes;
  }

  // Generate the code for invoking the TVM primitive \p func who's results have dynamic shapes.
  Expr DynamicInvoke(LetList* scope, const Expr& func, const Tuple& ins,
                     const CallLoweredAttrs& attrs, const std::vector<TensorType>& out_types,
                     const Type& ret_type, const VirtualDevice& virtual_device) {
    Array<Expr> out_shapes = EmitShapeFunc(scope, ins, attrs);
    std::vector<Var> storages;
    CHECK_EQ(out_shapes.size(), out_types.size());
    for (size_t i = 0; i < out_shapes.size(); ++i) {
      auto out_shape = out_shapes[i];
      auto out_type = out_types[i];
      auto size =
          MaybeOnDeviceFixed(ComputeStorageInRelay(out_shape, out_type), host_virtual_device_);
      // Alignment is directly captured in the instruction so don't wrap in "on_device".
      auto alignment = ComputeAlignment(out_type->dtype);
      Var sto_var("storage_" + std::to_string(i), Type(nullptr));
      auto val = AllocStorage(size, out_shape, alignment, virtual_device, out_type->dtype);
      storages.push_back(scope->Push(sto_var, MaybeOnDeviceFixed(val, virtual_device)));
    }

    Array<Expr> outs;
    for (size_t i = 0; i < storages.size(); ++i) {
      auto out_shape = out_shapes[i];
      auto out_type = out_types[i];
      auto storage = storages[i];
      auto alloc = AllocTensor(storage, out_shape, out_type->dtype, out_type->shape);
      Var out_var("out_" + std::to_string(i), Type(nullptr));
      outs.push_back(scope->Push(out_var, MaybeOnDeviceFixed(alloc, virtual_device)));
    }

    Tuple tuple_outs(outs);
    auto call =
        InvokeTVMOp(func, ins, tuple_outs, Downcast<DictAttrs>(attrs.metadata.at("relay_attrs")));
    scope->Push(MaybeOnDeviceFixed(call, virtual_device));
    return ToTupleType(ret_type,
                       std::vector<Expr>(tuple_outs->fields.begin(), tuple_outs->fields.end()));
  }

  Expr EmitReshapeTensor(LetList* scope, const Tuple& ins, const CallLoweredAttrs& attrs,
                         const Type& ret_type) {
    ICHECK_GE(ins->fields.size(), 1);  // static reshape
    ICHECK_LE(ins->fields.size(), 2);  // dynamic reshape, second arg is shape
    TensorType ret_ty = Downcast<TensorType>(ret_type);
    Expr shape_expr;
    if (IsDynamic(ret_type)) {
      // Even though the desired output shape has been passed as the second argument to
      // the dyn.reshape primitive, we'll still call that primitive's shape function. Go figure.
      Array<Expr> out_shapes = EmitShapeFunc(scope, ins, attrs);
      ICHECK_EQ(out_shapes.size(), 1);
      shape_expr = out_shapes[0];
    } else {
      std::vector<int64_t> shape;
      for (const auto& it : ret_ty->shape) {
        const auto* imm = it.as<IntImmNode>();
        CHECK(imm) << "expect static int shape";
        shape.push_back(imm->value);
      }
      shape_expr = MaybeOnDeviceFixed(MakeConstant(shape), host_virtual_device_);
    }
    return ReshapeTensor(ins->fields[0], shape_expr, ret_ty->shape);
  }

 private:
  const Op& device_copy_op_ = Op::Get("device_copy");
  runtime::DataType compute_dtype_ = runtime::DataType::Int(64);
  IRModule mod_;
  VirtualDevice host_virtual_device_;

  std::vector<LetList> scopes_;
};

namespace transform {

Pass ManifestAllocImportStorage() {
  auto pass_func = [](IRModule mod, tvm::transform::PassContext pass_cnxt) {
    mod.CopyOnWrite();
    mod->ImportFromStd("core.rly");
    return mod;
  };
  return tvm::transform::CreateModulePass(pass_func, /*opt_level=*/0, "ManifestAllocImportStorage",
                                          /*required=*/{});
}

Pass ManifestAllocImpl(VirtualDevice host_virtual_device) {
  auto pass_func = [host_virtual_device](Function func, IRModule mod, PassContext ctxt) {
    return DialectRewriter(mod, host_virtual_device).Rewrite(func);
  };
  return CreateFunctionPass(pass_func, 0, "ManifestAllocImpl", {});
}

Pass ManifestAlloc(VirtualDevice cpu_virtual_device) {
  std::vector<Pass> passes = {ManifestAllocImportStorage(), InferType(),
                              ManifestAllocImpl(std::move(cpu_virtual_device)), InferType()};
  return Sequential(passes, "ManifestAlloc");
}

TVM_REGISTER_GLOBAL("relay.transform.ManifestAlloc").set_body_typed(ManifestAlloc);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
