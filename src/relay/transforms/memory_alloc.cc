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

inline Constant MakeConstant(const std::vector<int64_t>& value) {
  return MakeConstantTensor(DataType::Int(64), {static_cast<int64_t>(value.size())}, value);
}

inline Expr AllocTensor(const Expr& storage, tvm::relay::Expr shape, DataType dtype,
                        Array<IndexExpr> assert_shape, DLDeviceType offset_device_type) {
  auto offset =
      OnDevice(MakeConstantScalar(DataType::Int(64), 0), offset_device_type, /*is_fixed=*/true);
  return AllocTensor(storage, offset, shape, dtype, assert_shape);
}

// Check if the primitive function contains only reshape ops.
bool IsReshapeOnly(const Expr& expr) {
  if (const FunctionNode* func = expr.as<FunctionNode>()) {
    return func->HasNonzeroAttr(attr::kReshapeOnly);
  }
  if (const CallNode* call = expr.as<CallNode>()) {
    if (call->op == CallLoweredOp()) {
      CallLoweredProps call_lowered_props = GetCallLoweredProps(call);
      Map<String, ObjectRef> metadata = call_lowered_props.attrs.metadata;
      return metadata.count(attr::kReshapeOnly) &&
             (Downcast<tvm::Integer>(metadata[attr::kReshapeOnly])->value == 1);
    }
  }
  return false;
}

class DialectRewriter : public transform::DeviceAwareExprMutator {
 public:
  DialectRewriter(IRModule mod, const Target& target_host)
      : transform::DeviceAwareExprMutator(std::move(mod)), target_host_(target_host) {}

  Function Rewrite(const Function& expr) { return Downcast<Function>(Mutate(expr)); }

  Expr VisitExpr_(const TupleNode* tn) final {
    LetList& scope = scopes_.back();
    Array<Expr> new_fields;
    for (auto field : tn->fields) {
      auto new_field = Mutate(field);
      if (new_field->IsInstance<ConstantNode>()) {
        Var const_var("const", Type(nullptr));
        new_field = scope.Push(const_var, new_field);
      }
      new_fields.push_back(new_field);
    }
    return Tuple(new_fields);
  }

  void PreVisitLetBlock_(const LetNode* let_node) final { scopes_.emplace_back(); }

  std::pair<Var, Expr> PreVisitLetBinding_(const Var& var, const Expr& value) final {
    Expr new_value = Mutate(value);
    scopes_.back().Push(var, new_value);
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

  Expr DeviceAwareVisitExpr_(const CallNode* cn) final {
    Call call = GetRef<Call>(cn);
    DLDeviceType device_type = GetInScopeDeviceType(call);
    if (IsPrimitive(cn)) {
      // Because we are in ANF we do not need to visit the arguments.
      // TODO(mbs): But does so anyway...
      LetList& scope = scopes_.back();
      std::vector<Expr> new_args;
      for (const auto& it : cn->args) {
        new_args.push_back(Mutate(it));
      }

      Tuple ins(new_args);
      Type ret_type = cn->checked_type_;
      std::vector<TensorType> out_types = FlattenTupleType(ret_type);

      // Handle fused op that only contains reshape op
      if (IsReshapeOnly(cn->op)) {
        Function func = Downcast<Function>(cn->op);
        return EmitReshapeTensor(&scope, func, new_args, ret_type);
      }

      // Handle device copy op
      if (IsDeviceCopy(cn->op)) {
        Attrs attr;
        if (const auto* fn = cn->op.as<FunctionNode>()) {
          const auto* copy_call = fn->body.as<CallNode>();
          CHECK(copy_call);
          attr = copy_call->attrs;
        } else {
          attr = cn->attrs;
        }
        const DeviceCopyAttrs* copy_attr = attr.as<DeviceCopyAttrs>();
        CHECK(copy_attr);
        return DeviceCopy(new_args[0], copy_attr->src_dev_type, copy_attr->dst_dev_type);
      } else if (IsDynamic(ret_type)) {
        Function func = Downcast<Function>(cn->op);
        // TODO(mbs): Device id is always zero.
        Device device{device_type, /*device_id=*/0};
        return DynamicInvoke(&scope, func, ins, new_args, out_types, ret_type, device);
      } else {
        // Handle the static case
        Array<Expr> outs;
        for (size_t i = 0; i < out_types.size(); ++i) {
          DLDeviceType device_type = GetInScopeDeviceType(GetRef<Call>(cn));
          // TODO(mbs): Device id is always zero.
          Device device{device_type, /*device_id=*/0};
          auto out = MakeStaticAllocation(&scope, out_types[i], device, std::to_string(i));
          outs.push_back(out);
        }
        Tuple output(outs);
        // TODO(mbs): Capture device in attributes.
        Expr invoke = InvokeTVMOp(cn->op, ins, output);
        scope.Push(OnDevice(invoke, device_type, /*is_fixed=*/true));
        return ToTupleType(ret_type,
                           std::vector<Expr>(output->fields.begin(), output->fields.end()));
      }
    } else {
      return transform::DeviceAwareExprMutator::DeviceAwareVisitExpr_(cn);
    }
  }

 private:
  // Insert a device copy node.
  Expr DeviceCopy(const Expr& inp, int src_dev, int dst_dev) {
    return Mutate(relay::DeviceCopy(inp, static_cast<DLDeviceType>(src_dev),
                                    static_cast<DLDeviceType>(dst_dev)));
  }

  // Check if a call invokes a primitive function.
  bool IsPrimitive(const CallNode* call) const {
    if (const auto* fn = call->op.as<FunctionNode>()) {
      return fn->HasNonzeroAttr(attr::kPrimitive);
    }
    return false;
  }

  // Check if the current relay expression is a device copy call. We can simply
  // check the body of it if it is a function because the device_copy op is opaque.
  bool IsDeviceCopy(const Expr& expr) const {
    if (const auto* fn = expr.as<FunctionNode>()) {
      auto body = fn->body;
      const CallNode* call = body.as<CallNode>();
      return call && call->op == device_copy_op_;
    } else if (const CallNode* cn = expr.as<CallNode>()) {
      return cn->op == device_copy_op_;
    } else {
      return false;
    }
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
  Var MakeStaticAllocation(LetList* scope, const TensorType& type, Device dev, String name_hint) {
    std::vector<int64_t> int_shape;
    for (auto it : type->shape) {
      const auto* imm = it.as<IntImmNode>();
      CHECK(imm) << "expect static int shape";
      int_shape.push_back(imm->value);
    }
    Expr shape = OnDevice(MakeConstant(int_shape), cpu_device_.device_type, /*is_fixed=*/true);
    Expr size = OnDevice(ComputeStorage(type), cpu_device_.device_type, /*is_fixed=*/true);
    // Alignment is directly captured in the instruction rather than calculated, so we
    // don't want to wrap it with an "on_device".
    Expr alignment = ComputeAlignment(type->dtype);
    // Run type inference later to get the correct type.
    Var var("storage_" + name_hint, Type(nullptr));
    Expr value = OnDevice(AllocStorage(size, alignment, dev, type->dtype), dev.device_type,
                          /*is_fixed=*/true);
    auto sto = scope->Push(var, value);

    // TODO(@jroesch): There is a bug with typing based on the constant shape.
    auto tensor = OnDevice(
        AllocTensor(sto, shape, type->dtype, /*assert_shape=*/type->shape, cpu_device_.device_type),
        dev.device_type, /*is_fixed=*/true);
    Var tensor_var("tensor_" + name_hint, Type(nullptr));
    return scope->Push(tensor_var, tensor);
  }

  // Insert the shape function given a primitive function.
  Array<Expr> EmitShapeFunc(LetList* scope, const Function& func,
                            const std::vector<Expr>& new_args) {
    Array<Expr> shape_func_ins;

    tec::TECompiler compiler;

    tec::CCacheKey key(func, target_host_);
    auto cfunc = compiler->LowerShapeFunc(key);
    auto input_states = cfunc->shape_func_param_states;

    Array<Integer> is_inputs;
    int input_pos = 0;
    CHECK_EQ(new_args.size(), input_states.size());
    for (size_t i = 0; i < new_args.size(); ++i) {
      Expr arg = new_args[i];
      Type ty;
      if (const auto* vn = arg.as<VarNode>()) {
        ty = vn->type_annotation;
      } else {
        ty = arg->checked_type();
      }
      int state = input_states[i]->value;
      // Pass Shapes
      if (state == tec::kNeedInputShape) {
        std::vector<Expr> exprs = FromTupleType(ty, arg);
        for (size_t j = 0; j < exprs.size(); ++j) {
          Expr sh_of = Mutate(ShapeOf(exprs[j]));  // already accounts for device
          Var in_shape_var("in_shape_" + std::to_string(input_pos + j), Type(nullptr));
          shape_func_ins.push_back(scope->Push(in_shape_var, sh_of));
          input_pos++;
        }
        is_inputs.push_back(0);
      } else if (state == tec::kNeedInputData) {
        auto new_arg = Mutate(arg);  // already accounts for device
        DLDeviceType device_type = GetInScopeDeviceType(arg);
        if (device_type != cpu_device_.device_type) {
          new_arg = OnDevice(DeviceCopy(new_arg, device_type, cpu_device_.device_type),
                             cpu_device_.device_type, /*is_fixed=*/true);
        }
        Var in_shape_var("in_shape_" + std::to_string(input_pos), Type(nullptr));
        shape_func_ins.push_back(scope->Push(in_shape_var, new_arg));
        input_pos++;
        is_inputs.push_back(1);
      } else {
        // TODO(@jroesch): handle kNeedBoth
        LOG(FATAL) << "unsupported shape function input state";
      }
    }

    Array<Expr> out_shapes;
    for (size_t i = 0; i < cfunc->outputs.size(); ++i) {
      auto out = cfunc->outputs[i];
      auto tt = TensorType(out->shape, out->dtype);
      // Put shape func on CPU. This also ensures that everything between
      // shape_of and shape_func are on CPU.
      auto alloc = OnDevice(MakeStaticAllocation(scope, tt, cpu_device_, std::to_string(i)),
                            cpu_device_.device_type, /*is_fixed=*/true);
      Var shape_func_out_var("shape_func_out_" + std::to_string(i), Type(nullptr));
      alloc = scope->Push(shape_func_out_var, alloc);
      out_shapes.push_back(alloc);
    }
    auto shape_call = OnDevice(ShapeFunc(func, Tuple(shape_func_ins), Tuple(out_shapes), is_inputs),
                               cpu_device_.device_type, /*is_fixed=*/true);
    Var shape_func_var("shape_func", Type(nullptr));
    scope->Push(shape_func_var, shape_call);
    return out_shapes;
  }

  // Generate the code for invoking a TVM op with a dynamic shape.
  Expr DynamicInvoke(LetList* scope, const Function& func, const Tuple& ins,
                     const std::vector<Expr>& new_args, const std::vector<TensorType>& out_types,
                     const Type& ret_type, Device dev) {
    auto out_shapes = EmitShapeFunc(scope, func, new_args);
    std::vector<Var> storages;
    CHECK_EQ(out_shapes.size(), out_types.size());
    for (size_t i = 0; i < out_shapes.size(); ++i) {
      auto out_shape = out_shapes[i];
      auto out_type = out_types[i];
      auto size = OnDevice(ComputeStorageInRelay(out_shape, out_type), cpu_device_.device_type,
                           /*is_fixed=*/true);
      // Alignment is directly captured in the instruction so don't wrap in "on_device".
      auto alignment = ComputeAlignment(out_type->dtype);
      Var sto_var("storage_" + std::to_string(i), Type(nullptr));
      auto val = OnDevice(AllocStorage(size, alignment, dev, out_type->dtype), dev.device_type,
                          /*is_fixed=*/true);
      storages.push_back(scope->Push(sto_var, val));
    }

    Array<Expr> outs;
    for (size_t i = 0; i < storages.size(); ++i) {
      auto out_shape = out_shapes[i];
      auto out_type = out_types[i];
      auto storage = storages[i];
      auto alloc = OnDevice(AllocTensor(storage, out_shape, out_type->dtype, out_type->shape,
                                        cpu_device_.device_type),
                            dev.device_type, /*is_fixed=*/true);
      Var out_var("out_" + std::to_string(i), Type(nullptr));
      outs.push_back(scope->Push(out_var, alloc));
    }

    Tuple tuple_outs(outs);
    auto call = InvokeTVMOp(func, ins, tuple_outs);
    auto invoke = OnDevice(call, dev.device_type, /*is_fixed=*/true);
    scope->Push(invoke);
    return ToTupleType(ret_type,
                       std::vector<Expr>(tuple_outs->fields.begin(), tuple_outs->fields.end()));
  }

  Expr EmitReshapeTensor(LetList* scope, const Function& func, const std::vector<Expr>& new_args,
                         const Type& ret_type) {
    TensorType ret_ty = Downcast<TensorType>(ret_type);
    Expr shape_expr;
    if (IsDynamic(ret_type)) {
      auto out_shapes = EmitShapeFunc(scope, func, new_args);
      shape_expr = out_shapes[0];
    } else {
      std::vector<int64_t> shape;
      for (const auto& it : ret_ty->shape) {
        const auto* imm = it.as<IntImmNode>();
        CHECK(imm) << "expect static int shape";
        shape.push_back(imm->value);
      }
      shape_expr = OnDevice(MakeConstant(shape), cpu_device_.device_type, /*is_fixed=*/true);
    }
    return ReshapeTensor(new_args[0], shape_expr, ret_ty->shape);
  }

 private:
  const Op& device_copy_op_ = Op::Get("device_copy");

  Target target_host_;
  std::vector<LetList> scopes_;

  runtime::DataType compute_dtype_ = runtime::DataType::Int(64);
  Device cpu_device_{kDLCPU, 0};
};

namespace transform {

Pass ManifestAlloc(Target target_host, Map<tvm::Integer, tvm::Target> targets) {
  CheckAndUpdateHostConsistency(&targets, &target_host);
  return tvm::transform::CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        // We need to mutate module, therefore making a copy of it.
        mod.CopyOnWrite();
        mod->ImportFromStd("core.rly");
        mod = relay::transform::InferType()(mod);

        auto glob_funcs = mod->functions;
        for (const auto& it : glob_funcs) {
          if (auto* func_node = it.second.as<FunctionNode>()) {
            auto func = GetRef<Function>(func_node);
            auto rewriter = DialectRewriter(mod, target_host);
            auto updated_func = rewriter.Rewrite(func);

            mod->Update(it.first, updated_func);
          }
        }

        mod = relay::transform::InferType()(mod);
        return mod;
      },
      0, "ManifestAlloc", {});
}

TVM_REGISTER_GLOBAL("relay.transform.ManifestAlloc")
    .set_body_typed([](Target target_host, Map<tvm::Integer, tvm::Target> targets) {
      CheckAndUpdateHostConsistency(&targets, &target_host);
      return ManifestAlloc(target_host, targets);
    });

}  // namespace transform

}  // namespace relay
}  // namespace tvm
