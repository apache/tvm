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

#include "../backend/compile_engine.h"
#include "../op/memory/memory.h"
#include "../op/vm/vm.h"
#include "./pass_utils.h"
#include "let_list.h"
#include "pattern_utils.h"

using namespace tvm::runtime;

namespace tvm {
namespace relay {

using AnalysisResultMap =
    std::unordered_map<Expr, Device, runtime::ObjectPtrHash, runtime::ObjectPtrEqual>;

inline Constant MakeConstant(const std::vector<int64_t>& value) {
  return MakeConstantTensor(DataType::Int(64), {static_cast<int64_t>(value.size())}, value);
}

inline Expr AllocTensor(const Expr& storage, tvm::relay::Expr shape, DataType dtype,
                        Array<IndexExpr> assert_shape) {
  auto offset = MakeConstantScalar(DataType::Int(64), 0);
  return AllocTensor(storage, offset, shape, dtype, assert_shape);
}

// Check if the primitive function contains only reshape ops.
bool IsReshapeOnly(const Expr& expr) {
  if (const FunctionNode* func = expr.as<FunctionNode>()) {
    return func->HasNonzeroAttr(attr::kReshapeOnly);
  }
  if (const CallNode* call = expr.as<CallNode>()) {
    if (call->attrs.defined()) {
      if (auto tir_call_attrs = call->attrs.as<TIRCallAttrs>()) {
        Map<String, ObjectRef> metadata = tir_call_attrs->metadata;
        return metadata.count(attr::kReshapeOnly) &&
               (Downcast<tvm::Integer>(metadata[attr::kReshapeOnly])->value == 1);
      }
    }
  }
  return false;
}

class DialectRewriter : public ExprMutator {
 public:
  DialectRewriter(const Target& target_host, const AnalysisResultMap& context_analysis_map)
      : target_host_(target_host), context_analysis_map_(context_analysis_map) {}

  // Get the device of an expression.
  Device GetDevice(const Expr& expr) const {
    auto it = context_analysis_map_.find(expr);
    CHECK(it != context_analysis_map_.end()) << "Cannot find expr in the context analysis map:\n"
                                             << AsText(expr, false);
    return it->second;
  }

  Function Rewrite(const Function& expr) {
    auto ret = ExprMutator::Mutate(expr);
    return Downcast<Function>(ret);
  }

  Expr VisitExpr_(const TupleNode* tn) final {
    LetList& scope = scopes_.back();
    Array<Expr> new_fields;
    for (auto field : tn->fields) {
      auto new_field = ExprMutator::Mutate(field);
      if (new_field->IsInstance<ConstantNode>()) {
        Var const_var("const", Type(nullptr));
        new_field = scope.Push(const_var, new_field);
      }
      new_fields.push_back(new_field);
    }
    return Tuple(new_fields);
  }

  Expr VisitExpr_(const LetNode* ln) final {
    scopes_.emplace_back();

    const LetNode* let = ln;
    Expr body;
    while (let) {
      auto new_value = ExprMutator::Mutate(let->value);
      scopes_.back().Push(let->var, new_value);
      body = let->body;
      let = body.as<LetNode>();
    }

    CHECK(body.defined());
    auto new_body = ExprMutator::Mutate(body);
    auto ret = scopes_.back().Get(new_body);
    scopes_.pop_back();
    return ret;
  }

  Expr VisitExpr_(const CallNode* cn) final {
    if (IsPrimitive(cn)) {
      // Because we are in ANF we do not need to visit the arguments.
      LetList& scope = scopes_.back();
      std::vector<Expr> new_args;
      for (const auto& it : cn->args) {
        new_args.push_back(ExprMutator::Mutate(it));
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
        return DynamicInvoke(&scope, func, ins, new_args, out_types, ret_type);
      } else {
        // Handle the static case
        Array<Expr> outs;
        for (size_t i = 0; i < out_types.size(); ++i) {
          Device dev = GetDevice(GetRef<Call>(cn));
          auto out = MakeStaticAllocation(&scope, out_types[i], dev, std::to_string(i));
          outs.push_back(out);
        }
        Tuple output(outs);
        Expr invoke = InvokeTVMOp(cn->op, ins, output);
        scope.Push(invoke);
        return ToTupleType(ret_type,
                           std::vector<Expr>(output->fields.begin(), output->fields.end()));
      }
    } else {
      return ExprMutator::VisitExpr_(cn);
    }
  }

 private:
  // Insert a device copy node.
  Expr DeviceCopy(const Expr& inp, int src_dev, int dst_dev) {
    return ExprMutator::Mutate(relay::DeviceCopy(inp, src_dev, dst_dev));
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
      return call && call->op == Op::Get("device_copy");
    } else if (const CallNode* cn = expr.as<CallNode>()) {
      return cn->op == Op::Get("device_copy");
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
    Expr shape = MakeConstant(int_shape);
    Expr size = ComputeStorage(type);
    Expr alignment = ComputeAlignment(type->dtype);
    // Run type inference later to get the correct type.
    Var var("storage_" + name_hint, Type(nullptr));
    Expr value = AllocStorage(size, alignment, dev, type->dtype);
    auto sto = scope->Push(var, value);

    // TODO(@jroesch): There is a bug with typing based on the constant shape.
    auto tensor = AllocTensor(sto, shape, type->dtype, type->shape);
    Var tensor_var("tensor_" + name_hint, Type(nullptr));
    return scope->Push(tensor_var, tensor);
  }

  // Insert the shape function given a primitive function.
  Array<Expr> EmitShapeFunc(LetList* scope, const Function& func,
                            const std::vector<Expr>& new_args) {
    Array<Expr> shape_func_ins;
    auto engine = CompileEngine::Global();
    CCacheKey key(func, target_host_);
    auto cfunc = engine->LowerShapeFunc(key);
    auto input_states = cfunc->shape_func_param_states;

    Array<Integer> is_inputs;
    int input_pos = 0;
    Device cpu_dev = default_device_;
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
      if (state == 2) {
        std::vector<Expr> exprs = FromTupleType(ty, arg);
        for (size_t j = 0; j < exprs.size(); ++j) {
          Expr sh_of = ExprMutator::Mutate(ShapeOf(exprs[j]));
          Var in_shape_var("in_shape_" + std::to_string(input_pos + j), Type(nullptr));
          shape_func_ins.push_back(scope->Push(in_shape_var, sh_of));
          input_pos++;
        }
        is_inputs.push_back(0);
      } else if (state == 1) {
        auto new_arg = ExprMutator::Mutate(arg);
        auto dev = GetDevice(arg);
        if (dev.device_type != cpu_dev.device_type) {
          new_arg = DeviceCopy(new_arg, dev.device_type, cpu_dev.device_type);
        }
        Var in_shape_var("in_shape_" + std::to_string(input_pos), Type(nullptr));
        shape_func_ins.push_back(scope->Push(in_shape_var, new_arg));
        input_pos++;
        is_inputs.push_back(1);
      } else {
        // TODO(@jroesch): handle 3rd case
        LOG(FATAL) << "unsupported shape function input state";
      }
    }

    Array<Expr> out_shapes;
    for (size_t i = 0; i < cfunc->outputs.size(); ++i) {
      auto out = cfunc->outputs[i];
      auto tt = TensorType(out->shape, out->dtype);
      // Put shape func on CPU. This also ensures that everything between
      // shape_of and shape_func are on CPU.
      auto alloc = MakeStaticAllocation(scope, tt, cpu_dev, std::to_string(i));
      Var shape_func_out_var("shape_func_out_" + std::to_string(i), Type(nullptr));
      alloc = scope->Push(shape_func_out_var, alloc);
      out_shapes.push_back(alloc);
    }
    auto shape_call = ShapeFunc(func, Tuple(shape_func_ins), Tuple(out_shapes), is_inputs);
    Var shape_func_var("shape_func", Type(nullptr));
    scope->Push(shape_func_var, shape_call);
    return out_shapes;
  }

  // Generate the code for invoking a TVM op with a dynamic shape.
  Expr DynamicInvoke(LetList* scope, const Function& func, const Tuple& ins,
                     const std::vector<Expr>& new_args, const std::vector<TensorType>& out_types,
                     const Type& ret_type) {
    auto out_shapes = EmitShapeFunc(scope, func, new_args);
    std::vector<Var> storages;
    auto func_dev = GetDevice(func);
    CHECK_EQ(out_shapes.size(), out_types.size());
    for (size_t i = 0; i < out_shapes.size(); ++i) {
      auto out_shape = out_shapes[i];
      auto out_type = out_types[i];
      auto size = ComputeStorageInRelay(out_shape, out_type);
      auto alignment = ComputeAlignment(out_type->dtype);
      Var sto_var("storage_" + std::to_string(i), Type(nullptr));
      auto val = AllocStorage(size, alignment, func_dev, out_type->dtype);
      storages.push_back(scope->Push(sto_var, val));
    }

    Array<Expr> outs;
    for (size_t i = 0; i < storages.size(); ++i) {
      auto out_shape = out_shapes[i];
      auto out_type = out_types[i];
      auto storage = storages[i];
      auto alloc = AllocTensor(storage, out_shape, out_type->dtype, out_type->shape);
      Var out_var("out_" + std::to_string(i), Type(nullptr));
      outs.push_back(scope->Push(out_var, alloc));
    }

    Tuple tuple_outs(outs);
    auto invoke = InvokeTVMOp(func, ins, tuple_outs);
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
      shape_expr = MakeConstant(shape);
    }
    return ReshapeTensor(new_args[0], shape_expr, ret_ty->shape);
  }

 private:
  Target target_host_;
  AnalysisResultMap context_analysis_map_;
  std::vector<LetList> scopes_;

  runtime::DataType compute_dtype_ = runtime::DataType::Int(64);
  Device default_device_{kDLCPU, 0};
};

namespace transform {

Pass ManifestAlloc(Target target_host, Map<tvm::Integer, tvm::Target> targets) {
  CheckAndUpdateHostConsistency(&targets, &target_host);
  return tvm::transform::CreateModulePass(
      [=](IRModule mod, const PassContext& pass_ctx) {
        DLOG(INFO) << "tvm::relay::transform::ManifestAlloc";
        // We need to mutate module, therefore making a copy of it.
        mod.CopyOnWrite();
        mod->ImportFromStd("core.rly");
        mod = relay::transform::InferType()(mod);

        Device fallback_dev;
        if (targets.size() > 1) {
          auto pass_ctx = PassContext::Current();
          Optional<Integer> opt_fallback_dev_type =
              pass_ctx->GetConfig("relay.fallback_device_type", Integer(static_cast<int>(kDLCPU)));
          auto fallback_dev_type = opt_fallback_dev_type.value();
          CHECK_GT(fallback_dev_type->value, 0U);
          fallback_dev.device_type = static_cast<DLDeviceType>(fallback_dev_type->value);
          fallback_dev.device_id = 0;
        } else {
          const auto& it = targets.begin();
          fallback_dev.device_type = static_cast<DLDeviceType>((*it).first->value);
          fallback_dev.device_id = 0;
        }
        auto ca = ContextAnalysis(mod, fallback_dev);

        auto glob_funcs = mod->functions;
        for (const auto& it : glob_funcs) {
          if (auto* func_node = it.second.as<FunctionNode>()) {
            auto func = GetRef<Function>(func_node);
            auto rewriter = DialectRewriter(target_host, ca);
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
