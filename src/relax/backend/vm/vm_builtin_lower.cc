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
 * \file src/relax/backend/vm/vm_builtin_lower.cc
 * \brief Lowers most builtin functions and packed calls.
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/backend.h>
#include <tvm/relax/expr_functor.h>
#include <tvm/relax/type.h>
#include <tvm/runtime/data_type.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace relax {

// This pass lowers most ops to VM specific builtins.
// TODO(relax-team): revisit after PrimValue.
class VMBuiltinLowerMutator : public ExprMutator {
 public:
  using ExprMutator::VisitExpr_;

  // A workaround to remove the CallNodes of killing tensors and storages.
  void VisitBinding_(const VarBindingNode* binding) final {
    const auto* call = binding->value.as<CallNode>();
    if (call != nullptr && (call->op == mem_kill_storage_op_ || call->op == mem_kill_tensor_op_)) {
      return;
    }
    ExprMutator::VisitBinding_(binding);
  }

  Expr VisitExpr_(const CallNode* call_node) final {
    // post-order mutation
    Call call = Downcast<Call>(VisitExprPostOrder_(call_node));

    if (call->op == call_tir_dyn_op_) {
      return CallTIRDyn(call);
    } else if (call->op == reshape_op_) {
      return Reshape(call);
    } else if (call->op == make_closure_op_) {
      return MakeClosure(call);
    } else if (call->op == invoke_closure_op_) {
      return InvokeClosure(call);
    } else if (call->op == alloc_tensor_op_) {
      return MakeAllocTensor(call);
    } else if (call->op == mem_alloc_storage_op_) {
      return MakeMemAllocStorage(call);
    } else if (call->op == mem_alloc_tensor_op_) {
      return MakeMemAllocTensor(call);
    } else {
      return call;
    }
  }

  Expr ComputeStorageSize(const Expr& shape, const DataType& dtype) const {
    // Question: what if the dtype of tensor_type is unknown?
    // Symbolic/static shape case
    if (auto* shape_expr = shape.as<ShapeExprNode>()) {
      int64_t elem_bytes = runtime::GetVectorBytes(dtype);
      PrimExpr ret = IntImm(DataType::Int(64), elem_bytes);
      for (PrimExpr dim : shape_expr->values) {
        ret = ret * dim;
      }
      return ShapeExpr({ret});
    } else {
      return Call(builtin_compute_alloc_shape_, {shape, DataTypeImm(dtype)}, Attrs(),
                  {GetStructInfo(shape)});
    }
  }

  Expr MakeAllocTensor(const Call& call) {
    ShapeExpr output_shape = Downcast<ShapeExpr>(call->args[0]);
    DataTypeImm output_dtype = Downcast<DataTypeImm>(call->args[1]);
    DataType dtype = output_dtype->value;
    Expr storage_size = ComputeStorageSize(output_shape, dtype);
    PrimValue runtime_device_index = Downcast<PrimValue>(call->args[2]);
    Var storage = builder_->Emit(
        Call(vm_alloc_storage_op_, {storage_size, runtime_device_index, output_dtype}, Attrs()),
        "storage");
    Expr shape = call->args[0];
    PrimValue offset = PrimValue::Int64(0);
    return Call(vm_alloc_tensor_op_, {storage, offset, shape, DataTypeImm(dtype)}, Attrs());
  }

  Expr MakeMemAllocStorage(const Call& call) {
    PrimValue runtime_device_index = Downcast<PrimValue>(call->args[1]);
    DataTypeImm output_dtype = Downcast<DataTypeImm>(call->args[3]);
    return Call(vm_alloc_storage_op_, {call->args[0], runtime_device_index, output_dtype}, Attrs());
  }

  Expr MakeMemAllocTensor(const Call& call) {
    PrimValue offset = Downcast<PrimValue>(call->args[1]);
    DataTypeImm dtype = Downcast<DataTypeImm>(call->args[3]);
    return Call(vm_alloc_tensor_op_, {call->args[0], offset, call->args[2], dtype}, Attrs());
  }

  Expr CallTIRDyn(const Call& call_node) {
    ICHECK(call_node->args.size() == 2);
    ICHECK(call_node->args[0]->IsInstance<GlobalVarNode>());
    ICHECK(call_node->args[1]->IsInstance<TupleNode>());
    Array<Expr> args;

    auto tir_args = Downcast<Tuple>(call_node->args[1]);
    args.push_back(call_node->args[0]);
    for (Expr arg : tir_args->fields) {
      args.push_back(arg);
    }
    return Call(builtin_call_tir_dyn_, args, Attrs(), {void_sinfo_});
  }

  Expr Reshape(const Call& call_node) {
    ICHECK(call_node->args.size() == 2);
    ICHECK(call_node->struct_info_.defined());
    CHECK(call_node->args[1]->IsInstance<ShapeExprNode>())
        << "VMBuiltinLower expects the shape arg of reshape op to be a ShapeExpr";
    return Call(builtin_reshape_, call_node->args, Attrs(), {GetStructInfo(call_node)});
  }

  Expr MakeClosure(const Call& call_node) {
    ICHECK(call_node->args.size() == 2);
    ICHECK(call_node->args[0]->IsInstance<GlobalVarNode>());
    ICHECK(call_node->args[1]->IsInstance<TupleNode>());

    Array<Expr> args;
    auto func = call_node->args[0];
    auto closure_args = Downcast<Tuple>(call_node->args[1]);

    args.push_back(func);
    for (Expr arg : closure_args->fields) {
      args.push_back(arg);
    }

    return Call(builtin_make_closure_, args, Attrs(), {object_sinfo_});
  }

  Expr InvokeClosure(const Call& call_node) {
    ICHECK(call_node->args.size() == 2);
    ICHECK(call_node->args[0]->IsInstance<VarNode>());
    ICHECK(call_node->args[1]->IsInstance<TupleNode>());

    Array<Expr> args;

    args.push_back(call_node->args[0]);

    // args for the invoke_closure
    auto invoke_closure_args = Downcast<Tuple>(call_node->args[1]);
    for (Expr arg : invoke_closure_args->fields) {
      args.push_back(arg);
    }
    return Call(call_builtin_with_ctx_op_, {builtin_invoke_closure_, Tuple(args)}, Attrs(),
                {object_sinfo_});
  }

  const Op& call_builtin_with_ctx_op_ = Op::Get("relax.call_builtin_with_ctx");
  const StructInfo object_sinfo_ = ObjectStructInfo();
  const StructInfo void_sinfo_ = TupleStructInfo(Array<StructInfo>({}));
  // object to pattern match.
  const Op& call_tir_dyn_op_ = Op::Get("relax.vm.call_tir_dyn");
  const Op& reshape_op_ = Op::Get("relax.reshape");
  const Op& make_closure_op_ = Op::Get("relax.make_closure");
  const Op& invoke_closure_op_ = Op::Get("relax.invoke_closure");
  const Op& alloc_tensor_op_ = Op::Get("relax.builtin.alloc_tensor");
  const Op& mem_alloc_storage_op_ = Op::Get("relax.memory.alloc_storage");
  const Op& mem_alloc_tensor_op_ = Op::Get("relax.memory.alloc_tensor");
  const Op& mem_kill_storage_op_ = Op::Get("relax.memory.kill_storage");
  const Op& mem_kill_tensor_op_ = Op::Get("relax.memory.kill_tensor");
  // functions to lower to
  const Op& vm_alloc_storage_op_ = Op::Get("relax.vm.alloc_storage");
  const Op& vm_alloc_tensor_op_ = Op::Get("relax.vm.alloc_tensor");
  // Function to compute allocated shape.
  const ExternFunc builtin_compute_alloc_shape_{"vm.builtin.compute_alloc_shape"};
  const ExternFunc builtin_call_tir_dyn_{"vm.builtin.call_tir_dyn"};
  const ExternFunc builtin_reshape_{"vm.builtin.reshape"};
  const ExternFunc builtin_make_closure_{"vm.builtin.make_closure"};
  const ExternFunc builtin_invoke_closure_{"vm.builtin.invoke_closure"};
};

Expr VMBuiltinLower(const Expr& e) { return VMBuiltinLowerMutator().VisitExpr(e); }

namespace transform {

Pass VMBuiltinLower() {
  runtime::TypedPackedFunc<Function(Function, IRModule, PassContext)> pass_func =
      [=](Function f, IRModule m, PassContext pc) { return Downcast<Function>(VMBuiltinLower(f)); };
  return CreateFunctionPass(pass_func, 0, "VMBuiltinLower", {});
}

TVM_REGISTER_GLOBAL("relax.transform.VMBuiltinLower").set_body_typed(VMBuiltinLower);

}  // namespace transform
}  // namespace relax
}  // namespace tvm
