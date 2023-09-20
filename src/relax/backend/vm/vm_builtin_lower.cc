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
#include <tvm/relax/attrs/op.h>
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

  Expr VisitExpr_(const CallNode* call_node) final {
    // post-order mutation
    Call call = Downcast<Call>(VisitExprPostOrder_(call_node));

    if (call->op == call_tir_dyn_op_) {
      return CallTIRDyn(call);
    } else if (call->op == reshape_op_) {
      return Reshape(call);
    } else if (call->op == shape_of_op_) {
      return ShapeOf(call);
    } else if (call->op == to_vdevice_op_) {
      return ToDevice(call);
    } else if (call->op == make_closure_op_) {
      return MakeClosure(call);
    } else if (call->op == invoke_closure_op_) {
      return InvokeClosure(call);
    } else if (call->op == alloc_tensor_op_) {
      LOG(FATAL) << "VMBuiltinLower encountered " << call->op << " in expression "
                 << GetRef<Call>(call_node) << ".  "
                 << "This operation should have been lowered earlier "
                 << "using the 'relax.transform.LowerAllocTensor' pass.";
    } else if (call->op == mem_alloc_storage_op_) {
      return MakeMemAllocStorage(call);
    } else if (call->op == mem_alloc_tensor_op_) {
      return MakeMemAllocTensor(call);
    } else if (call->op == mem_kill_storage_op_ || call->op == mem_kill_tensor_op_) {
      return MakeMemKillObject(call);
    } else {
      return call;
    }
  }

  Expr MakeMemAllocStorage(const Call& call) {
    PrimValue runtime_device_index = Downcast<PrimValue>(call->args[1]);
    StringImm storage_scope = Downcast<StringImm>(call->args[2]);
    DataTypeImm output_dtype = DataTypeImm(DataType::UInt(8));
    return Call(vm_alloc_storage_op_,
                {call->args[0], runtime_device_index, output_dtype, storage_scope}, Attrs());
  }

  Expr MakeMemAllocTensor(const Call& call) {
    PrimValue offset = Downcast<PrimValue>(call->args[1]);
    DataTypeImm dtype = Downcast<DataTypeImm>(call->args[3]);
    return Call(vm_alloc_tensor_op_, {call->args[0], offset, call->args[2], dtype}, Attrs());
  }

  Expr MakeMemKillObject(const Call& call) {
    ICHECK_EQ(call->args.size(), 1);
    return Call(vm_kill_object_op_, {call->args[0]}, Attrs());
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
    auto arg = call_node->args[1];
    CHECK(arg->IsInstance<ShapeExprNode>() || arg->IsInstance<VarNode>())
        << "VMBuiltinLower expects the shape arg of reshape op to be a ShapeExpr or VarNode bound "
           "to a ShapeExpr";

    if (arg->IsInstance<ShapeExprNode>()) {
      return Call(builtin_reshape_, call_node->args, Attrs(), {GetStructInfo(call_node)});
    } else {
      // Handling the case when arg is VarNode
      Optional<Expr> _bound_val = LookupBinding(Downcast<Var>(arg));
      ICHECK(_bound_val.defined());
      Expr bound_val = _bound_val.value();
      CHECK(bound_val->IsInstance<ShapeExprNode>())
          << "VMBuiltinLower expects bound value to be a ShapeExpr";
      return Call(builtin_reshape_, {call_node->args[0], bound_val}, Attrs(),
                  {GetStructInfo(call_node)});
    }
  }

  Expr ShapeOf(const Call& call_node) {
    ICHECK(call_node->args.size() == 1);
    ICHECK(call_node->struct_info_.defined());
    return Call(builtin_shape_of_, call_node->args, Attrs(), {GetStructInfo(call_node)});
  }

  Expr ToDevice(const Call& call_node) {
    // TODO(yongwww): replace ToVDeviceAttrs with related Expr
    ICHECK(call_node->args.size() == 1);
    ICHECK(call_node->struct_info_.defined());
    auto attrs = call_node->attrs.as<ToVDeviceAttrs>();
    Array<Expr> args;
    args.push_back(call_node->args[0]);
    // Get the DLDeviceType and device_id from VDevice
    VDevice vdev = attrs->dst_vdevice;
    int dev_type = vdev->target->GetTargetDeviceType();
    int dev_id = vdev->vdevice_id;
    args.push_back(PrimValue::Int64(dev_type));
    args.push_back(PrimValue::Int64(dev_id));
    return Call(builtin_to_device_, args, call_node->attrs, {GetStructInfo(call_node)});
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
  const Op& shape_of_op_ = Op::Get("relax.shape_of");
  const Op& to_vdevice_op_ = Op::Get("relax.to_vdevice");
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
  const Op& vm_kill_object_op_ = Op::Get("relax.vm.kill_object");
  // Function to compute allocated shape.
  const ExternFunc builtin_compute_alloc_shape_{"vm.builtin.compute_alloc_shape"};
  const ExternFunc builtin_call_tir_dyn_{"vm.builtin.call_tir_dyn"};
  const ExternFunc builtin_reshape_{"vm.builtin.reshape"};
  const ExternFunc builtin_shape_of_{"vm.builtin.shape_of"};
  const ExternFunc builtin_to_device_{"vm.builtin.to_device"};
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
