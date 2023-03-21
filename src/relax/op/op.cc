/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include <tvm/relax/analysis.h>
#include <tvm/relax/expr.h>
#include <tvm/relax/utils.h>
#include <tvm/relay/op.h>

#include "op_common.h"

namespace tvm {
namespace relax {

bool EqualConstInt(const PrimExpr& lhs, int64_t value) {
  if (const int64_t* pvalue = tir::as_const_int(lhs)) {
    return pvalue[0] == value;
  }
  return false;
}

bool EqualCheck(const PrimExpr& lhs, const PrimExpr& rhs) {
  PrimExpr diff = lhs - rhs;
  if (const int64_t* pdiff = tir::as_const_int(diff)) {
    return pdiff[0] == 0;
  }
  tvm::arith::Analyzer ana;
  diff = ana.Simplify(diff);
  if (const int64_t* pdiff = tir::as_const_int(diff)) {
    return pdiff[0] == 0;
  }
  return false;
}

StructInfo ReturnVoidStructInfo(const Call& call, const BlockBuilder& ctx) {
  return TupleStructInfo(Array<StructInfo>());
}

StructInfo ReturnObjectStructInfo(const Call& call, const BlockBuilder& ctx) {
  return ObjectStructInfo();
}

StructInfo InferStructInfoShapeOf(const Call& call, const BlockBuilder& ctx) {
  // use the StructInfo of the argument
  auto arg_sinfo = GetStructInfo(call->args[0]);
  auto* tensor_sinfo = GetStructInfo(call->args[0]).as<TensorStructInfoNode>();
  CHECK(tensor_sinfo) << "shape_of expects a tensor input, but received " << arg_sinfo
                      << "; use MatchCast if necessary";
  if (tensor_sinfo->ndim == kUnknownNDim) {
    return ShapeStructInfo(kUnknownNDim);
  }
  // if the tensor shape is a Relax var or omitted, do not try to construct a shape expr from it
  if (!tensor_sinfo->shape.defined() || tensor_sinfo->shape.as<VarNode>()) {
    return ShapeStructInfo(tensor_sinfo->ndim);
  }
  // otherwise, copy over the values from the tensor shape
  auto* tensor_shape = tensor_sinfo->shape.as<ShapeExprNode>();
  CHECK(tensor_shape);
  return ShapeStructInfo(tensor_shape->values);
}

// call_tir

StructInfo InferStructInfoCallTIR(const Call& call, const BlockBuilder& ctx) {
  if (call->sinfo_args.size() != 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "sinfo_args should have exact 1 output struct info.");
  }
  CHECK(call->args[0]->IsInstance<GlobalVarNode>())
      << "call_tir expects the first argument to be a GlobalVar referring to a TIR PrimFunc. "
      << "However, gets " << call->args[0];
  return call->sinfo_args[0];
}

RELAY_REGISTER_OP("relax.call_tir")
    .set_num_inputs(3)
    .add_argument("func", "Expr", "The destination-passing-style function.")
    .add_argument("args", "Tuple", "The input arguments.")
    .add_argument("packed_ints", "Expr",
                  "ShapeExpr representing a tuple of ints to unpack during runtime. Omitted from "
                  "args if unused")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoCallTIR);

Expr MakeCallTIR(Expr func, Tuple args, Array<TensorStructInfo> out_sinfo_list,
                 Optional<Expr> packed_ints) {
  for (const TensorStructInfo& sinfo : out_sinfo_list) {
    const auto* shape = sinfo->shape.as<ShapeExprNode>();
    CHECK(shape != nullptr) << "out_sinfo of call_tir should have defined ShapeExpr as shape. "
                               "However, one given structure info is "
                            << sinfo;
  }

  StructInfo out_sinfo{nullptr};
  if (out_sinfo_list.size() == 1) {
    out_sinfo = out_sinfo_list[0];
  } else {
    out_sinfo = TupleStructInfo({out_sinfo_list.begin(), out_sinfo_list.end()});
  }

  static const Op& op = Op::Get("relax.call_tir");
  Call call;
  if (!packed_ints) {
    // don't use additional optional argument
    call = Call(op, {func, args}, {}, {out_sinfo});
  } else {
    call = Call(op, {func, args, packed_ints.value()}, {}, {out_sinfo});
  }
  return call;
}

TVM_REGISTER_GLOBAL("relax.op.call_tir").set_body_typed(MakeCallTIR);

// call_dps_packed

StructInfo InferStructInfoCallDPSPacked(const Call& call, const BlockBuilder& ctx) {
  if (call->sinfo_args.size() != 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "sinfo_args should have exact 1 output struct info.");
  }
  return call->sinfo_args[0];
}

RELAY_REGISTER_OP("relax.call_dps_packed")
    .set_num_inputs(2)
    .add_argument("func", "Expr", "The destination-passing-style function.")
    .add_argument("args", "Tuple", "The input arguments.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoCallDPSPacked);

Expr MakeCallDPSPacked(Expr func, Tuple args, Array<TensorStructInfo> out_sinfo_list) {
  for (const TensorStructInfo& sinfo : out_sinfo_list) {
    const auto* shape = sinfo->shape.as<ShapeExprNode>();
    CHECK(shape != nullptr)
        << "out_sinfo of call_dps_packed should have defined ShapeExpr as shape. "
           "However, one given structure info is "
        << sinfo;
  }

  StructInfo out_sinfo{nullptr};
  if (out_sinfo_list.size() == 1) {
    out_sinfo = out_sinfo_list[0];
  } else {
    out_sinfo = TupleStructInfo({out_sinfo_list.begin(), out_sinfo_list.end()});
  }

  static const Op& op = Op::Get("relax.call_dps_packed");
  return Call(op, {func, args}, {}, {out_sinfo});
}

TVM_REGISTER_GLOBAL("relax.op.call_dps_packed").set_body_typed(MakeCallDPSPacked);

// call builtin
StructInfo InferStructInfoCallBuiltinWithCtx(const Call& call, const BlockBuilder& ctx) {
  if (call->sinfo_args.size() == 0) {
    // by default return void.
    return TupleStructInfo(Array<StructInfo>());
  } else {
    ICHECK_EQ(call->sinfo_args.size(), 1);
    return call->sinfo_args[0];
  }
}

TVM_REGISTER_OP("relax.call_builtin_with_ctx")
    .set_num_inputs(4)
    .add_argument("func", "Expr", "The builtin packed func.")
    .add_argument("args", "Tuple", "The input arguments.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoCallBuiltinWithCtx);

Expr MakeCallBuiltinWithCtx(Expr func, Tuple args, Array<StructInfo> sinfo_args) {
  static const Op& op = Op::Get("relax.call_builtin_with_ctx");
  return Call(op, {func, args}, Attrs(), sinfo_args);
}

TVM_REGISTER_GLOBAL("relax.op.call_builtin_with_ctx").set_body_typed(MakeCallBuiltinWithCtx);

TVM_REGISTER_OP("relax.null_value")
    .set_num_inputs(0)
    .set_attr<FInferStructInfo>("FInferStructInfo", ReturnObjectStructInfo);

Expr MakeCallNullValue() {
  static const Op& op = Op::Get("relax.null_value");
  return Call(op, {}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.null_value").set_body_typed(MakeCallNullValue);

// print

RELAY_REGISTER_OP("relax.print")
    .set_num_inputs(-1)
    .add_argument("vals", "Array<Expr>",
                  "The first value is Python-style format string to use to print. The others "
                  "are values to print")
    .set_attr<FInferStructInfo>("FInferStructInfo", ReturnVoidStructInfo)
    .set_attr<FCallPacked>("FCallPacked", "relax.run.print");

Expr MakePrint(Array<Expr> vals, StringImm format) {
  Array<Expr> params;
  params.push_back(format);
  for (const auto val : vals) {
    params.push_back(val);
  }
  static const Op& op = Op::Get("relax.print");
  return Call(op, params);
}

TVM_REGISTER_GLOBAL("relax.op.print").set_body_typed(MakePrint);

// assert_op

// can't actually name it assert or else Python will consider it a syntax error

StructInfo InferAssertStructInfo(const Call& call, const BlockBuilder& ctx) {
  // Ensure that the condition argument is a boolean scalar.
  // Also permitted is a tensor with unknown shape and unknown dtype
  // (checked dynamically in that case). Returns void.
  if (call->args.size() < 1) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Assert must have at least one argument (the condition).");
  }
  StructInfo arg_struct_info = GetStructInfo(call->args[0]);
  if (!IsBoolStructInfo(arg_struct_info)) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "The argument to assert must be a boolean scalar, but received "
                     << arg_struct_info);
  }
  return ReturnVoidStructInfo(call, ctx);
}

RELAY_REGISTER_OP("relax.assert_op")
    .set_num_inputs(-1)
    .add_argument("vals", "Array<Expr>",
                  "The first value is used as the assertion condition. The second value is "
                  "Python-style format string to use for displaying an error message, if the "
                  "assert fails. The others are used as format arguments if there is an error.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferAssertStructInfo)
    .set_attr<FCallPacked>("FCallPacked", "relax.run.assert_op");

Expr MakeAssertOp(Expr condition, Array<Expr> vals, StringImm format) {
  static const Op& op = Op::Get("relax.assert_op");
  Array<Expr> args = {condition};
  args.push_back(format);
  for (auto val : vals) {
    args.push_back(val);
  }
  return Call(op, args);
}

TVM_REGISTER_GLOBAL("relax.op.assert_op").set_body_typed(MakeAssertOp);

// make_closure

RELAY_REGISTER_OP("relax.make_closure")
    .set_num_inputs(2)
    .add_argument("func", "Expr", "The closure.")
    .add_argument("args", "Tuple", "The captured variables.")
    .set_attr<FInferStructInfo>("FInferStructInfo", ReturnObjectStructInfo);

Expr MakeClosure(Expr func, Tuple args) {
  static const Op& op = Op::Get("relax.make_closure");
  return Call(op, {func, args}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.make_closure").set_body_typed(MakeClosure);

// invoke_closure

StructInfo InferStructInfoInvokeClosure(const Call& call, const BlockBuilder& ctx) {
  if (call->sinfo_args.empty()) {
    return ObjectStructInfo();
  } else if (call->sinfo_args.size() == 1) {
    return call->sinfo_args[0];
  } else {
    return TupleStructInfo(call->sinfo_args);
  }
}

RELAY_REGISTER_OP("relax.invoke_closure")
    .set_num_inputs(2)
    .add_argument("closure", "Expr", "The VMClosure.")
    .add_argument("args", "Tuple", "The captured variables.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoInvokeClosure);

Expr InvokeClosure(Expr closure, Tuple args, Array<StructInfo> sinfo_args) {
  static const Op& op = Op::Get("relax.invoke_closure");
  return Call(op, {closure, args}, {}, sinfo_args);
}

TVM_REGISTER_GLOBAL("relax.op.invoke_closure").set_body_typed(InvokeClosure);

// shape_of

RELAY_REGISTER_OP("relax.shape_of")
    .set_num_inputs(1)
    .add_argument("input", "Expr", "The input expression")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoShapeOf);

Expr MakeShapeOf(Expr expr) {
  static const Op& op = Op::Get("relax.shape_of");
  return Call(op, {expr}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.shape_of").set_body_typed(MakeShapeOf);

// tensor_to_shape

StructInfo ReturnTensorToShapeStructInfo(const Call& call, const BlockBuilder& ctx) {
  ICHECK(call->args.size() == 1);
  ICHECK(call->args[0]->struct_info_.defined());
  const auto* tsinfo = GetStructInfoAs<TensorStructInfoNode>(call->args[0]);
  ICHECK(tsinfo && tsinfo->shape.defined());
  ShapeExpr shape_expr = Downcast<ShapeExpr>(tsinfo->shape.value());
  ICHECK(shape_expr->values.size() == 1);
  const IntImmNode* ndim = shape_expr->values[0].as<IntImmNode>();
  ICHECK(ndim);
  return ShapeStructInfo(ndim->value);
}

RELAY_REGISTER_OP("relax.tensor_to_shape")
    .set_num_inputs(1)
    .add_argument("input", "Expr", "The input expression")
    .set_attr<FInferStructInfo>("FInferStructInfo", ReturnTensorToShapeStructInfo);

Expr MakeTensorToShape(Expr expr) {
  static const Op& op = Op::Get("relax.tensor_to_shape");
  return Call(op, {expr}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.tensor_to_shape").set_body_typed(MakeTensorToShape);

// alloc_tensor

StructInfo InferStructInfoAllocateTensor(const Call& call, const BlockBuilder& ctx) {
  ICHECK(call->args[0].as<ShapeExprNode>())
      << "must be ShapeExpr, but got " << call->args[0]->GetTypeKey();
  ICHECK(call->args[1].as<DataTypeImmNode>())
      << "must be DataTypeImm, but got " << call->args[1]->GetTypeKey();
  DataType out_dtype;
  if (const auto* dtype_node = call->args[1].as<DataTypeImmNode>()) {
    const DataTypeImm dtype_imm = GetRef<DataTypeImm>(dtype_node);
    out_dtype = dtype_imm->value;
  }
  return TensorStructInfo(call->args[0], out_dtype);
}

RELAY_REGISTER_OP("relax.builtin.alloc_tensor")
    .set_num_inputs(3)
    .add_argument("shape", "Expr", "The shape of the tensor to allocate.")
    .add_argument("dtype", "DataTypeImm", "The dtype of the tensor to allocate.")
    .add_argument("runtime_device_index", "PrimValue",
                  "The device index indicating on which device the tensor is to be "
                  "allocated at runtime. Index -1 is reserved for the host device.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoAllocateTensor);

Expr MakeAllocTensor(Expr shape, DataTypeImm dtype, PrimValue runtime_device_index) {
  static const Op& op = Op::Get("relax.builtin.alloc_tensor");
  return Call(op, {shape, dtype, runtime_device_index}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.builtin.alloc_tensor").set_body_typed(MakeAllocTensor);

// memory planning alloc_storage

RELAY_REGISTER_OP("relax.memory.alloc_storage")
    .set_num_inputs(4)
    .add_argument("total_space", "Expr", "The total space of the storage to allocate.")
    .add_argument(
        "virtual_device_index", "PrimValue",
        "The virtual device index indicating on which device the storage is to be allocated, "
        "Index -1 is reserved for the host device.")
    .add_argument("storage_scope", "StringImm",
                  "The storage scope of the storage to allocate. Default is global.")
    .add_argument("dtype", "DataTypeImm", "The dtype of the tensor to allocate.")
    .set_attr<FInferStructInfo>("FInferStructInfo", ReturnObjectStructInfo);

Expr MakeAllocStorage(Expr size, PrimValue virtual_device_index, StringImm storage_scope,
                      DataTypeImm dtype) {
  static const Op& op = Op::Get("relax.memory.alloc_storage");
  return Call(op, {size, virtual_device_index, storage_scope, dtype}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.memory.alloc_storage").set_body_typed(MakeAllocStorage);

// memory planning alloc_tensor

StructInfo InferStructInfoMemAllocTensor(const Call& call, const BlockBuilder& ctx) {
  ICHECK(GetStructInfoAs<ShapeStructInfoNode>(call->args[2]))
      << "must be a Expr of ShapeStructInfo, but got " << call->args[1]->GetTypeKey();
  DataType out_dtype;
  if (const auto* dtype_node = call->args[3].as<DataTypeImmNode>()) {
    const DataTypeImm dtype_imm = GetRef<DataTypeImm>(dtype_node);
    out_dtype = dtype_imm->value;
  }
  return TensorStructInfo(call->args[2], out_dtype);
}

RELAY_REGISTER_OP("relax.memory.alloc_tensor")
    .set_num_inputs(4)
    .add_argument("storage", "Expr", "The storage to allocate the tensor to.")
    .add_argument("offset", "PrimValue", "Storage offset to allocate the tensor.")
    .add_argument("shape", "Expr", "The shape of the tensor to allocate.")
    .add_argument("dtype", "DataTypeImm", "The dtype of the tensor to allocate.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoMemAllocTensor);

Expr MakeMemAllocTensor(Expr storage, PrimValue offset, Expr shape, DataTypeImm dtype) {
  static const Op& op = Op::Get("relax.memory.alloc_tensor");
  return Call(op, {storage, offset, shape, dtype}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.memory.alloc_tensor").set_body_typed(MakeMemAllocTensor);

// memory planning kill_storage

RELAY_REGISTER_OP("relax.memory.kill_storage")
    .set_num_inputs(1)
    .add_argument("storage", "Expr", "The storage to be killed.")
    .set_attr<FInferStructInfo>("FInferStructInfo", ReturnVoidStructInfo);

Expr MakeMemKillStorage(Expr storage) {
  static const Op& op = Op::Get("relax.memory.kill_storage");
  return Call(op, {storage}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.memory.kill_storage").set_body_typed(MakeMemKillStorage);

// memory planning kill_tensor

RELAY_REGISTER_OP("relax.memory.kill_tensor")
    .set_num_inputs(1)
    .add_argument("tensor", "Expr", "The tensor to be killed.")
    .set_attr<FInferStructInfo>("FInferStructInfo", ReturnVoidStructInfo);

Expr MakeMemKillTensor(Expr tensor) {
  static const Op& op = Op::Get("relax.memory.kill_tensor");
  return Call(op, {tensor}, {}, {});
}

TVM_REGISTER_GLOBAL("relax.op.memory.kill_tensor").set_body_typed(MakeMemKillTensor);

// vm alloc_storage

RELAY_REGISTER_OP("relax.vm.alloc_storage")
    .set_num_inputs(3)
    .add_argument("size", "Expr", "The size of the storage to allocate.")
    .add_argument("dtype", "DataTypeImm", "The dtype of the tensor to allocate.")
    .add_argument("runtime_device_index", "PrimValue",
                  "The device index indicating on which device the tensor is "
                  "to be allocated at runtime.")
    .set_attr<FInferStructInfo>("FInferStructInfo", ReturnObjectStructInfo);

Expr MakeVMAllocStorage(Expr size, PrimValue runtime_device_index, DataTypeImm dtype) {
  static const Op& op = Op::Get("relax.vm.alloc_storage");
  return Call(op, {size, runtime_device_index, dtype}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.vm.alloc_storage").set_body_typed(MakeVMAllocStorage);

// vm alloc_tensor

StructInfo InferStructInfoVMAllocTensor(const Call& call, const BlockBuilder& ctx) {
  DataType out_dtype;
  if (const auto* dtype_node = call->args[3].as<DataTypeImmNode>()) {
    const DataTypeImm dtype_imm = GetRef<DataTypeImm>(dtype_node);
    out_dtype = dtype_imm->value;
  }
  if (const auto* output_shape = call->args[2].as<ShapeExprNode>()) {
    return TensorStructInfo(GetRef<Expr>(output_shape), out_dtype);
  }
  return TensorStructInfo(out_dtype, kUnknownNDim);
}

RELAY_REGISTER_OP("relax.vm.alloc_tensor")
    .set_num_inputs(4)
    .add_argument("storage", "Expr", "The storage to allocate the tensor to.")
    .add_argument("offset", "PrimValue", "Storage offset to allocate the tensor.")
    .add_argument("shape", "Expr", "The shape of the tensor to allocate.")
    .add_argument("dtype", "DataTypeImm", "The dtype of the tensor to allocate.")
    .set_attr<FInferStructInfo>("FInferStructInfo", InferStructInfoVMAllocTensor);

Expr MakeVMAllocTensor(Expr storage, PrimValue offset, Expr shape, DataTypeImm dtype) {
  static const Op& op = Op::Get("relax.vm.alloc_tensor");
  return Call(op, {storage, offset, shape, dtype}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.vm.alloc_tensor").set_body_typed(MakeVMAllocTensor);

// vm call_tir_dyn

RELAY_REGISTER_OP("relax.vm.call_tir_dyn")
    .set_num_inputs(2)
    .add_argument("func", "Expr", "The destination-passing-style function.")
    .add_argument("args", "Tuple",
                  "The input arguments (list of tensors and last argument is ShapeExpr)")
    .set_attr<FInferStructInfo>("FInferStructInfo", ReturnVoidStructInfo);

Expr MakeCallTIRDyn(Expr func, Tuple args) {
  static const Op& op = Op::Get("relax.vm.call_tir_dyn");
  return Call(op, {func, args}, Attrs(), {});
}

TVM_REGISTER_GLOBAL("relax.op.vm.call_tir_dyn").set_body_typed(MakeCallTIRDyn);

}  // namespace relax
}  // namespace tvm
