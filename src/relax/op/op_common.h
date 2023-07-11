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
 * \file op_common.h
 * \brief A set of utilities and common functionality
 * for Relax ops.
 */
#ifndef TVM_RELAX_OP_OP_COMMON_H_
#define TVM_RELAX_OP_OP_COMMON_H_

#include <tvm/arith/analyzer.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/tir/data_layout.h>

#include <utility>
#include <vector>

#include "../transform/infer_amp_utils.h"
#include "../transform/infer_layout_utils.h"

namespace tvm {
namespace relax {

/************ Op input struct info getter ************/

/*!
 * \brief Get the tensor struct info of the operator input.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \return The tensor struct info of each input.
 * \note This function require every input to be Tensor. The number of call arguments is required
 * to match the number of inputs of the op being called.
 */
Array<TensorStructInfo> GetInputTensorStructInfo(const Call& call, const BlockBuilder& ctx);

/*!
 * \brief Get the tensor struct info of the unary operator input.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \return The tensor struct info of the unary operator input.
 * \throw Throw exception if the number of input is not one, or the struct info of the input is not
 * a tensor struct info.
 */
inline TensorStructInfo GetUnaryInputTensorStructInfo(const Call& call, const BlockBuilder& ctx) {
  return GetInputTensorStructInfo(call, ctx)[0];
}

/*!
 * \brief Get the tensor struct info of tuple input.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param tup The input tuple.
 * \return The tensor struct infos of tuple input.
 * \throw Throw exception if input expression is not a tuple.
 */
Array<TensorStructInfo> GetTensorStructInfoFromTuple(const Call& call, const BlockBuilder& ctx,
                                                     const Expr& tup);

/************ Op registration macro ************/

/*!
 * \brief Quick helper macro to register the operator to registry
 * \param OpRegName The name of operator to register. The name passed in will
 * be prepended with a prefix "relax." as the identifier string in the operator registry.
 */
#define RELAX_REGISTER_UNARY_OP(OpRegName)                                                         \
  TVM_REGISTER_OP("relax." OpRegName)                                                              \
      .set_num_inputs(1)                                                                           \
      .add_argument("x", "Tensor", "The input tensor.")                                            \
      .set_attr<FRelaxInferLayout>("FRelaxInferLayout", InferLayoutUnaryEwise)                     \
      .set_attr<TMixedPrecisionPolicy>("TMixedPrecisionPolicy", MixedPrecisionPolicyKind::kFollow) \
      .set_attr<Bool>("FPurity", Bool(true))

/*!
 * \brief Quick helper macro to expose a make-function to construct the operator.
 * \param OpName The name of the operator as well as the make-function name, which will
 * be prepended with a prefix "relax.op." as the FFI identifier string for the make function,
 * \param OpRegName The identifier of the operator in the registry.
 */
#define RELAX_UNARY_OP_INTERFACE(OpName, OpRegName)    \
  Expr OpName(Expr x) {                                \
    static const Op& op = Op::Get("relax." OpRegName); \
    return Call(op, {std::move(x)}, Attrs(), {});      \
  }                                                    \
  TVM_REGISTER_GLOBAL("relax.op." OpRegName).set_body_typed(OpName)

/************ Utilities ************/

/*!
 * \brief Infer the struct info for unary elementwise ops.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param f_compute_out_dtype The function to compute the output dtype, with
 * signature DataType f_compute_out_dtype(const TensorStructInfo& input_sinfo).
 * \tparam require_float_dtype whether this op requires the input dtype to be float
 * \tparam Ftype the type of f_compute_out_dtype
 * \return The inferred struct info.
 */
template <bool require_float_dtype, typename FType>
inline StructInfo InferStructInfoUnary(const Call& call, const BlockBuilder& ctx,
                                       FType f_compute_out_dtype) {
  TensorStructInfo input_sinfo = GetUnaryInputTensorStructInfo(call, ctx);
  if (require_float_dtype && !input_sinfo->IsUnknownDtype() && !input_sinfo->dtype.is_float()) {
    ctx->ReportFatal(
        Diagnostic::Error(call)
        << call->op
        << " requires the input tensor to have float dtype. However, the given input dtype is "
        << input_sinfo->dtype);
  }
  auto output_sinfo = make_object<TensorStructInfoNode>(*input_sinfo.get());
  output_sinfo->dtype = f_compute_out_dtype(input_sinfo);
  return TensorStructInfo(output_sinfo);
}

/*!
 * \brief Infer the struct info by returning the struct info of the input argument.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \tparam arg_index The index of the argument to infer the output dtype from.
 * \return The inferred struct info.
 */
template <int arg_index>
StructInfo ReturnStructInfoFromArg(const Call& call, const BlockBuilder& ctx) {
  Op op = Downcast<Op>(call->op);
  int n_input = op->arguments.size();
  if (static_cast<int>(call->args.size()) != n_input) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << op << " op should have " << n_input << " arguments");
  }
  if (arg_index >= n_input) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << op << " op has only " << n_input
                     << "arguments, but try to get the arg with index " << arg_index);
  }
  return GetStructInfo(call->args[arg_index]);
}

/*!
 * \brief Infer the struct info for unary arithmetic elementwise ops. It's also
 * used in some NN operators.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \tparam require_float_dtype whether this op requires the input dtype to be float
 * \return The inferred struct info.
 */
template <bool require_float_dtype>
StructInfo InferStructInfoUnaryArith(const Call& call, const BlockBuilder& ctx) {
  return InferStructInfoUnary<require_float_dtype>(
      call, ctx, [](const TensorStructInfo& input_sinfo) { return input_sinfo->dtype; });
}

/*!
 * \brief Layout infer util for unary elementwise ops. It will simply take the layout of the input.
 * \param call The context Call to the operator.
 * \param desired_layouts The desired layouts of certain ops.
 * \param var_layout_map The layout of vars.
 * \return The inferred layout result.
 */
InferLayoutOutput InferLayoutUnaryEwise(const Call& call,
                                        const Map<String, Array<String>>& desired_layouts,
                                        const VarLayoutMap& var_layout_map);

/*!
 * \brief Infer the output datatype for binary arithmetic operators.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param x1_sinfo The struct info of the first operand
 * \param x2_sinfo The struct info of the second operand
 * \return The inferred output dtype.
 * \throw Throw exception if the dtype of two input TensorStructInfo don’t match
 */
inline DataType InferBinaryArithOpOutDtype(const Call& call, const BlockBuilder& ctx,
                                           const TensorStructInfo& x1_sinfo,
                                           const TensorStructInfo& x2_sinfo) {
  if (x1_sinfo->IsUnknownDtype() || x2_sinfo->IsUnknownDtype()) {
    return DataType::Void();
  } else if (x1_sinfo->dtype != x2_sinfo->dtype) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "Data types " << x1_sinfo->dtype << " and " << x2_sinfo->dtype
                     << " must be equal for binary operators");
  }
  return x1_sinfo->dtype;
}

/*!
 * \brief Infer the output shape for binary broadcast operators.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param x1_shape The shape of the first operand.
 * \param x2_shape The shape of the second operand.
 * \return The inferred output shape after broadcasting. Or `NullOpt` if the output shape cannot be
 * determined due to symbolic broadcast.
 */
Optional<Array<PrimExpr>> InferBinaryBroadcastShape(const Call& call, const BlockBuilder& ctx,
                                                    const Array<PrimExpr>& x1_shape,
                                                    const Array<PrimExpr>& x2_shape);

/*!
 * \brief Convert all axes to non-negative indices, and meanwhile check if the given array of axes
 * are all in range and non-repetitive with regards to the given ndim.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param ndim The ndim constraint, which is required to be known already.
 * \param axes The axis indices to be checked
 * \return The input axes in non-negative indexing.
 * \throw Throw exception if there exists out-of-range axis index or repetitive indices.
 */
std::vector<int> NormalizeAxes(const Call& call, const BlockBuilder& ctx, int ndim,
                               const Array<Integer>& axes);

/*!
 * \brief Convert the given axis to non-negative index. Meanwhile check if the axis is in range
 * with regards to the given ndim.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param ndim The ndim constraint.
 * \param axis The axis index to be checked
 * \return The input axis in non-negative indexing.
 * \throw Throw exception the given axis is out-of-range.
 */
inline int NormalizeAxis(const Call& call, const BlockBuilder& ctx, int ndim, int axis) {
  return NormalizeAxes(call, ctx, ndim, {axis})[0];
}

/*!
 * \brief Compute the product of all the given shape values.
 * \param shape_values The given shape values.
 * \return The product of all the given shape values.
 */
PrimExpr ComputeShapeProduct(const Array<PrimExpr>& shape_values);

/*!
 * \brief Check if the given permutation is identity permutation.
 * \param permutation The given permutation.
 * \return Whether the given permutation is identity permutation.
 */
bool IsIdentityPermutation(const std::vector<int>& permutation);

/*!
 * \brief Convert an array of integers to int64 dtype.
 * \param int_imms The input IntImms to be converted.
 * \return The conversion result, where every IntImm has dtype int64
 */
inline Array<IntImm> ConvertIntImmToInt64(const Array<IntImm>& int_imms) {
  return int_imms.Map([](const IntImm& i) { return Downcast<IntImm>(cast(DataType::Int(64), i)); });
}

/************ Utilities for NN operators ************/

/*!
 * \brief Complete the padding to a 2-length array.
 * - If the padding length is 1, the same padding is used on all left/right sides
 * - If the padding length is 2, padding is in the order of (left, right)
 * \param padding The given padding to be completed
 * \return The completed padding.
 * \throws Throws error if the input padding length is neither 1 or 2.
 */
inline Array<IntImm> GetCompletePadding1D(Array<IntImm> padding) {
  if (padding.size() == 1) {
    return {padding[0], padding[0]};
  } else if (padding.size() == 2) {
    return padding;
  }
  LOG(FATAL) << "The input padding length is expected to be either 1 or 2. However, the given "
                "padding is "
             << padding;
  throw;
}

/*!
 * \brief Complete the padding to a 4-length array.
 * - If the padding length is 1, the same padding is used on all top/left/bottom/right sides
 * - If the padding length is 2, top/bottom sides use padding[0] and left/right use padding[1]
 * - If the padding length is 4, padding is in the order of (top, left, bottom, right)
 * \param padding The given padding to be completed
 * \return The completed padding.
 * \throws Throws error if the input padding length is neither 1, 2 or 4.
 */
inline Array<IntImm> GetCompletePadding2D(Array<IntImm> padding) {
  if (padding.size() == 1) {
    return {padding[0], padding[0], padding[0], padding[0]};
  } else if (padding.size() == 2) {
    return {padding[0], padding[1], padding[0], padding[1]};
  } else if (padding.size() == 4) {
    return padding;
  }
  LOG(FATAL) << "The input padding length is expected to be either 1, 2 or 4. However, the given "
                "padding is "
             << padding;
  throw;
}

/*!
 * \brief Check if the given tensor layout can be converted to the given target layout.
 * If convertible, return the tensor layout and the bijective conversion in tir::Layout and
 * tir::BijectiveLayout accordingly.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param tensor_layout The tensor layout to be checked
 * \param tgt_layout The target layout to be matched
 * \param tensor_name The name of the input tensor
 * \return The tensor layout and the bijective conversion in tir::Layout and tir::BijectiveLayout
 * accordingly.
 */
inline std::pair<tir::Layout, tir::BijectiveLayout> CheckTensorLayout(const Call& call,
                                                                      const BlockBuilder& ctx,
                                                                      const String& tensor_layout,
                                                                      const String& tgt_layout,
                                                                      const String& tensor_name) {
  tir::Layout _tensor_layout(tensor_layout, DataType::Int(64));
  tir::BijectiveLayout tensor2tgt(_tensor_layout, tir::Layout(tgt_layout, DataType::Int(64)));
  if (!tensor2tgt.defined()) {
    ctx->ReportFatal(Diagnostic::Error(call) << call->op << " requires the given " << tensor_name
                                             << " layout to be convertible from " << tgt_layout
                                             << " layout. However, the given layout "
                                             << tensor_layout << " is not convertible.");
  }
  return {_tensor_layout, tensor2tgt};
}

/*!
 * \brief Check if the given tensor struct info has expected ndim per the given layout (or the ndim
 * is unknown), and try to cast the shape to ShapeExpr.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param sinfo The input tensor struct info to be checked.
 * \param layout The layout that the given tensor is expected to have.
 * \return The shape of the input tensor in ShapeExpr, or `NullOpt` if the shape is unknown.
 */
inline Optional<ShapeExpr> CheckNdimPerLayoutAndGetShape(const Call& call, const BlockBuilder& ctx,
                                                         const TensorStructInfo& sinfo,
                                                         const tir::Layout& layout) {
  if (!sinfo->IsUnknownNdim() && sinfo->ndim != static_cast<int>(layout.ndim())) {
    ctx->ReportFatal(Diagnostic::Error(call)
                     << "In " << call->op << ", layout " << layout << " requires the input to be "
                     << layout.ndim() << "-dim tensor. However, the given input has ndim "
                     << sinfo->ndim);
  }
  if (const auto* shape_expr = sinfo->shape.as<ShapeExprNode>()) {
    return GetRef<ShapeExpr>(shape_expr);
  }
  return NullOpt;
}

Expr MakeVMAllocStorage(Expr size, PrimValue runtime_device_index, DataTypeImm dtype,
                        StringImm storage_scope = StringImm("global"));
Expr MakeVMAllocTensor(Expr storage, PrimValue offset, Expr shape, DataTypeImm dtype);

/**
 * \brief Return the argument of the call.
 *        Note: If this is a call_tir, return the arguments passed to the TIR func
 *
 * \param call The call node
 * \return The arguments of the call
 */
Array<Expr> GetCallArgs(const Call& call);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_OP_COMMON_H_
