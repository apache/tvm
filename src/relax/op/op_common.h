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
#include <tvm/ffi/cast.h>
#include <tvm/ffi/extra/visit_error_context.h>
#include <tvm/relax/op_attr_types.h>
#include <tvm/s_tir/data_layout.h>

#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "../transform/infer_amp_utils.h"
#include "../transform/infer_layout_utils.h"

namespace tvm {
namespace relax {

/************ Op input type getter ************/

/*!
 * \brief Check that the operator has
 *
 * Verify that the number of arguments matches the expected number for
 * the operator.
 *
 * \param call The context Call to the operator.
 *
 * \param ctx The error reporting context.
 */
void CheckNumArguments(const Call& call, const BlockBuilder& ctx);

/*!
 * \brief Get the tensor type of the operator input.
 * \param call The context Call to the operator.
 * \param i_arg The index of the argument to check
 * \param ctx The error reporting context.
 * \return The tensor type of the argument
 */
TensorType GetInputTensorType(const Call& call, size_t i_arg, const BlockBuilder& ctx);

/*!
 * \brief Get the tensor type of the operator input.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \return The tensor type of each input.
 * \note This function require every input to be Tensor. The number of call arguments is required
 * to match the number of inputs of the op being called.
 */
ffi::Array<TensorType> GetInputTensorType(const Call& call, const BlockBuilder& ctx);

/*!
 * \brief Get the tensor type of the unary operator input.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \return The tensor type of the unary operator input.
 * \throw Throw exception if the number of input is not one, or the type of the input is not
 * a tensor type.
 */
inline TensorType GetUnaryInputTensorType(const Call& call, const BlockBuilder& ctx) {
  return GetInputTensorType(call, ctx)[0];
}

/*!
 * \brief Get the tensor type of tuple input.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param tup The input tuple.
 * \return The tensor struct infos of tuple input.
 * \throw Throw exception if input expression is not a tuple.
 */
ffi::Array<TensorType> GetTensorTypeFromTuple(const Call& call, const BlockBuilder& ctx,
                                              const Expr& tup);

namespace detail {
/*! \brief Implementation helper for GetArgType */
template <typename ArgType>
ArgType GetArgTypeByIndex(const Call& call, const Op& op, const BlockBuilder& ctx, size_t index) {
  if (!call->args[index]->ty.defined()) {
    TVM_FFI_VISIT_THROW(InternalError, call)
        << op << " op should have arguments with defined Type.  "
        << "However, args[" << index << "] has undefined type.";
  }

  auto ty = GetType(call->args[index]);
  auto typed_ty = ty.as<ArgType>();

  if (!typed_ty.has_value()) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << op << " requires that args[" << index << "] be a " << ArgType::ContainerType::_type_key
        << ", but was instead " << ty << " of type " << ty->GetTypeKey();
  }

  return typed_ty.value();
}

/*! \brief Implementation helper for GetArgType */
template <typename... ArgTypes, size_t... Indices>
std::tuple<ArgTypes...> GetArgTypeHelper(const Call& call, const Op& op, const BlockBuilder& ctx,
                                         std::index_sequence<Indices...>) {
  return std::tuple<ArgTypes...>{GetArgTypeByIndex<ArgTypes>(call, op, ctx, Indices)...};
}
}  // namespace detail

/*!
 * \brief Get all arg struct infos as expected types
 *
 * \tparam ArgTypes The expected types of arguments, in the order they appear.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \return The tensor struct infos of tuple input.
 * \throw Throw exception if input expression is not a tuple.
 */
template <typename... ArgTypes>
std::tuple<ArgTypes...> GetArgType(const Call& call, const BlockBuilder& ctx) {
  Op op = Downcast<Op>(call->op);
  size_t n_input = op->arguments.size();

  // Unfortunately, because the `.add_argument()` calls in
  // TVM_REGISTER_OP occur during initialization of globals and are
  // not available at compile-time, this cannot be a static_assert.
  TVM_FFI_ICHECK_EQ(n_input, sizeof...(ArgTypes))
      << "Internal error: " << op << " op defines " << n_input
      << " arguments in its TVM_REGISTER_OP() call, "
      << "but GetArgType was given " << sizeof...(ArgTypes) << " template arguments.";

  return detail::GetArgTypeHelper<ArgTypes...>(call, op, ctx,
                                               std::make_index_sequence<sizeof...(ArgTypes)>());
}

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
      .set_attr<bool>("FPurity", true)

/*!
 * \brief Quick helper macro to expose a make-function to construct the operator.
 * \param OpName The name of the operator as well as the make-function name, which will
 * be prepended with a prefix "relax.op." as the FFI identifier string for the make function,
 * \param OpRegName The identifier of the operator in the registry.
 */
#define RELAX_UNARY_OP_INTERFACE(OpName, OpRegName)                       \
  Expr OpName(Expr x) {                                                   \
    static const Op& op = Op::Get("relax." OpRegName);                    \
    return Call(op, {std::move(x)}, Attrs(), {});                         \
  }                                                                       \
  TVM_FFI_STATIC_INIT_BLOCK() {                                           \
    tvm::ffi::reflection::GlobalDef().def("relax.op." OpRegName, OpName); \
  }

/************ Utilities ************/

/*!
 * \brief Infer the type for unary elementwise ops.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param f_compute_out_dtype The function to compute the output dtype, with
 * signature DataType f_compute_out_dtype(const TensorType& input_ty).
 * \tparam require_float_dtype whether this op requires the input dtype to be float
 * \tparam Ftype the type of f_compute_out_dtype
 * \return The inferred type.
 */
template <bool require_float_dtype, typename FType>
inline Type InferTypeUnary(const Call& call, const BlockBuilder& ctx, FType f_compute_out_dtype) {
  TensorType input_ty = GetUnaryInputTensorType(call, ctx);
  if (require_float_dtype && !input_ty->IsUnknownDtype() &&
      (!input_ty->dtype.is_float() && !input_ty->dtype.is_bfloat())) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << call->op
        << " requires the input tensor to have float dtype. However, the given input dtype is "
        << input_ty->dtype;
  }
  auto output_ty = ffi::make_object<TensorTypeNode>(*input_ty.get());
  output_ty->dtype = f_compute_out_dtype(input_ty);
  if (call->ty_args.size() > 0) {
    auto defined_ty = call->ty_args[0].as<TensorTypeNode>();
    TVM_FFI_ICHECK(defined_ty);
    auto shape = output_ty->GetShape();
    TVM_FFI_ICHECK(shape.defined());
    TVM_FFI_ICHECK(defined_ty->vdevice.has_value());
    return TensorType(ShapeExpr(shape.value()), output_ty->dtype, defined_ty->vdevice.value());
  } else {
    return TensorType(output_ty);
  }
}

/*!
 * \brief Infer the type by returning the type of the input argument.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \tparam arg_index The index of the argument to infer the output dtype from.
 * \return The inferred type.
 */
template <int arg_index>
Type ReturnTypeFromArg(const Call& call, const BlockBuilder& ctx) {
  Op op = Downcast<Op>(call->op);
  int n_input = op->arguments.size();
  if (static_cast<int>(call->args.size()) != n_input) {
    TVM_FFI_VISIT_THROW(ValueError, call) << op << " op should have " << n_input << " arguments";
  }
  if (arg_index >= n_input) {
    TVM_FFI_VISIT_THROW(IndexError, call)
        << op << " op has only " << n_input << "arguments, but try to get the arg with index "
        << arg_index;
  }
  return GetType(call->args[arg_index]);
}

/*!
 * \brief Infer the type for unary arithmetic elementwise ops. It's also
 * used in some NN operators.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \tparam require_float_dtype whether this op requires the input dtype to be float
 * \return The inferred type.
 */
template <bool require_float_dtype>
Type InferTypeUnaryArith(const Call& call, const BlockBuilder& ctx) {
  return InferTypeUnary<require_float_dtype>(
      call, ctx, [](const TensorType& input_ty) { return input_ty->dtype; });
}

/*!
 * \brief SLayout infer util for unary elementwise ops. It will simply take the layout of the input.
 * \param call The context Call to the operator.
 * \param desired_layouts The desired layouts of certain ops.
 * \param var_layout_map The layout of vars.
 * \return The inferred layout result.
 */
InferLayoutOutput InferLayoutUnaryEwise(
    const Call& call, const ffi::Map<ffi::String, ffi::Array<ffi::String>>& desired_layouts,
    const VarLayoutMap& var_layout_map);

/*!
 * \brief Get the element dtype from Type
 *
 * \param ty The Type to expect
 * \return The inferred element dtype.
 * \throw Throw exception if the Type doesn't have an element type.
 */
inline std::optional<DataType> GetElementDType(const Type& ty) {
  if (const auto* prim = ty.as<PrimTypeNode>()) {
    return prim->dtype;
  } else if (const auto* tensor = ty.as<TensorTypeNode>()) {
    return tensor->dtype;
  } else {
    return std::nullopt;
    TVM_FFI_THROW(TypeError) << "Only PrimType and TensorType "
                             << "have an associated data type.  "
                             << "Cannot determine element type of " << ty;
  }
}

/*!
 * \brief Infer the output datatype for binary arithmetic operators.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param lhs_ty The type of the first operand
 * \param rhs_ty The type of the second operand
 * \return The inferred output dtype.
 * \throw Throw exception if the dtype of two input TensorType don’t match
 */
inline DataType InferBinaryArithOpOutDtype(const Call& call, const BlockBuilder& ctx,
                                           const Type& lhs_ty, const Type& rhs_ty) {
  auto opt_lhs_dtype = GetElementDType(lhs_ty);
  if (!opt_lhs_dtype) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Binary operators must have the same datatype for both operands.  "
        << "However, " << call << " has argument " << call->args[0] << " on the LHS, with type "
        << lhs_ty << ".   This is of type " << lhs_ty->GetTypeKey()
        << ", which does not have a datatype.";
  }
  auto lhs_dtype = opt_lhs_dtype.value();

  auto opt_rhs_dtype = GetElementDType(rhs_ty);
  if (!opt_rhs_dtype) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Binary operators must have the same datatype for both operands.  "
        << "However, " << call << " has argument " << call->args[1] << " on the RHS, with type "
        << rhs_ty << ".   This is of type " << rhs_ty->GetTypeKey()
        << ", which does not have a datatype.";
  }
  auto rhs_dtype = opt_rhs_dtype.value();

  if (lhs_dtype.is_void() || rhs_dtype.is_void()) {
    return DataType::Void();
  } else if (lhs_dtype != rhs_dtype && !lhs_dtype.is_bool() && !rhs_dtype.is_bool()) {
    TVM_FFI_VISIT_THROW(TypeError, call)
        << "Binary operators must have the same datatype for both operands.  "
        << "However, " << call << " uses datatype " << lhs_dtype << " on the LHS (Type of "
        << lhs_ty << "), and datatype " << rhs_dtype << " on the RHS (Type of " << rhs_ty << ").";
  }
  return lhs_dtype;
}

/*!
 * \brief Infer the output virtual device for binary arithmetic operators.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param lhs_ty The type of the first operand
 * \param rhs_ty The type of the second operand
 * \return The inferred output vdevice.
 * \throw Throw exception if the vdevice of two input TensorType don’t match
 */
inline ffi::Optional<VDevice> InferBinaryArithOpOutVDevice(const Call& call,
                                                           const BlockBuilder& ctx,
                                                           const Type& lhs_ty, const Type& rhs_ty) {
  auto get_vdevice = [&](const Type& ty) -> ffi::Optional<VDevice> {
    if (const auto* tensor = ty.as<TensorTypeNode>()) {
      return tensor->vdevice;
    } else {
      return std::nullopt;
    }
  };

  /*
   * This is the case where the output VDevice defined by a customization pass.
   * Like targets that supports mixed VDevices (like differed by memory_scope for Adreno)
   * and have specialized derivation for output VDevice.
   */
  if (call->ty_args.size() > 0) {
    return get_vdevice(call->ty_args[0]);
  }

  auto lhs_vdevice = get_vdevice(lhs_ty);
  auto rhs_vdevice = get_vdevice(rhs_ty);

  if (!lhs_vdevice.defined() || !lhs_vdevice.value()->target.defined()) {
    return rhs_vdevice;
  }
  if (!rhs_vdevice.defined() || !rhs_vdevice.value()->target.defined()) {
    return lhs_vdevice;
  }

  if (lhs_vdevice.value() != rhs_vdevice.value()) {
    TVM_FFI_VISIT_THROW(ValueError, call) << "Binary operators with Tensor arguments "
                                          << "must have the same VDevice for both operands.  "
                                          << "However, " << call << " has a LHS on VDevice "
                                          << lhs_vdevice << " and a RHS on VDevice " << rhs_vdevice;
  }
  return lhs_vdevice;
}

/*! \brief Result of binary broadcast shape inference without diagnostic context. */
struct BinaryBroadcastShapeInferResult {
  enum class Status {
    /*! \brief Broadcast output shape is known. */
    kSuccess,
    /*! \brief Shapes may be broadcastable but cannot be proved symbolically. */
    kUnknown,
    /*! \brief Concrete shapes are not broadcastable. */
    kConflict,
  };

  /*! \brief Inference status. */
  Status status = Status::kUnknown;
  /*! \brief Broadcasted shape if status is kSuccess. */
  ffi::Optional<ffi::Array<PrimExpr>> shape;
  /*! \brief Human-readable conflict description if status is kConflict. */
  ffi::Optional<ffi::String> message;
};

/*!
 * \brief Infer the output shape for binary broadcast operators.
 * \param analyzer The arithmetic analyzer used to prove shape equality.
 * \param x1_shape The shape of the first operand.
 * \param x2_shape The shape of the second operand.
 * \return Inference status and broadcasted shape, or a conflict message.
 */
BinaryBroadcastShapeInferResult InferBinaryBroadcastShape(arith::AnalyzerObj* analyzer,
                                                          const ffi::Array<PrimExpr>& x1_shape,
                                                          const ffi::Array<PrimExpr>& x2_shape);

/*!
 * \brief Infer the output shape for binary broadcast operators.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param x1_shape The shape of the first operand.
 * \param x2_shape The shape of the second operand.
 * \return The inferred output shape after broadcasting. Or `std::nullopt` if the output shape
 * cannot be determined due to symbolic broadcast.
 */
ffi::Optional<ffi::Array<PrimExpr>> InferBinaryBroadcastShape(const Call& call,
                                                              const BlockBuilder& ctx,
                                                              const ffi::Array<PrimExpr>& x1_shape,
                                                              const ffi::Array<PrimExpr>& x2_shape);

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
                               const ffi::Array<int64_t>& axes);

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
PrimExpr ComputeShapeProduct(const ffi::Array<PrimExpr>& shape_values);

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
inline ffi::Array<IntImm> ConvertIntImmToInt64(const ffi::Array<IntImm>& int_imms) {
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
inline ffi::Array<int64_t> GetCompletePadding1D(ffi::Array<int64_t> padding) {
  if (padding.size() == 1) {
    return {padding[0], padding[0]};
  } else if (padding.size() == 2) {
    return padding;
  }
  TVM_FFI_THROW(InternalError)
      << "The input padding length is expected to be either 1 or 2. However, the given "
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
inline ffi::Array<int64_t> GetCompletePadding2D(ffi::Array<int64_t> padding) {
  if (padding.size() == 1) {
    return {padding[0], padding[0], padding[0], padding[0]};
  } else if (padding.size() == 2) {
    return {padding[0], padding[1], padding[0], padding[1]};
  } else if (padding.size() == 4) {
    return padding;
  }
  TVM_FFI_THROW(InternalError)
      << "The input padding length is expected to be either 1, 2 or 4. However, the given "
         "padding is "
      << padding;
  throw;
}

/*!
 * \brief Complete the padding to a 6-length array.
 * - If the padding length is 1, the same padding is used on all front/top/left/back/bottom/right
 * sides
 * - If the padding length is 3, front/back sides use padding[0], top/bottom sides use padding[1]
 * and left/right use padding[2]
 * - If the padding length is 6, padding is in the order of (front, top, left, back, bottom, right)
 * \param padding The given padding to be completed
 * \return The completed padding.
 * \throws Throws error if the input padding length is neither 1, 3 or 6.
 */
inline ffi::Array<int64_t> GetCompletePadding3D(ffi::Array<int64_t> padding) {
  if (padding.size() == 1) {
    return {padding[0], padding[0], padding[0], padding[0], padding[0], padding[0]};
  } else if (padding.size() == 3) {
    return {padding[0], padding[1], padding[2], padding[0], padding[1], padding[2]};
  } else if (padding.size() == 6) {
    return padding;
  }
  TVM_FFI_THROW(InternalError)
      << "The input padding length is expected to be either 1, 3 or 6. However, the given "
         "padding is "
      << padding;
  throw;
}

/*!
 * \brief Check if the given tensor layout can be converted to the given target layout.
 * If convertible, return the tensor layout and the bijective conversion in tirx::SLayout and
 * tirx::SBijectiveLayout accordingly.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param tensor_layout The tensor layout to be checked
 * \param tgt_layout The target layout to be matched
 * \param tensor_name The name of the input tensor
 * \return The tensor layout and the bijective conversion in tirx::SLayout and
 * tirx::SBijectiveLayout accordingly.
 */
inline std::pair<tirx::SLayout, tirx::SBijectiveLayout> CheckTensorLayout(
    const Call& call, const BlockBuilder& ctx, const ffi::String& tensor_layout,
    const ffi::String& tgt_layout, const ffi::String& tensor_name) {
  tirx::SLayout _tensor_layout(tensor_layout, DataType::Int(64));
  tirx::SBijectiveLayout tensor2tgt(_tensor_layout, tirx::SLayout(tgt_layout, DataType::Int(64)));
  if (!tensor2tgt.defined()) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << call->op << " requires the given " << tensor_name << " layout to be convertible from "
        << tgt_layout << " layout. However, the given layout " << tensor_layout
        << " is not convertible.";
  }
  return {_tensor_layout, tensor2tgt};
}

/*!
 * \brief Check if the given tensor type has expected ndim per the given layout (or the ndim
 * is unknown), and try to cast the shape to ShapeExpr.
 * \param call The context Call to the operator.
 * \param ctx The error reporting context.
 * \param ty The input tensor type to be checked.
 * \param layout The layout that the given tensor is expected to have.
 * \return The shape of the input tensor in ShapeExpr, or `std::nullopt` if the shape is unknown.
 */
inline ffi::Optional<ShapeExpr> CheckNdimPerLayoutAndGetShape(const Call& call,
                                                              const BlockBuilder& ctx,
                                                              const TensorType& ty,
                                                              const tirx::SLayout& layout) {
  if (!ty->IsUnknownNdim() && ty->ndim != static_cast<int>(layout.ndim())) {
    TVM_FFI_VISIT_THROW(ValueError, call)
        << "In " << call->op << ", layout " << layout << " requires the input to be "
        << layout.ndim() << "-dim tensor. However, the given input has ndim " << ty->ndim;
  }
  if (const auto* shape_expr = ty->shape.as<ShapeExprNode>()) {
    return ffi::GetRef<ShapeExpr>(shape_expr);
  }
  return std::nullopt;
}

Expr MakeVMAllocStorage(Expr size, PrimValue runtime_device_index, DataTypeImm dtype,
                        StringImm storage_scope = StringImm("global"));
Expr MakeVMAllocTensor(Expr storage, PrimValue offset, Expr shape, DataTypeImm dtype,
                       PrimValue runtime_device_index);

Expr MakeAllocTensor(Expr shape, DataTypeImm dtype, PrimValue runtime_device_index,
                     StringImm storage_scope = StringImm("global"));

/**
 * \brief Return the argument of the call.
 *        Note: If this is a call_tir, return the arguments passed to the TIR func
 *
 * \param call The call node
 * \return The arguments of the call
 */
ffi::Array<Expr> GetCallArgs(const Call& call);

/**
 * \brief Checks the given shape can be proved from the source layout to dst layout
 * \param input_layout is the layout of given shape
 * \param desired_layout is the target layout the shape to be transformed
 * \param shape array
 * \return true or false depending on the compatibility
 */
bool CanProveLayoutTransform(const SLayout& input_layout, const SLayout& desired_layout,
                             ffi::Array<PrimExpr> shape);

}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_OP_OP_COMMON_H_
