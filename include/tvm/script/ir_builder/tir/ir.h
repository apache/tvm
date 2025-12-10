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
#ifndef TVM_SCRIPT_IR_BUILDER_TIR_IR_H_
#define TVM_SCRIPT_IR_BUILDER_TIR_IR_H_

#include <tvm/script/ir_builder/base.h>
#include <tvm/script/ir_builder/tir/frame.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace tir {

using tvm::runtime::Tensor;
using tvm::tir::Buffer;
using tvm::tir::Var;

/*!
 * \brief The buffer declaration function.
 * \param shape The type of the buffer prior to flattening.
 * \param dtype The data type in the content of the buffer.
 * \param buffer_name The name of the buffer.
 * \param data The pointer to the head of the data.
 * \param strides The strides of each dimension.
 * \param elem_offset The offset in terms of number of dtype elements (including lanes).
 * \param storage_scope The optional storage scope of buffer data pointer.
 * \param align The alignment requirement of data pointer in bytes.
 * \param offset_factor The factor of elem_offset field.
 * \param buffer_type The buffer type.
 * \param axis_separators The separators between input axes when generating flattened output axes.
 * \return The declared buffer.
 */
Buffer BufferDecl(ffi::Array<PrimExpr> shape, DataType dtype, ffi::String buffer_name,
                  ffi::Optional<Var> data, ffi::Optional<ffi::Array<PrimExpr>> strides,
                  ffi::Optional<PrimExpr> elem_offset, ffi::String storage_scope, int align,
                  int offset_factor, ffi::String buffer_type,
                  ffi::Optional<ffi::Array<IntImm>> axis_separators);

/*!
 * \brief The primitive function statement.
 * \return The PrimFuncFrame.
 */
PrimFuncFrame PrimFunc(bool is_private);

/*!
 * \brief The PrimFunc variable arguments adding function.
 * \param name The name of the variable.
 * \param var The variable argument.
 * \return The variable.
 */
Var Arg(ffi::String name, Var var);

/*!
 * \brief The PrimFunc buffer arguments adding function.
 * \param name The name of the buffer.
 * \param buffer The buffer argument.
 * \return The buffer.
 */
Buffer Arg(ffi::String name, Buffer buffer);

/*!
 * \brief The PrimFunc naming statement.
 * \param name The name of the PrimFunc.
 */
void FuncName(ffi::String name);

/*!
 * \brief The PrimFunc annotation statement.
 * \param attrs The annotations of the PrimFunc.
 */
void FuncAttrs(ffi::Map<ffi::String, ffi::Any> attrs);

/*!
 * \brief The PrimFunc return type statement.
 * \param ret_type The return type of the PrimFunc.
 * \return The return type.
 */
Type FuncRet(Type ret_type);

/*!
 * \brief The buffer match statement.
 * \param param The parameter of the PrimFunc to match.
 * \param shape The type of the buffer prior to flattening.
 * \param dtype The data type in the content of the buffer.
 * \param data The pointer to the head of the data.
 * \param strides The strides of each dimension.
 * \param elem_offset The offset in terms of number of dtype elements (including lanes).
 * \param storage_scope The optional storage scope of buffer data pointer.
 * \param align The alignment requirement of data pointer in bytes.
 * \param offset_factor The factor of elem_offset field.
 * \param buffer_type The buffer type.
 * \param axis_separators The separators between input axes when generating flattened output axes.
 * \return The matched buffer.
 */
Buffer MatchBuffer(ObjectRef param, ffi::Array<PrimExpr> shape,
                   DataType dtype = DataType::Float(32), ffi::Optional<Var> data = std::nullopt,
                   ffi::Array<PrimExpr> strides = {}, PrimExpr elem_offset = PrimExpr(),
                   ffi::String storage_scope = "global", int align = -1, int offset_factor = 0,
                   ffi::String buffer_type = "default",
                   ffi::Optional<ffi::Array<IntImm>> axis_separators = std::nullopt);

/*!
 * \brief The block declaration statement.
 * \param name The name of the block.
 * \param no_realize The flag whether to construct BlockRealize or Block.
 * \return The BlockFrame.
 */
BlockFrame Block(ffi::String name, bool no_realize = false);

/*!
 * \brief The block initialization statement.
 * \return The BlockInitFrame.
 */
BlockInitFrame Init();

/*!
 * \brief The block predicate statement.
 * \param predicate The predicate condition.
 */
void Where(PrimExpr predicate);

/*!
 * \brief The block buffer region reading statement.
 * \param buffer_slices The array of buffer regions to read.
 */
void Reads(ffi::Array<ObjectRef> buffer_slices);

/*!
 * \brief The block buffer region writing statement.
 * \param buffer_slices The array of buffer regions to write.
 */
void Writes(ffi::Array<ObjectRef> buffer_slices);

/*!
 * \brief The block annotation statement.
 * \param attrs The annotation of the block.
 */
void BlockAttrs(ffi::Map<ffi::String, ffi::Any> attrs);

/*!
 * \brief The buffer allocation function.
 * \param shape The type of the buffer prior to flattening.
 * \param dtype The data type in the content of the buffer.
 * \param data The pointer to the head of the data.
 * \param strides The strides of each dimension.
 * \param elem_offset The offset in terms of number of dtype elements (including lanes).
 * \param storage_scope The optional storage scope of buffer data pointer.
 * \param align The alignment requirement of data pointer in bytes.
 * \param offset_factor The factor of elem_offset field.
 * \param buffer_type The buffer type.
 * \param axis_separators The separators between input axes when generating flattened output axes.
 * \return The allocated buffer.
 */
Buffer AllocBuffer(ffi::Array<PrimExpr> shape, DataType dtype = DataType::Float(32),
                   ffi::Optional<Var> data = std::nullopt, ffi::Array<PrimExpr> strides = {},
                   PrimExpr elem_offset = PrimExpr(), ffi::String storage_scope = "",
                   int align = -1, int offset_factor = 0, ffi::String buffer_type = "default",
                   ffi::Optional<ffi::Array<IntImm>> axis_separators = std::nullopt);
namespace axis {

/*!
 * \brief The spatial block axis defining function.
 * \param dom The domain of the iteration variable.
 * \param binding The binding value of the iteration variable.
 * \param dtype The data type of the iteration variable.
 * \return The iteration variable.
 */
Var Spatial(Range dom, PrimExpr binding, DataType dtype = DataType::Int(32));

/*!
 * \brief The reduced block axis defining function.
 * \param dom The domain of the iteration variable.
 * \param binding The binding value of the iteration variable.
 * \param dtype The data type of the iteration variable.
 * \return The iteration variable.
 */
Var Reduce(Range dom, PrimExpr binding, DataType dtype = DataType::Int(32));

/*!
 * \brief The scanning block axis defining function.
 * \param dom The domain of the iteration variable.
 * \param binding The binding value of the iteration variable.
 * \param dtype The data type of the iteration variable.
 * \return The iteration variable.
 */
Var Scan(Range dom, PrimExpr binding, DataType dtype = DataType::Int(32));

/*!
 * \brief The opaque block axis defining function.
 * \param dom The domain of the iteration variable.
 * \param binding The binding value of the iteration variable.
 * \param dtype The data type of the iteration variable.
 * \return The iteration variable.
 */
Var Opaque(Range dom, PrimExpr binding, DataType dtype = DataType::Int(32));

/*!
 * \brief The block axis remapping function.
 * \param kinds The types of the iteration variables.
 * \param bindings The binding values of the iteration variables.
 * \param dtype The data types of the iteration variables.
 * \return The iteration variables.
 */
ffi::Array<Var> Remap(ffi::String kinds, ffi::Array<PrimExpr> bindings,
                      DataType dtype = DataType::Int(32));

}  // namespace axis

/*!
 * \brief The serial For statement.
 * \param start The minimum value of iteration.
 * \param stop The maximum value of iteration.
 * \param annotations The optional annotations of the For statement.
 * \param step The optional step value of iteration.
 * \return The ForFrame.
 */
ForFrame Serial(PrimExpr start, PrimExpr stop,
                ffi::Optional<ffi::Map<ffi::String, Any>> annotations = std::nullopt,
                ffi::Optional<PrimExpr> step = std::nullopt);
/*!
 * \brief The parallel For statement.
 * \param start The minimum value of iteration.
 * \param stop The maximum value of iteration.
 * \param annotations The optional annotations of the For statement.
 * \param step The optional step value of iteration.
 * \return The ForFrame.
 */
ForFrame Parallel(PrimExpr start, PrimExpr stop,
                  ffi::Optional<ffi::Map<ffi::String, Any>> annotations = std::nullopt,
                  ffi::Optional<PrimExpr> step = std::nullopt);
/*!
 * \brief The vectorized For statement.
 * \param start The minimum value of iteration.
 * \param stop The maximum value of iteration.
 * \param annotations The optional annotations of the For statement.
 * \param step The optional step value of iteration.
 * \return The ForFrame.
 */
ForFrame Vectorized(PrimExpr start, PrimExpr stop,
                    ffi::Optional<ffi::Map<ffi::String, Any>> annotations = std::nullopt,
                    ffi::Optional<PrimExpr> step = std::nullopt);
/*!
 * \brief The unrolled For statement.
 * \param start The minimum value of iteration.
 * \param stop The maximum value of iteration.
 * \param annotations The optional annotations of the For statement.
 * \param step The optional step value of iteration.
 * \return The ForFrame.
 */
ForFrame Unroll(PrimExpr start, PrimExpr stop,
                ffi::Optional<ffi::Map<ffi::String, Any>> annotations = std::nullopt,
                ffi::Optional<PrimExpr> step = std::nullopt);
/*!
 * \brief The thread-binding For statement.
 * \param start The minimum value of iteration.
 * \param stop The maximum value of iteration.
 * \param thread The thread for loop variable to bind.
 * \param annotations The optional annotations of the For statement.
 * \return The ForFrame.
 */
ForFrame ThreadBinding(PrimExpr start, PrimExpr stop, ffi::String thread,
                       ffi::Optional<ffi::Map<ffi::String, Any>> annotations = std::nullopt);
/*!
 * \brief The grid For statement.
 * \param extents The extents of the iteration.
 * \return The ForFrame.
 */
ForFrame Grid(ffi::Array<PrimExpr> extents);

/*!
 * \brief The assertion statement.
 * \param condition The assertion condition.
 * \param message The error message when the assertion fails.
 * \return The AssertFrame.
 */
AssertFrame Assert(PrimExpr condition, ffi::String message);

/*!
 * \brief The let binding.
 * \param value The value to be bound.
 * \param type_annotation  The type annotation of the let binding.
 *                         Usually it is used for fine-grained var typing,
 *                         particularly, PointerType.
 * \param var The variable to be bound. If not specified, a new variable will be created.
 * \return The created LetFrame.
 */
LetFrame LetStmt(PrimExpr value, ffi::Optional<Type> type_annotation = std::nullopt,
                 ffi::Optional<Var> var = std::nullopt);

/*!
 * \brief The realization.
 * \param buffer_slice The region of buffer access.
 * \param storage_scope The storage scope associated with this realization.
 * \param condition The condition expression.
 * \return The result RealizeFrame.
 */
RealizeFrame Realize(tvm::tir::BufferRegion buffer_slice, ffi::String storage_scope,
                     PrimExpr condition);

/*!
 * \brief The allocate node.
 * \param extents The extents of the allocate.
 * \param dtype The data type of the buffer.
 * \param storage_scope The storage scope.
 * \param condition The condition.
 * \param annotations Additional annotation hints.
 * \return The created AllocateFrame.
 */
AllocateFrame Allocate(ffi::Array<PrimExpr> extents, DataType dtype, ffi::String storage_scope = "",
                       ffi::Optional<PrimExpr> condition = std::nullopt,
                       ffi::Optional<ffi::Map<ffi::String, Any>> annotations = std::nullopt);

/*!
 * \brief The allocate constant node.
 * \param data The data associated with the constant.
 * \param dtype The data type of the buffer.
 * \param extents The extents of the allocate.
 * \param annotations Additional annotation hints.
 * \return The created AllocateConstFrame.
 */
AllocateConstFrame AllocateConst(
    Tensor data, DataType dtype, ffi::Array<PrimExpr> extents,
    ffi::Optional<ffi::Map<ffi::String, Any>> annotations = std::nullopt);

/*!
 * \brief Create an attribute.
 * \param node The node to annotate the attribute.
 * \param attr_key Attribute type key.
 * \param value The value of the attribute.
 * \return The result AttrFrame.
 */
AttrFrame Attr(ffi::Any node, ffi::String attr_key, PrimExpr value);

/*!
 * \brief Create a while loop.
 * \param condition The termination condition of the loop.
 * \return The result WhileFrame.
 */
WhileFrame While(PrimExpr condition);

/*!
 * \brief Create an if statement.
 * \param condition The condition of if statement.
 * \return The result IfFrame.
 */
IfFrame If(PrimExpr condition);

/*!
 * \brief Create a then.
 * \return The result ThenFrame.
 */
ThenFrame Then();

/*!
 * \brief Create an else.
 * \return The result ElseFrame.
 */
ElseFrame Else();

/*!
 * \brief The buffer declaration frame.
 * \param shape The type of the buffer prior to flattening.
 * \param dtype The data type in the content of the buffer.
 * \param buffer_name The name of the buffer.
 * \param data The pointer to the head of the data.
 * \param strides The strides of each dimension.
 * \param elem_offset The offset in terms of number of dtype elements (including lanes).
 * \param storage_scope The optional storage scope of buffer data pointer.
 * \param align The alignment requirement of data pointer in bytes.
 * \param offset_factor The factor of elem_offset field.
 * \param buffer_type The buffer type.
 * \param axis_separators The separators between input axes when generating flattened output axes.
 * \return The declared buffer.
 */
DeclBufferFrame DeclBuffer(ffi::Array<PrimExpr> shape, DataType dtype, ffi::String buffer_name,
                           ffi::Optional<Var> data, ffi::Optional<ffi::Array<PrimExpr>> strides,
                           ffi::Optional<PrimExpr> elem_offset, ffi::String storage_scope,
                           int align, int offset_factor, ffi::String buffer_type,
                           ffi::Optional<ffi::Array<IntImm>> axis_separators);

/*!
 * \brief Launch a thread.
 * \param var The iteration variable.
 * \param extent The extent of environment thread.
 * \return The result LaunchThreadFrame.
 */
LaunchThreadFrame LaunchThread(Var var, PrimExpr extent);

/*!
 * \brief Launch a new thread.
 * \param thread_tag The thread type tag.
 * \param extent The extent of environment thread.
 * \return The result LaunchThreadFrame.
 */
LaunchThreadFrame LaunchThread(ffi::String thread_tag, PrimExpr extent);

/*!
 * \brief Bind a var to thread env.
 * \param thread_tag The thread type tag.
 * \param dtype The data type of the variable.
 * \return The result variable which gets bound to the thread env.
 */
Var EnvThread(ffi::String thread_tag, DataType dtype = DataType::Int(32));

/*!
 * \brief Store data in a buffer.
 * \param buffer The buffer.
 * \param value The value to be stored.
 * \param indices The indices location to be stored.
 * \param predicate A vector mask of boolean values indicating which lanes of a vector are to be
 * stored. The number lanes of the mask must be equal to the number of lanes in value.
 */
void BufferStore(Buffer buffer, PrimExpr value, ffi::Array<PrimExpr> indices,
                 ffi::Optional<PrimExpr> predicate);

/*!
 * \brief Evaluate the input expression.
 * \param value The input expression to evaluate.
 */
void Evaluate(PrimExpr value);

/*!
 * \brief Create a TIR var that represents a pointer
 *
 * \param dtype The data type of the pointer.
 *
 * \param storage_scope The storage scope of the pointer.
 *
 * \param is_size_var Whether the pointer is a size var.
 *
 * \param is_unknown_type Used to distinguish between
 * `PrimType(DataType::Handle())` and
 * `PointerType(PrimType(DataType::Void()))`.  If true, resolve dtype
 * of `Void()` as `PrimType`, and if false resolve dtype of `Void()`
 * as a `PointerType`.
 *
 * \return The pointer.
 */
inline Var Handle(runtime::DataType dtype = runtime::DataType::Void(),
                  ffi::String storage_scope = "global", bool is_size_var = false,
                  bool is_unknown_type = false) {
  Type type_annotation{nullptr};
  if (is_unknown_type && storage_scope == "global") {
    type_annotation = PrimType(runtime::DataType::Handle());
  } else {
    type_annotation = PointerType(PrimType(dtype), storage_scope);
  }
  return is_size_var ? tvm::tir::SizeVar("", type_annotation) : tvm::tir::Var("", type_annotation);
}

inline Var TensormapHandle() { return tvm::tir::Var("", PointerType(TensorMapType())); }

#define TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(FuncName, DType)                                \
  inline PrimExpr FuncName(ffi::Optional<PrimExpr> expr = std::nullopt,                   \
                           bool is_size_var = false) {                                    \
    DataType dtype = DType;                                                               \
    return expr.defined()                                                                 \
               ? tvm::cast(dtype, expr.value())                                           \
               : (is_size_var ? tvm::tir::SizeVar("", dtype) : tvm::tir::Var("", dtype)); \
  }

#define TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_SIZES(DType, FDType) \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(DType##8, FDType(8));      \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(DType##16, FDType(16));    \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(DType##32, FDType(32));    \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(DType##64, FDType(64));

TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_SIZES(BFloat, DataType::BFloat);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_SIZES(Float, DataType::Float);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_SIZES(UInt, DataType::UInt);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_SIZES(Int, DataType::Int);

#define TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_LANES(FuncName, FDType, Size) \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(FuncName##x2, FDType(Size, 2));     \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(FuncName##x4, FDType(Size, 4));     \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(FuncName##x8, FDType(Size, 8));     \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(FuncName##x16, FDType(Size, 16));   \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(FuncName##x32, FDType(Size, 32));   \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(FuncName##x64, FDType(Size, 64));

#define TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_SIZES_LANES(DType, FDType) \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_LANES(DType##8, FDType, 8);      \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_LANES(DType##16, FDType, 16);    \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_LANES(DType##32, FDType, 32);    \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_LANES(DType##64, FDType, 64);

TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_SIZES_LANES(BFloat, DataType::BFloat);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_SIZES_LANES(Float, DataType::Float);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_SIZES_LANES(UInt, DataType::UInt);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_SIZES_LANES(Int, DataType::Int);

#define TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_LANES_FIXED_SIZE(DType, FDType) \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(DType, FDType(1));                    \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(DType##x2, FDType(2));                \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(DType##x4, FDType(4));                \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(DType##x8, FDType(8));                \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(DType##x16, FDType(16));              \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(DType##x32, FDType(32));              \
  TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(DType##x64, FDType(64));

TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_LANES_FIXED_SIZE(Float8E3M4, DataType::Float8E3M4);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_LANES_FIXED_SIZE(Float8E4M3, DataType::Float8E4M3);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_LANES_FIXED_SIZE(Float8E4M3B11FNUZ, DataType::Float8E4M3B11FNUZ);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_LANES_FIXED_SIZE(Float8E4M3FN, DataType::Float8E4M3FN);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_LANES_FIXED_SIZE(Float8E4M3FNUZ, DataType::Float8E4M3FNUZ);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_LANES_FIXED_SIZE(Float8E5M2, DataType::Float8E5M2);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_LANES_FIXED_SIZE(Float8E5M2FNUZ, DataType::Float8E5M2FNUZ);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_LANES_FIXED_SIZE(Float8E8M0FNU, DataType::Float8E8M0FNU);

TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_LANES_FIXED_SIZE(Float6E2M3FN, DataType::Float6E2M3FN);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_LANES_FIXED_SIZE(Float6E3M2FN, DataType::Float6E3M2FN);

TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_LANES_FIXED_SIZE(Float4E2M1FN, DataType::Float4E2M1FN);

TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(Boolean, DataType::Bool());
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(Void, DataType::Void());

#undef TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST

}  // namespace tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_TIR_IR_H_
