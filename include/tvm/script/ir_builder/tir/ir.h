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

using tvm::runtime::NDArray;
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
Buffer BufferDecl(Array<PrimExpr> shape, DataType dtype, String buffer_name, Optional<Var> data,
                  Optional<Array<PrimExpr>> strides, Optional<PrimExpr> elem_offset,
                  String storage_scope, int align, int offset_factor, String buffer_type,
                  Optional<Array<IntImm>> axis_separators);

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
Var Arg(String name, Var var);

/*!
 * \brief The PrimFunc buffer arguments adding function.
 * \param name The name of the buffer.
 * \param buffer The buffer argument.
 * \return The buffer.
 */
Buffer Arg(String name, Buffer buffer);

/*!
 * \brief The PrimFunc naming statement.
 * \param name The name of the PrimFunc.
 */
void FuncName(String name);

/*!
 * \brief The PrimFunc annotation statement.
 * \param attrs The annotations of the PrimFunc.
 */
void FuncAttrs(Map<String, ObjectRef> attrs);

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
Buffer MatchBuffer(ObjectRef param, Array<PrimExpr> shape, DataType dtype = DataType::Float(32),
                   Optional<Var> data = NullOpt, Array<PrimExpr> strides = {},
                   PrimExpr elem_offset = PrimExpr(), String storage_scope = "global",
                   int align = -1, int offset_factor = 0, String buffer_type = "default",
                   Array<IntImm> axis_separators = {});

/*!
 * \brief The block declaration statement.
 * \param name The name of the block.
 * \param no_realize The flag whether to construct BlockRealize or Block.
 * \return The BlockFrame.
 */
BlockFrame Block(String name, bool no_realize = false);

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
void Reads(Array<ObjectRef> buffer_slices);

/*!
 * \brief The block buffer region writing statement.
 * \param buffer_slices The array of buffer regions to write.
 */
void Writes(Array<ObjectRef> buffer_slices);

/*!
 * \brief The block annotation statement.
 * \param attrs The annotation of the block.
 */
void BlockAttrs(Map<String, ObjectRef> attrs);

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
Buffer AllocBuffer(Array<PrimExpr> shape, DataType dtype = DataType::Float(32),
                   Optional<Var> data = NullOpt, Array<PrimExpr> strides = {},
                   PrimExpr elem_offset = PrimExpr(), String storage_scope = "", int align = -1,
                   int offset_factor = 0, String buffer_type = "default",
                   Array<IntImm> axis_separators = {});
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
Array<Var> Remap(String kinds, Array<PrimExpr> bindings, DataType dtype = DataType::Int(32));

}  // namespace axis

/*!
 * \brief The serial For statement.
 * \param start The minimum value of iteration.
 * \param stop The maximum value of iteration.
 * \param annotations The optional annotations of the For statement.
 * \return The ForFrame.
 */
ForFrame Serial(PrimExpr start, PrimExpr stop,
                Optional<Map<String, ObjectRef>> annotations = NullOpt);
/*!
 * \brief The parallel For statement.
 * \param start The minimum value of iteration.
 * \param stop The maximum value of iteration.
 * \param annotations The optional annotations of the For statement.
 * \return The ForFrame.
 */
ForFrame Parallel(PrimExpr start, PrimExpr stop,
                  Optional<Map<String, ObjectRef>> annotations = NullOpt);
/*!
 * \brief The vectorized For statement.
 * \param start The minimum value of iteration.
 * \param stop The maximum value of iteration.
 * \param annotations The optional annotations of the For statement.
 * \return The ForFrame.
 */
ForFrame Vectorized(PrimExpr start, PrimExpr stop,
                    Optional<Map<String, ObjectRef>> annotations = NullOpt);
/*!
 * \brief The unrolled For statement.
 * \param start The minimum value of iteration.
 * \param stop The maximum value of iteration.
 * \param annotations The optional annotations of the For statement.
 * \return The ForFrame.
 */
ForFrame Unroll(PrimExpr start, PrimExpr stop,
                Optional<Map<String, ObjectRef>> annotations = NullOpt);
/*!
 * \brief The thread-binding For statement.
 * \param start The minimum value of iteration.
 * \param stop The maximum value of iteration.
 * \param thread The thread for loop variable to bind.
 * \param annotations The optional annotations of the For statement.
 * \return The ForFrame.
 */
ForFrame ThreadBinding(PrimExpr start, PrimExpr stop, String thread,
                       Optional<Map<String, ObjectRef>> annotations = NullOpt);
/*!
 * \brief The grid For statement.
 * \param extents The extents of the iteration.
 * \return The ForFrame.
 */
ForFrame Grid(Array<PrimExpr> extents);

/*!
 * \brief The assertion statement.
 * \param condition The assertion condition.
 * \param message The error message when the assertion fails.
 * \return The AssertFrame.
 */
AssertFrame Assert(PrimExpr condition, String message);

/*!
 * \brief The let binding.
 * \param value The value to be bound.
 * \param type_annotation  The type annotation of the let binding.
 *                         Usually it is used for fine-grained var typing,
 *                         particularly, PointerType.
 * \param var The variable to be bound. If not specified, a new variable will be created.
 * \return The created LetFrame.
 */
LetFrame LetStmt(PrimExpr value, Optional<Type> type_annotation = NullOpt,
                 Optional<Var> var = NullOpt);

/*!
 * \brief The realization.
 * \param buffer_slice The region of buffer access.
 * \param storage_scope The storage scope associated with this realization.
 * \param condition The condition expression.
 * \return The result RealizeFrame.
 */
RealizeFrame Realize(tvm::tir::BufferRegion buffer_slice, String storage_scope, PrimExpr condition);

/*!
 * \brief The allocate node.
 * \param extents The extents of the allocate.
 * \param dtype The data type of the buffer.
 * \param storage_scope The storage scope.
 * \param condition The condition.
 * \param annotations Additional annotation hints.
 * \return The created AllocateFrame.
 */
AllocateFrame Allocate(Array<PrimExpr> extents, DataType dtype, String storage_scope = "",
                       Optional<PrimExpr> condition = NullOpt,
                       Optional<Map<String, ObjectRef>> annotations = NullOpt);

/*!
 * \brief The allocate constant node.
 * \param data The data associated with the constant.
 * \param dtype The data type of the buffer.
 * \param extents The extents of the allocate.
 * \param annotations Additional annotation hints.
 * \return The created AllocateConstFrame.
 */
AllocateConstFrame AllocateConst(NDArray data, DataType dtype, Array<PrimExpr> extents,
                                 Optional<Map<String, ObjectRef>> annotations = NullOpt);

/*!
 * \brief Create an attribute.
 * \param node The node to annotate the attribute.
 * \param attr_key Attribute type key.
 * \param value The value of the attribute.
 * \return The result AttrFrame.
 */
AttrFrame Attr(ObjectRef node, String attr_key, PrimExpr value);

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
DeclBufferFrame DeclBuffer(Array<PrimExpr> shape, DataType dtype, String buffer_name,
                           Optional<Var> data, Optional<Array<PrimExpr>> strides,
                           Optional<PrimExpr> elem_offset, String storage_scope, int align,
                           int offset_factor, String buffer_type,
                           Optional<Array<IntImm>> axis_separators);

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
LaunchThreadFrame LaunchThread(String thread_tag, PrimExpr extent);

/*!
 * \brief Bind a var to thread env.
 * \param thread_tag The thread type tag.
 * \return The result variable which gets bound to the thread env.
 */
Var EnvThread(String thread_tag);

/*!
 * \brief Store data in a buffer.
 * \param buffer The buffer.
 * \param value The value to be stored.
 * \param indices The indices location to be stored.
 */
void BufferStore(Buffer buffer, PrimExpr value, Array<PrimExpr> indices);

/*!
 * \brief The prefetch hint for a buffer
 * \param buffer The buffer to be prefetched.
 * \param bounds The bounds to be prefetched.
 */
void Prefetch(Buffer buffer, Array<Range> bounds);

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
                  String storage_scope = "global", bool is_size_var = false,
                  bool is_unknown_type = false) {
  Type type_annotation{nullptr};
  if (is_unknown_type && storage_scope == "global") {
    type_annotation = PrimType(runtime::DataType::Handle());
  } else {
    type_annotation = PointerType(PrimType(dtype), storage_scope);
  }
  return is_size_var ? tvm::tir::SizeVar("", type_annotation) : tvm::tir::Var("", type_annotation);
}

#define TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(FuncName, DType)                                \
  inline PrimExpr FuncName(Optional<PrimExpr> expr = NullOpt, bool is_size_var = false) { \
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

TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_SIZES(Float, DataType::Float);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_SIZES(UInt, DataType::UInt);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_SIZES(Int, DataType::Int);

#define TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_LANES(FuncName, FDType, Size) \
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

TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_SIZES_LANES(Float, DataType::Float);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_SIZES_LANES(UInt, DataType::UInt);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST_SIZES_LANES(Int, DataType::Int);
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(Boolean, DataType::Bool());
TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST(Void, DataType::Void());

#undef TVM_TIR_IR_BUILDER_DEF_DTYPE_CAST

}  // namespace tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_IR_BUILDER_TIR_IR_H_
