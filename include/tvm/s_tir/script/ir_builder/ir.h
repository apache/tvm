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
#ifndef TVM_S_TIR_SCRIPT_IR_BUILDER_IR_H_
#define TVM_S_TIR_SCRIPT_IR_BUILDER_IR_H_

#include <tvm/s_tir/script/ir_builder/frame.h>
#include <tvm/tirx/script/ir_builder/ir.h>

namespace tvm {
namespace script {
namespace ir_builder {
namespace s_tir {

using tvm::tirx::BufferVar;
using tvm::tirx::Layout;
using tvm::tirx::Var;

PrimFuncFrame PrimFunc(bool is_private = false, bool persistent = false);
PrimFuncFrame DeclFunction(bool is_private = false, bool persistent = false);

/*!
 * \brief The block declaration statement.
 * \param name The name of the block.
 * \param no_realize The flag whether to construct s_tir::SBlockRealize or s_tir::SBlock.
 * \return The SBlockFrame.
 */
SBlockFrame Block(ffi::String name, bool no_realize = false, ffi::String exec_scope = "");

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
void Reads(ffi::Array<ffi::ObjectRef> buffer_slices);

/*!
 * \brief The block buffer region writing statement.
 * \param buffer_slices The array of buffer regions to write.
 */
void Writes(ffi::Array<ffi::ObjectRef> buffer_slices);

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
 * \param layout The layout of the buffer.
 * \param allocated_addr The allocated address of the buffer. Might be multi-dimensional.
 * \return The buffer attached to its enclosing block or function allocation list.
 */
BufferVar SBlockAllocBuffer(ffi::Array<PrimExpr> shape, PrimType dtype = PrimType::Float(32),
                            ffi::Optional<Expr> data = std::nullopt,
                            ffi::Array<PrimExpr> strides = {}, PrimExpr elem_offset = PrimExpr(),
                            ffi::String storage_scope = "", int align = -1, int offset_factor = 0,
                            ffi::Optional<Layout> layout = std::nullopt,
                            ffi::Array<PrimExpr> allocated_addr = {});

namespace axis {

/*!
 * \brief The spatial block axis defining function.
 * \param dom The domain of the iteration variable.
 * \param binding The binding value of the iteration variable.
 * \param dtype The data type of the iteration variable.
 * \return The iteration variable.
 */
Var Spatial(Range dom, PrimExpr binding, PrimType dtype = PrimType::Int(32));

/*!
 * \brief The reduced block axis defining function.
 * \param dom The domain of the iteration variable.
 * \param binding The binding value of the iteration variable.
 * \param dtype The data type of the iteration variable.
 * \return The iteration variable.
 */
Var Reduce(Range dom, PrimExpr binding, PrimType dtype = PrimType::Int(32));

/*!
 * \brief The scanning block axis defining function.
 * \param dom The domain of the iteration variable.
 * \param binding The binding value of the iteration variable.
 * \param dtype The data type of the iteration variable.
 * \return The iteration variable.
 */
Var Scan(Range dom, PrimExpr binding, PrimType dtype = PrimType::Int(32));

/*!
 * \brief The opaque block axis defining function.
 * \param dom The domain of the iteration variable.
 * \param binding The binding value of the iteration variable.
 * \param dtype The data type of the iteration variable.
 * \return The iteration variable.
 */
Var Opaque(Range dom, PrimExpr binding, PrimType dtype = PrimType::Int(32));

/*!
 * \brief The block axis remapping function.
 * \param kinds The types of the iteration variables.
 * \param bindings The binding values of the iteration variables.
 * \param dtype The data types of the iteration variables.
 * \return The iteration variables.
 */
ffi::Array<Var> Remap(ffi::String kinds, ffi::Array<PrimExpr> bindings,
                      PrimType dtype = PrimType::Int(32));

}  // namespace axis

}  // namespace s_tir
}  // namespace ir_builder
}  // namespace script
}  // namespace tvm

#endif  // TVM_S_TIR_SCRIPT_IR_BUILDER_IR_H_
