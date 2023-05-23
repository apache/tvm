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
 * \file detail/extern.h
 * \brief Helpers for using external functions
 */
#ifndef TVM_TOPI_DETAIL_EXTERN_H_
#define TVM_TOPI_DETAIL_EXTERN_H_

#include <tvm/te/operation.h>
#include <tvm/tir/builtin.h>

#include <string>
#include <vector>

namespace tvm {
namespace topi {
namespace detail {

using namespace tvm::te;

/*!
 * \brief Construct a buffer to pass to an external function
 *
 * \param shape The shape of the buffer
 * \param dtype The type of the buffer elements
 * \param name The name of the buffer
 *
 * \return The Buffer object
 */
inline Buffer DeclExternBuffer(Array<PrimExpr> shape, DataType dtype, std::string name) {
  auto data = var(name, DataType::Handle());
  auto elem_offset = PrimExpr();
  return Buffer(data, dtype, shape, Array<PrimExpr>(), elem_offset, name, -1, 0, kDefault);
}

/*!
 * \brief A function which constructs an Expr representing the invocation of an external
 * function. The function expects two arguments: an array of Buffers holding the input
 * tensor values, and a pre-allocated array of Buffers to be filled with the outputs.
 */
using FExtern = std::function<PrimExpr(Array<Buffer>, Array<Buffer>)>;

/*!
 * \brief Create tensors representing the result of invoking an external function.
 * This function will create the necessary buffers to hold input and output tensor values.
 *
 * \param out_shapes An array where each element is the shape of the corresponding output tensor.
 * \param out_types An array where each element is the dtype of the corresponding output tensor.
 * \param inputs An array of input Tensors
 * \param fextern A function that constructs an Expr representing the invocation of
 * the external function given the input and output buffers.
 * \param name The name of the operation
 * \param tag The tag to mark the operation
 * \param attrs The additional auxiliary attributes of the operation.
 *
 * \return An array of Tensors representing the outputs of the function invocation. There will
 * be one output Tensor for each element of out_shapes, with dtype equal to the corresponding
 * element of out_types.
 */
inline Array<Tensor> make_extern(const Array<Array<PrimExpr>>& out_shapes,
                                 const std::vector<DataType>& out_types,
                                 const Array<Tensor>& inputs, FExtern fextern, std::string name,
                                 std::string tag, ::tvm::Map<String, ObjectRef> attrs) {
  ICHECK_EQ(out_shapes.size(), out_types.size())
      << "make_extern: out_shapes and out_types must have equal size";

  Array<Buffer> input_placeholders;
  for (auto t : inputs) {
    input_placeholders.push_back(DeclExternBuffer(t->shape, t->dtype, t->op->name));
  }
  Array<Buffer> output_placeholders;
  for (size_t i = 0; i < out_shapes.size(); ++i) {
    output_placeholders.push_back(DeclExternBuffer(out_shapes[i], out_types[i], name));
  }

  auto body = fextern(input_placeholders, output_placeholders);
  auto body_stmt = tvm::tir::Evaluate(body);

  auto op = ExternOp(name, tag, attrs, inputs, input_placeholders, output_placeholders, body_stmt);

  Array<Tensor> outputs;
  for (size_t i = 0; i < output_placeholders.size(); ++i) {
    outputs.push_back(op.output(i));
  }
  return outputs;
}

/*!
 * \brief This function is used to create a DLTensor structure on the stack to
 * be able to pass a symbolic buffer as arguments to TVM PackedFunc
 *
 * \param buf The buffer to pack
 *
 * \return An expression representing the pack operation
 */
inline PrimExpr pack_buffer(Buffer buf) {
  ICHECK_GT(buf->shape.size(), 0) << "buf shape must have at least one element";
  auto shape =
      tvm::tir::Call(DataType::Handle(), tvm::tir::builtin::tvm_stack_make_shape(), buf->shape);
  PrimExpr strides;
  if (buf->strides.size() > 0) {
    strides =
        tvm::tir::Call(DataType::Handle(), tvm::tir::builtin::tvm_stack_make_shape(), buf->strides);
  } else {
    strides = 0;
  }
  Array<PrimExpr> pack_args{buf->data,
                            shape,
                            strides,
                            make_const(DataType::Int(32), static_cast<int64_t>(buf->shape.size())),
                            make_const(buf->dtype, 0),
                            buf->elem_offset};
  return tvm::tir::Call(DataType::Handle(), tvm::tir::builtin::tvm_stack_make_array(), pack_args);
}

/*!
 * \brief Construct an Expr representing the invocation of a PackedFunc
 *
 * \param args An array containing the registered name of the PackedFunc followed
 * by the arguments to pass to the PackedFunc when called. The first element of the
 * array must be a constant string expression.
 *
 * \return An expression representing the invocation
 */
inline PrimExpr call_packed(Array<PrimExpr> args) {
  return tvm::tir::Call(DataType::Int(32), tvm::tir::builtin::tvm_call_packed(), args);
}

}  // namespace detail
}  // namespace topi
}  // namespace tvm
#endif  // TVM_TOPI_DETAIL_EXTERN_H_
