/*!
*  Copyright (c) 2017 by Contributors
* \file detail/extern.h
* \brief Helpers for using external functions
*/
#ifndef TOPI_DETAIL_EXTERN_H_
#define TOPI_DETAIL_EXTERN_H_

#include "tvm/tvm.h"

namespace topi {
using namespace tvm;

Buffer DeclExternBuffer(Array<Expr> shape,
  Type dtype,
  std::string name) {
  auto data = var(name, Handle());
  auto elem_offset = Expr();
  return BufferNode::make(data, dtype, shape, Array<Expr>(), elem_offset, name, "",
    -1, 0);
}

using FExtern = std::function<Expr(Array<Buffer>, Array<Buffer>)>;

Array<Tensor> make_extern(const Array<Array<Expr>>& out_shapes,
  const std::vector<Type>& out_types,
  const Array<Tensor>& inputs,
  FExtern fextern,
  std::string name,
  std::string tag) {
  CHECK_EQ(out_shapes.size(), out_types.size())
    << "make_extern: out_shapes and out_types must have equal size";

  Array<Buffer> input_placeholders;
  for (auto t : inputs) {
    input_placeholders.push_back(DeclExternBuffer(t->shape, t->dtype, t->op->name));
  }
  Array<Buffer> output_placeholders;
  for (int i = 0; i < out_shapes.size(); ++i) {
    output_placeholders.push_back(DeclExternBuffer(out_shapes[i], out_types[i], name));
  }

  auto body = fextern(input_placeholders, output_placeholders);
  auto body_stmt = tvm::ir::Evaluate::make(body);

  auto op = ExternOpNode::make(
    name, tag, inputs, input_placeholders, output_placeholders, body_stmt);

  Array<Tensor> outputs;
  for (int i = 0; i < output_placeholders.size(); ++i) {
    outputs.push_back(op.output(i));
  }
  return outputs;
}

Expr pack_buffer(Buffer buf) {
  CHECK_GT(buf->shape.size(), 0) << "buf shape must have at least one element";
  auto shape = tvm::ir::Call::make(Handle(), tvm::ir::intrinsic::tvm_stack_make_shape,
    buf->shape, tvm::ir::Call::CallType::Intrinsic);
  Expr strides;
  if (buf->strides.size() > 0) {
    strides = tvm::ir::Call::make(Handle(), tvm::ir::intrinsic::tvm_stack_make_shape,
      buf->shape, tvm::ir::Call::CallType::Intrinsic);
  } else {
    strides = 0;
  }
  Array<Expr> pack_args{
    buf->data,
    shape,
    strides,
    make_const(Int(32), buf->shape.size()),
    make_const(buf->dtype, 0),
    buf->elem_offset
  };
  return tvm::ir::Call::make(Handle(), tvm::ir::intrinsic::tvm_stack_make_array,
    pack_args, tvm::ir::Call::CallType::Intrinsic);
}

Expr call_packed(Array<Expr> args) {
  return tvm::ir::Call::make(Int(32), tvm::ir::intrinsic::tvm_call_packed,
    args, tvm::ir::Call::CallType::Intrinsic);
}

}  // namespace topi
#endif  // TOPI_DETAIL_EXTERN_H_
