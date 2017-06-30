/*!
 *  Copyright (c) 2016 by Contributors
 * \file buffer.cc
 */
#include <tvm/buffer.h>
#include <tvm/ir.h>

namespace tvm {

Array<Expr> GetStrides(Array<Expr> shape) {
  CHECK_NE(shape.size(), 0U);
  std::vector<Expr> vec{make_const(shape[0].type(), 1)};
  for (size_t i = shape.size() - 1; i != 0; --i) {
    vec.push_back(shape[i - 1] * vec.back());
  }
  return Array<Expr>(vec.rbegin(), vec.rend());
}

Buffer decl_buffer(Array<Expr> shape,
                   Type dtype,
                   std::string name) {
  return BufferNode::make(
      Var(name, Handle()),
      dtype,
      shape,
      Array<Expr>(),
      Expr(),
      name, "", 0);
}

inline Expr BufferOffset(const BufferNode* n, Array<Expr> index) {
  Expr base;
  if (n->strides.size() == 0) {
    CHECK_EQ(n->shape.size(), index.size());
    base = index[0];
    for (size_t i = 1; i < index.size(); ++i) {
      base = base * n->shape[i] + index[i];
    }
  } else {
    CHECK_EQ(n->strides.size(), index.size());
    base = index[0] * n->strides[0];
    for (size_t i = 1; i < index.size(); ++i) {
      base = base + index[i] * n->strides[i];
    }
  }
  if (!is_zero(n->byte_offset)) {
    base = base + (n->byte_offset / n->dtype.bytes());
  }
  return base;
}

Expr Buffer::MakeLoad(Array<Expr> index) const {
  const BufferNode* n = operator->();
  return ir::Load::make(
      n->dtype, n->data, BufferOffset(n, index),
      const_true(n->dtype.lanes()));
}

Stmt Buffer::MakeStore(Array<Expr> index, Expr value) const {
  const BufferNode* n = operator->();
  CHECK_EQ(value.type(), n->dtype);
  return ir::Store::make(n->data, value, BufferOffset(n, index),
                         const_true(n->dtype.lanes()));
}

Buffer BufferNode::make(Var data,
                        Type dtype,
                        Array<Expr> shape,
                        Array<Expr> strides,
                        Expr byte_offset,
                        std::string name,
                        std::string scope,
                        int offset_alignment) {
  auto n = std::make_shared<BufferNode>();
  n->data = std::move(data);
  n->dtype = dtype;
  n->shape = std::move(shape);
  n->strides = std::move(strides);
  n->name = std::move(name);
  n->scope = std::move(scope);
  if (!byte_offset.defined()) {
    byte_offset = make_const(n->shape[0].type(), 0);
  }
  if (offset_alignment != 0) {
    CHECK_EQ(offset_alignment % dtype.bytes(), 0)
        << "Offset alignments must be at least " << dtype.bytes();
  } else {
    offset_alignment = dtype.bytes();
  }
  n->byte_offset = byte_offset;
  n->offset_alignment = offset_alignment;
  return Buffer(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<BufferNode>([](const BufferNode *op, IRPrinter *p) {
    p->stream << "buffer(" << op->name << ", " << op << ")";
});

TVM_REGISTER_NODE_TYPE(BufferNode);

}  // namespace tvm
