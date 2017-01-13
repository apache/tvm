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

Buffer::Buffer(Array<Expr> shape,
               Type dtype,
               std::string name)
    : Buffer(BufferNode::make(
          name,
          Var(name, Type(Type::Handle, 0, 0)),
          shape, Array<Expr>(), dtype)) {
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
  return base;
}

Expr Buffer::MakeLoad(Array<Expr> index) const {
  const BufferNode* n = operator->();
  return ir::Load::make(n->dtype, n->ptr, BufferOffset(n, index));
}

Stmt Buffer::MakeStore(Array<Expr> index, Expr value) const {
  const BufferNode* n = operator->();
  CHECK_EQ(value.type(), n->dtype);
  return ir::Store::make(n->ptr, BufferOffset(n, index), value);
}

Buffer BufferNode::make(std::string name,
                        Var ptr,
                        Array<Expr> shape,
                        Array<Expr> strides,
                        Type dtype) {
  auto n = std::make_shared<BufferNode>();
  n->name = name;
  n->ptr = ptr;
  n->shape = shape;
  n->strides = strides;
  n->dtype = dtype;
  return Buffer(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<BufferNode>([](const BufferNode *op, IRPrinter *p) {
    p->stream << "buffer(" << op->name << ", " << op << ")";
});

TVM_REGISTER_NODE_TYPE(BufferNode);

}  // namespace tvm
