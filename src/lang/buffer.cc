/*!
 *  Copyright (c) 2016 by Contributors
 * \file buffer.cc
 */
#include <tvm/buffer.h>
#include <tvm/runtime/device_api.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>

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
      name,
      "",
      0, 0);
}

// The buffer offset in convention of number of elements of
// original data ignoring number of lanes.
inline Expr ElemOffset(const BufferNode* n, Array<Expr> index) {
  Expr base = n->elem_offset;
  if (n->strides.size() == 0) {
    CHECK_EQ(n->shape.size(), index.size());
    if (is_zero(base)) {
      base = index[0];
    } else {
      base = base + index[0];
    }
    for (size_t i = 1; i < index.size(); ++i) {
      base = base * n->shape[i] + index[i];
    }
  } else {
    CHECK_EQ(n->strides.size(), index.size());
    if (is_zero(base)) {
      base = index[0] * n->strides[0];
    } else {
      base = base + index[0] * n->strides[0];
    }
    for (size_t i = 1; i < index.size(); ++i) {
      base = base + index[i] * n->strides[i];
    }
  }
  return base;
}

// Buffer access offset.
inline Expr BufferOffset(const BufferNode* n, Array<Expr> index) {
  Expr offset = ElemOffset(n, index);
  if (n->dtype.lanes() != 1) {
    offset = offset * make_const(offset.type(), n->dtype.lanes());
  }
  return offset;
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

Buffer Buffer::MakeStrideView() const {
  if ((*this)->strides.size() != 0) return *this;
  std::vector<Expr> temp;
  auto n = std::make_shared<BufferNode>(*operator->());
  Expr acc = make_const(n->shape[0].type(), 1);
  for (size_t i = n->shape.size(); i != 0 ; --i) {
    temp.push_back(acc);
    acc = acc * n->shape[i - 1];
  }
  for (size_t i = temp.size(); i != 0; --i) {
    n->strides.push_back(temp[i - 1]);
  }
  return Buffer(n);
}

Buffer Buffer::MakeSlice(Array<Expr> begins, Array<Expr> extents) const {
  const BufferNode* n = operator->();
  Expr elem_offset = ElemOffset(n, begins);
  Array<Expr> strides = n->strides;
  if (strides.size() == 0) {
    bool can_relax = true;
    bool need_stride = false;
    // check if stride is needed.
    for (size_t i = 0; i < extents.size(); ++i) {
      if (!can_relax) {
        if (!is_zero(begins[i]) ||
            !is_zero(ir::Simplify(extents[i] - n->shape[i]))) {
          need_stride = true;
        }
      }
      if (!is_one(extents[i])) can_relax = false;
    }
    // make stride.
    if (need_stride) {
      return MakeStrideView().MakeSlice(begins, extents);
    }
  }
  return BufferNode::make(n->data,
                          n->dtype,
                          extents,
                          strides,
                          elem_offset,
                          n->name + "_slice",
                          n->scope,
                          n->data_alignment,
                          0);
}

Buffer BufferNode::make(Var data,
                        Type dtype,
                        Array<Expr> shape,
                        Array<Expr> strides,
                        Expr elem_offset,
                        std::string name,
                        std::string scope,
                        int data_alignment,
                        int offset_factor) {
  auto n = std::make_shared<BufferNode>();
  n->data = std::move(data);
  n->dtype = dtype;
  n->shape = std::move(shape);
  n->strides = std::move(strides);
  n->name = std::move(name);
  if (scope.length() == 0) {
    scope = "global";
  }
  n->scope = std::move(scope);
  if (!elem_offset.defined()) {
    elem_offset = make_const(n->shape[0].type(), 0);
  }
  if (data_alignment == 0) {
    data_alignment = runtime::kAllocAlignment;
  }
  if (offset_factor == 0) {
    offset_factor = 1;
  }
  n->elem_offset = std::move(elem_offset);
  n->data_alignment = data_alignment;
  n->offset_factor = offset_factor;
  return Buffer(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<BufferNode>([](const BufferNode *op, IRPrinter *p) {
    p->stream << "buffer(" << op->name << ", " << op << ")";
});

TVM_REGISTER_NODE_TYPE(BufferNode);

}  // namespace tvm
