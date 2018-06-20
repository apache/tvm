/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Placeholder op.
 * \file placeholder_op.cc
 */
#include <tvm/operation.h>

namespace tvm {

// PlaceholderOpNode
TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<PlaceholderOpNode>([](const PlaceholderOpNode *op, IRPrinter *p) {
    p->stream << "placeholder(" << op->name << ", " << op << ")";
});

TVM_REGISTER_NODE_TYPE(PlaceholderOpNode);

int PlaceholderOpNode::num_outputs() const {
  return 1;
}

Array<IterVar> PlaceholderOpNode::root_iter_vars() const {
  return {};
}

Type PlaceholderOpNode::output_dtype(size_t i) const {
  CHECK_EQ(i, 0U);
  return dtype;
}

Array<Expr> PlaceholderOpNode::output_shape(size_t i) const {
  CHECK_EQ(i, 0U);
  return shape;
}

Operation PlaceholderOpNode::make(std::string name,
                                  Array<Expr> shape,
                                  Type dtype) {
  auto n = std::make_shared<PlaceholderOpNode>();
  n->name = name;
  n->shape = shape;
  n->dtype = dtype;
  return Operation(n);
}

Tensor placeholder(Array<Expr> shape, Type dtype, std::string name) {
  return PlaceholderOpNode::make(name, shape, dtype).output(0);
}

Array<Tensor> PlaceholderOpNode::InputTensors() const {
  return {};
}

Operation PlaceholderOpNode::ReplaceInputs(
    const Operation& self,
    const std::unordered_map<Tensor, Tensor>& rmap) const {
  return self;
}

void PlaceholderOpNode::PropBoundToInputs(
    const Operation& self,
    const std::unordered_map<const Variable*, IntSet>& dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
}

void PlaceholderOpNode::GatherBound(
    const Operation& self,
    const std::unordered_map<Tensor, TensorDom>& tensor_dom,
    std::unordered_map<IterVar, Range>* out_dom_map) const {
}

Stmt PlaceholderOpNode::BuildRealize(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& realize_map,
    const Stmt& body) const {
  return body;
}

Stmt PlaceholderOpNode::BuildProvide(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool debug_keep_trivial_loop) const {
  return Stmt();
}
}  // namespace tvm
