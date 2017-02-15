/*!
 *  Copyright (c) 2016 by Contributors
 * \file operation.cc
 */
#include <tvm/operation.h>
#include <tvm/tensor.h>
#include <tvm/ir.h>
#include <tvm/ir_pass.h>
#include <memory>

namespace tvm {

Tensor Operation::output(size_t i) const {
  auto node = std::make_shared<TensorNode>();
  node->op = *this;
  node->value_index = 0;
  node->dtype = (*this)->output_dtype(i);
  node->shape = (*this)->output_shape(i);
  return Tensor(node);
}

// PlaceholderOpNode
TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<PlaceholderOpNode>([](const PlaceholderOpNode *op, IRPrinter *p) {
    p->stream << "placeholder(" << op->name << ", " << op << ")";
});

TVM_REGISTER_NODE_TYPE(PlaceholderOpNode);

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

// ComputeOpNode
Array<IterVar> ComputeOpNode::root_iter_vars() const {
  if (reduce_axis.size() == 0) return axis;
  Array<IterVar> ret = axis;
  for (IterVar iv : reduce_axis) {
    ret.push_back(iv);
  }
  return ret;
}

Type ComputeOpNode::output_dtype(size_t i) const {
  CHECK_EQ(i, 0U);
  return body.type();
}

Array<Expr> ComputeOpNode::output_shape(size_t i) const {
  CHECK_EQ(i, 0U);
  std::vector<Expr> shape;
  for (size_t i = 0; i < axis.size(); ++i) {
    const Range& r = axis[i]->dom;
    shape.push_back(r->extent);
  }
  return Array<Expr>(shape);
}

Tensor compute(Array<Expr> shape, FCompute fcompute, std::string name) {
  auto op_node = std::make_shared<ComputeOpNode>();
  // compute dimension.
  size_t ndim = shape.size();
  std::vector<IterVar> axis;
  std::vector<Var> args;
  for (size_t i = 0; i < ndim; ++i) {
    std::ostringstream os;
    os << "ax" << i;
    axis.emplace_back(IterVar(Range(0, shape[i]), os.str()));
    args.push_back(axis.back()->var);
  }

  op_node->axis = Array<IterVar>(axis);
  op_node->body = fcompute(args);
  op_node->name = name;
  return Operation(op_node).output(0);
}

Operation ComputeOpNode::make(std::string name,
                              Array<IterVar> axis,
                              Expr body) {
  auto n = std::make_shared<ComputeOpNode>();
  n->name = name;
  n->axis = axis;
  n->body = body;
  if (n->body->is_type<ir::Reduce>()) {
    n->reduce_axis = n->body.as<ir::Reduce>()->axis;
  }
  return Operation(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<ComputeOpNode>([](const ComputeOpNode *op, IRPrinter *p) {
    p->stream << "compute(" << op->name << ", " << op << ")";
});

TVM_REGISTER_NODE_TYPE(ComputeOpNode);

// Scan
inline bool prove_equal(Expr lhs, Expr rhs) {
  return is_zero(ir::Simplify(lhs - rhs));
}

int ScanOpNode::num_outputs() const {
  return update.size();
}
Array<IterVar> ScanOpNode::root_iter_vars() const {
  return Array<IterVar>{scan_axis};
}

Type ScanOpNode::output_dtype(size_t i) const {
  return update[i]->dtype;
}

Array<Expr> ScanOpNode::output_shape(size_t i) const {
  CHECK_LT(i, state_placeholder.size());
  return state_placeholder[i]->shape;
}

Operation ScanOpNode::make(std::string name,
                           IterVar axis,
                           Array<Tensor> init,
                           Array<Tensor> update,
                           Array<Tensor> state_placeholder) {
  auto n = std::make_shared<ScanOpNode>();
  CHECK_EQ(init.size(), update.size());
  CHECK_EQ(init.size(), state_placeholder.size());

  for (size_t i = 0; i < init.size(); ++i) {
    CHECK_EQ(init[i]->dtype, state_placeholder[i]->dtype);
    CHECK_EQ(init[i]->dtype, update[i]->dtype);
    CHECK(can_prove(init[i]->shape[0] == axis->dom->min))
        << "init.shape[0] need to match scan_axis.dom.min";
    CHECK(prove_equal(
        state_placeholder[i]->shape[0], axis->dom->min + axis->dom->extent))
        << "shate_placeholder.shape[0] need to match"
        << " scan_axis.dom.min + scan_axis.dom.extent";
    CHECK_EQ(state_placeholder[i].ndim(), init[i].ndim())
        << "The dimension of init need to match state_placeholder";
    CHECK_EQ(update[i].ndim() + 1, state_placeholder[i].ndim())
        << "The update.ndim need to be state_placeholder.ndim - 1";
    for (size_t k = 0;  k < update[i].ndim(); ++k) {
      CHECK(prove_equal(
          update[i]->shape[k], state_placeholder[i]->shape[k + 1]));
      // setup spatial axis
      std::ostringstream spatial_name;
      spatial_name << name << ".out" << i << ".i" << k + 1;
      n->spatial_axis_.push_back(
          IterVar(Range::make_with_min_extent(0, update[i]->shape[k]),
                  spatial_name.str()));
    }
    for (size_t k = 1;  k < init[i].ndim(); ++k) {
      CHECK(prove_equal(
          init[i]->shape[k], state_placeholder[i]->shape[k]));
    }
  }

  n->name = name;
  n->scan_axis = axis;
  n->init = init;
  n->update = update;
  n->state_placeholder = state_placeholder;
  return Operation(n);
}

Array<Tensor> scan(IterVar scan_axis,
                   Array<Tensor> init,
                   Array<Tensor> update,
                   Array<Tensor> state_placeholder,
                   std::string name) {
  Operation op = ScanOpNode::make(
      name, scan_axis, init, update, state_placeholder);
  Array<Tensor> res;
  for (int i = 0; i < op->num_outputs(); ++i) {
    res.push_back(op.output(i));
  }
  return res;
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<ScanOpNode>([](const ScanOpNode *op, IRPrinter *p) {
    p->stream << "scan(" << op->name << ", " << op << ")";
});

}  // namespace tvm
