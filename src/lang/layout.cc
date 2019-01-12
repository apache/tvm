/*!
 *  Copyright (c) 2018 by Contributors
 * \file src/lang/layout.cc
 * \brief Layout expression.
 */
#include <tvm/layout.h>
#include <tvm/ir_pass.h>

namespace tvm {

TVM_REGISTER_NODE_TYPE(LayoutNode);
TVM_REGISTER_NODE_TYPE(BijectiveLayoutNode);

inline Array<Expr> TransformIndex(const Array<Expr>& src_index,
                                  const Array<IterVar>& src_axis,
                                  const Array<Expr>& transform_rule) {
  Array<Expr> result;
  std::unordered_map<const Variable*, Expr> bind_map;
  for (size_t i = 0; i < src_index.size(); ++i) {
    bind_map[src_axis[i]->var.get()] = src_index[i];
  }
  for (Expr rule : transform_rule) {
    result.push_back(ir::Simplify(ir::Substitute(rule, bind_map)));
  }
  return result;
}

Array<Expr> BijectiveLayout::ForwardIndex(const Array<Expr>& orig_index) const {
  const BijectiveLayoutNode* self = operator->();
  CHECK_EQ(orig_index.size(), self->orig_axis.size())
    << "Input mismatch with layout " << self->orig_layout;
  return TransformIndex(orig_index, self->orig_axis, self->forward_rule);
}


Array<Expr> BijectiveLayout::BackwardIndex(const Array<Expr>& store_index) const {
  const BijectiveLayoutNode* self = operator->();
  CHECK_EQ(store_index.size(), self->store_axis.size())
    << "Output mismatch with layout " << self->store_layout;
  return TransformIndex(store_index, self->store_axis, self->backward_rule);
}

inline Array<Expr> TransformShape(const Array<Expr>& src_shape,
                                  const Array<IterVar>& src_axis,
                                  const Array<IterVar>& target_axis,
                                  const Array<Expr>& transform_rule) {
  CHECK_EQ(src_shape.size(), src_axis.size());
  // bind variables for original axes
  // for major-axis, bind the corresponding size
  // for minor-axis, simply bind it as 0, so that we can reuse forward/backward_rule,
  // e.g., (C * 16 + c) / 32
  std::unordered_map<const Variable*, Expr> bind_map;
  for (size_t i = 0; i < src_shape.size(); ++i) {
    Expr orig_shape = src_shape[i];
    IterVar orig_axis = src_axis[i];
    if (!BijectiveLayoutNode::IsMajorAxis(orig_axis)) {
      /* TODO
      if (orig_shape.defined()) {
        CHECK_EQ(orig_shape, orig_axis->dom->extent) << "Input shape mismatch at index " << i
                                                     << ". Expected " << orig_axis->dom->extent
                                                     << ", get " << orig_shape;
      }
      */
      bind_map[orig_axis->var.get()] = Expr(0);
    } else {
      bind_map[orig_axis->var.get()] = orig_shape;
    }
  }
  // infer the target shape,
  // for major-axis, use the forward/backward_rule directly,
  // for minor-axis, simply use the extent.
  Array<Expr> result;
  CHECK_EQ(transform_rule.size(), target_axis.size());
  for (size_t i = 0; i < transform_rule.size(); ++i) {
    Expr rule = transform_rule[i];
    IterVar axis = target_axis[i];
    if (!BijectiveLayoutNode::IsMajorAxis(axis)) {
      result.push_back(axis->dom->extent);
    } else {
      result.push_back(ir::Simplify(ir::Substitute(rule, bind_map)));
    }
  }
  return result;
}

Array<Expr> BijectiveLayout::ForwardShape(const Array<Expr>& shape) const {
  const BijectiveLayoutNode* self = operator->();
  return TransformShape(shape, self->orig_axis, self->store_axis, self->forward_rule);
}

Array<Expr> BijectiveLayout::BackwardShape(const Array<Expr>& shape) const {
  const BijectiveLayoutNode* self = operator->();
  return TransformShape(shape, self->store_axis, self->orig_axis, self->backward_rule);
}

BijectiveLayout BijectiveLayoutNode::make(const std::string& orig_layout,
                                const std::string& store_layout) {
  auto n = make_node<BijectiveLayoutNode>();

  auto LayoutParser = [](const std::string& layout, Array<IterVar>& axes) {
    std::vector<int32_t> axis_factor(256, -1);
    int32_t factor = 0;
    for (size_t i = 0; i < layout.size(); ++i) {
      const char axis_name = layout.at(i);
      if (axis_name >= 'A' && axis_name <= 'Z') {
        CHECK_EQ(axis_factor[axis_name], -1) << "Invalid layout " << layout
                                             << ": duplicated axis " << axis_name;
        CHECK_EQ(factor, 0) << "Invalid layout " << layout
                            << ": invalid factor size " << factor
                            << " before dimension " << axis_name;
        const std::string shape_name(std::string(1, axis_name) + "_shape");
        IterVar axis = IterVarNode::make(Range(Expr(0), Var(shape_name)),
                                         Var(std::string(1, axis_name)), kDataPar);
        axes.push_back(axis);
        axis_factor[axis_name] = 0;
      } else if (axis_name >= 'a' && axis_name <= 'z') {
        CHECK_EQ(axis_factor[axis_name], -1) << "Invalid layout " << layout
                                             << ": duplicated axis " << axis_name;
        CHECK_GT(factor, 0) << "Invalid layout " << layout << ": invalid factor size "
                            << factor << " for dimension " << axis_name;
        IterVar axis = IterVarNode::make(Range(Expr(0), Expr(factor)),
                                         Var(std::string(1, axis_name)), kDataPar);
        axes.push_back(axis);
        axis_factor[axis_name] = factor;
        factor = 0;
      } else if (axis_name >= '0' && axis_name <= '9') {
        CHECK(factor >= 0) << "Invalid layout " << layout << ": _ is adjacent to a number.";
        factor = factor * 10 + axis_name - '0';
      } else {
        LOG(FATAL) << "Invalid layout " << layout;
      }
    }
  };

  n->orig_layout = orig_layout;
  LayoutParser(orig_layout, n->orig_axis);

  n->store_layout = store_layout;
  LayoutParser(store_layout, n->store_axis);

  if (!GetStoreRule(n->forward_rule, n->orig_axis, n->store_axis)) {
    // not convertible
    return BijectiveLayout();
  }
  CHECK(GetStoreRule(n->backward_rule, n->store_axis, n->orig_axis));

  return BijectiveLayout(n);
}

}  // namespace tvm
