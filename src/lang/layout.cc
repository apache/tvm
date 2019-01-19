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

const LayoutAxis LayoutAxis::A = LayoutAxis('A');
const LayoutAxis LayoutAxis::B = LayoutAxis('B');
const LayoutAxis LayoutAxis::C = LayoutAxis('C');
const LayoutAxis LayoutAxis::D = LayoutAxis('D');
const LayoutAxis LayoutAxis::E = LayoutAxis('E');
const LayoutAxis LayoutAxis::F = LayoutAxis('F');
const LayoutAxis LayoutAxis::G = LayoutAxis('G');
const LayoutAxis LayoutAxis::H = LayoutAxis('H');
const LayoutAxis LayoutAxis::I = LayoutAxis('I');
const LayoutAxis LayoutAxis::J = LayoutAxis('J');
const LayoutAxis LayoutAxis::K = LayoutAxis('K');
const LayoutAxis LayoutAxis::L = LayoutAxis('L');
const LayoutAxis LayoutAxis::M = LayoutAxis('M');
const LayoutAxis LayoutAxis::N = LayoutAxis('N');
const LayoutAxis LayoutAxis::O = LayoutAxis('O');
const LayoutAxis LayoutAxis::P = LayoutAxis('P');
const LayoutAxis LayoutAxis::Q = LayoutAxis('Q');
const LayoutAxis LayoutAxis::R = LayoutAxis('R');
const LayoutAxis LayoutAxis::S = LayoutAxis('S');
const LayoutAxis LayoutAxis::T = LayoutAxis('T');
const LayoutAxis LayoutAxis::U = LayoutAxis('U');
const LayoutAxis LayoutAxis::V = LayoutAxis('V');
const LayoutAxis LayoutAxis::W = LayoutAxis('W');
const LayoutAxis LayoutAxis::X = LayoutAxis('X');
const LayoutAxis LayoutAxis::Y = LayoutAxis('Y');
const LayoutAxis LayoutAxis::Z = LayoutAxis('Z');
const LayoutAxis LayoutAxis::a = LayoutAxis('a');
const LayoutAxis LayoutAxis::b = LayoutAxis('b');
const LayoutAxis LayoutAxis::c = LayoutAxis('c');
const LayoutAxis LayoutAxis::d = LayoutAxis('d');
const LayoutAxis LayoutAxis::e = LayoutAxis('e');
const LayoutAxis LayoutAxis::f = LayoutAxis('f');
const LayoutAxis LayoutAxis::g = LayoutAxis('g');
const LayoutAxis LayoutAxis::h = LayoutAxis('h');
const LayoutAxis LayoutAxis::i = LayoutAxis('i');
const LayoutAxis LayoutAxis::j = LayoutAxis('j');
const LayoutAxis LayoutAxis::k = LayoutAxis('k');
const LayoutAxis LayoutAxis::l = LayoutAxis('l');
const LayoutAxis LayoutAxis::m = LayoutAxis('m');
const LayoutAxis LayoutAxis::n = LayoutAxis('n');
const LayoutAxis LayoutAxis::o = LayoutAxis('o');
const LayoutAxis LayoutAxis::p = LayoutAxis('p');
const LayoutAxis LayoutAxis::q = LayoutAxis('q');
const LayoutAxis LayoutAxis::r = LayoutAxis('r');
const LayoutAxis LayoutAxis::s = LayoutAxis('s');
const LayoutAxis LayoutAxis::t = LayoutAxis('t');
const LayoutAxis LayoutAxis::u = LayoutAxis('u');
const LayoutAxis LayoutAxis::v = LayoutAxis('v');
const LayoutAxis LayoutAxis::w = LayoutAxis('w');
const LayoutAxis LayoutAxis::x = LayoutAxis('x');
const LayoutAxis LayoutAxis::y = LayoutAxis('y');
const LayoutAxis LayoutAxis::z = LayoutAxis('z');

Layout LayoutNode::make(const std::string& layout) {
  return Layout(layout);
}

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
    if (!LayoutAxis::Get(orig_axis).IsPrimal()) {
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
    if (!LayoutAxis::Get(axis).IsPrimal()) {
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
  return BijectiveLayout(Layout(orig_layout), Layout(store_layout));
}

}  // namespace tvm
