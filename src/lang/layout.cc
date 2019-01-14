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

const LayoutAxis Layout::A = LayoutAxis('A');
const LayoutAxis Layout::B = LayoutAxis('B');
const LayoutAxis Layout::C = LayoutAxis('C');
const LayoutAxis Layout::D = LayoutAxis('D');
const LayoutAxis Layout::E = LayoutAxis('E');
const LayoutAxis Layout::F = LayoutAxis('F');
const LayoutAxis Layout::G = LayoutAxis('G');
const LayoutAxis Layout::H = LayoutAxis('H');
const LayoutAxis Layout::I = LayoutAxis('I');
const LayoutAxis Layout::J = LayoutAxis('J');
const LayoutAxis Layout::K = LayoutAxis('K');
const LayoutAxis Layout::L = LayoutAxis('L');
const LayoutAxis Layout::M = LayoutAxis('M');
const LayoutAxis Layout::N = LayoutAxis('N');
const LayoutAxis Layout::O = LayoutAxis('O');
const LayoutAxis Layout::P = LayoutAxis('P');
const LayoutAxis Layout::Q = LayoutAxis('Q');
const LayoutAxis Layout::R = LayoutAxis('R');
const LayoutAxis Layout::S = LayoutAxis('S');
const LayoutAxis Layout::T = LayoutAxis('T');
const LayoutAxis Layout::U = LayoutAxis('U');
const LayoutAxis Layout::V = LayoutAxis('V');
const LayoutAxis Layout::W = LayoutAxis('W');
const LayoutAxis Layout::X = LayoutAxis('X');
const LayoutAxis Layout::Y = LayoutAxis('Y');
const LayoutAxis Layout::Z = LayoutAxis('Z');
const LayoutAxis Layout::a = LayoutAxis('a');
const LayoutAxis Layout::b = LayoutAxis('b');
const LayoutAxis Layout::c = LayoutAxis('c');
const LayoutAxis Layout::d = LayoutAxis('d');
const LayoutAxis Layout::e = LayoutAxis('e');
const LayoutAxis Layout::f = LayoutAxis('f');
const LayoutAxis Layout::g = LayoutAxis('g');
const LayoutAxis Layout::h = LayoutAxis('h');
const LayoutAxis Layout::i = LayoutAxis('i');
const LayoutAxis Layout::j = LayoutAxis('j');
const LayoutAxis Layout::k = LayoutAxis('k');
const LayoutAxis Layout::l = LayoutAxis('l');
const LayoutAxis Layout::m = LayoutAxis('m');
const LayoutAxis Layout::n = LayoutAxis('n');
const LayoutAxis Layout::o = LayoutAxis('o');
const LayoutAxis Layout::p = LayoutAxis('p');
const LayoutAxis Layout::q = LayoutAxis('q');
const LayoutAxis Layout::r = LayoutAxis('r');
const LayoutAxis Layout::s = LayoutAxis('s');
const LayoutAxis Layout::t = LayoutAxis('t');
const LayoutAxis Layout::u = LayoutAxis('u');
const LayoutAxis Layout::v = LayoutAxis('v');
const LayoutAxis Layout::w = LayoutAxis('w');
const LayoutAxis Layout::x = LayoutAxis('x');
const LayoutAxis Layout::y = LayoutAxis('y');
const LayoutAxis Layout::z = LayoutAxis('z');

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

BijectiveLayout BijectiveLayoutNode::make(const Layout& orig_layout,
                                          const Layout& store_layout) {
  auto n = make_node<BijectiveLayoutNode>();

  auto LayoutParser = [](const Layout& layout, Array<IterVar>& axes) {
    for (size_t i = 0; i < layout.ndim(); ++i) {
      auto axis_layout = layout[i];
      if (Layout::IsSuperdim(axis_layout)) {
        std::string shape_name("_shape");
        shape_name.insert(0, 1, axis_layout);
        IterVar axis = IterVarNode::make(Range(Expr(0), Var(shape_name)),
                                         Var(std::string(1, axis_layout)), kDataPar);
        axes.push_back(axis);
      } else {
        IterVar axis = IterVarNode::make(Range(Expr(0), Expr(layout.Subsizeof(axis_layout))),
                                         Var(std::string(1, axis_layout)), kDataPar);
        axes.push_back(axis);
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
