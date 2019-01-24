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

Layout::Layout(const Array<IterVar>& axes) {
  node_ = make_node<LayoutNode>();
  LayoutNode *node = operator->();
  node->axes = axes;
  std::ostringstream repr;
  for (const IterVar& axis : axes) {
    if (const auto* factor = axis->dom->extent.as<IntImm>()) {
      CHECK_GT(factor->value, 0);
      repr << factor;
    }
    CHECK_EQ(axis->var.get()->name_hint.size(), 1) << "Invalid layout axis "
                                                   << axis->var.get()->name_hint;
    char c = axis->var.get()->name_hint[0];
    CHECK((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) << "Invalid layout axis " << c;
    repr << axis->var.get()->name_hint;
  }
  node->name = repr.str();
}

Layout::Layout(const std::string& name) { // NOLINT(*)
  if (name.empty() || name == "__undef__") return;

  node_ = make_node<LayoutNode>();
  LayoutNode *node = operator->();
  node->name = name;

  // parse layout string
  int64_t factor = 0;
  for (char c : name) {
    if (c >= 'A' && c <= 'Z') {
      CHECK_EQ(factor, 0) << "Invalid layout " << name
                          << ": invalid factor size " << factor
                          << " before dimension " << c;
      std::string shape_name("_shape");
      shape_name.insert(0, 1, c);
      IterVar axis = IterVarNode::make(Range(Expr(0), Var(shape_name)),
                                       Var(std::string(1, c)), kDataPar);
      node->axes.push_back(axis);
    } else if (c >= 'a' && c <= 'z') {
      CHECK_GT(factor, 0) << "Invalid layout " << name << ": invalid factor size "
                          << factor << " for dimension " << c;
      IterVar axis = IterVarNode::make(Range(Expr(0), Expr(factor)),
                                       Var(std::string(1, c)), kDataPar);
      node->axes.push_back(axis);
      factor = 0;
    } else if (c >= '0' && c <= '9') {
      CHECK(factor >= 0) << "Invalid layout " << name << ": _ is adjacent to a number.";
      factor = factor * 10 + c - '0';
    } else {
      LOG(FATAL) << "Invalid layout " << name;
    }
  }

  // validate layout
  std::vector<bool> exist_axis(256, false);
  for (const IterVar& v : node->axes) {
    auto axis_str = v->var.get()->name_hint;
    CHECK_EQ(axis_str.size(), 1);
    char axis = axis_str[0];
    CHECK((axis >= 'a' && axis <= 'z') || (axis >= 'A' && axis <= 'Z'));
    CHECK(!exist_axis[axis]) << "Invalid layout " << name << ": duplicate axis " << axis;
    exist_axis[axis] = true;
  }
  for (const IterVar& v : node->axes) {
    char axis = v->var.get()->name_hint[0];
    if (axis >= 'a' && axis <= 'z') {
      CHECK(exist_axis[axis-'a'+'A']) << "Invalid layout " << name << ": missing axis "
                                      << axis - 'a' + 'A';
    }
  }
}

Layout LayoutNode::make(const std::string& layout) {
  return Layout(layout);
}

Layout Layout::SubLayout(size_t pos, size_t len) const {
  if (!defined() || pos > ndim()) return Layout::Undef();
  if (pos + len > ndim()) len = ndim() - pos;
  Array<IterVar> new_layout;
  const auto axes = operator->()->axes;
  for (size_t i = pos; i < pos + len; ++i) {
    new_layout.push_back(axes[i]);
  }
  return Layout(new_layout);
}

Layout Layout::Split(const LayoutAxis &axis, size_t target_pos, int64_t factor) const {
  if (!defined()) return Layout::Undef();
  const std::string& name = operator->()->name;
  const auto axes = operator->()->axes;
  CHECK(target_pos <= this->ndim()) << "Invalid split position "
                                    << target_pos << " for layout " << name;
  CHECK(axis.IsPrimal()) << "Cannot split a subordinate axis " << axis;
  CHECK(this->Contains(axis)) << "Axis " << axis << " does not exist in " << name;
  CHECK(!this->Contains(axis.to_subordinate())) << "Axis " << axis
                                                << " has already been split in " << name;
  CHECK(factor > 0) << "Invalid split size " << factor;
  Array<IterVar> new_layout;
  for (size_t i = 0; i <= this->ndim(); ++i) {
    if (i == target_pos) {
      new_layout.push_back(IterVarNode::make(Range(Expr(0), Expr(factor)),
                                             Var(axis.to_subordinate().name()), kDataPar));
    }
    if (i == this->ndim()) break;
    new_layout.push_back(axes[i]);
  }
  return Layout(new_layout);
}

int64_t Layout::FactorOf(const LayoutAxis& axis) const {
  if (!defined()) return -1;
  const LayoutAxis& sub = axis.to_subordinate();
  if (!this->defined()) return -1;
  for (const IterVar& itvar : operator->()->axes) {
    if (sub == LayoutAxis::Get(itvar)) {
      const auto* factor = itvar->dom->extent.as<IntImm>();
      CHECK(factor);
      return factor->value;
    }
  }
  return -1;
}

inline bool GetStoreRule(Array<Expr>& rule,
                         const Layout& src_layout,
                         const Layout& dst_layout) {
  for (size_t i = 0; i < dst_layout.ndim(); ++i) {
    const auto& store_axis = dst_layout[i];
    const IterVar& store_axis_impl = dst_layout->axes[i];
    Expr store(0);

    for (size_t j = 0; j < src_layout.ndim(); ++j) {
      const auto& orig_axis = src_layout[j];
      const IterVar& orig_axis_impl = src_layout->axes[j];
      if (store_axis.to_primal() == orig_axis.to_primal()) {
        if (orig_axis.IsPrimal()) {
          Expr orig_var = orig_axis_impl->var;
          const int64_t factor = src_layout.FactorOf(orig_axis);
          if (factor > 0) {
            orig_var = orig_var * Expr(factor);
          }
          store = store + orig_var;
        } else {
          store = store + orig_axis_impl->var;
        }
      }
    }
    if (is_zero(store)) {
      // Not convertible
      return false;
    }

    if (store_axis.IsPrimal()) {
      const int64_t factor = dst_layout.FactorOf(store_axis);
      if (factor > 0) {
        store = store / Expr(factor);
      }
    } else {
      store = store % store_axis_impl->dom->extent;
    }

    rule.push_back(store);
  }
  return true;
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

Array<Expr> BijectiveLayout::ForwardIndex(const Array<Expr>& src_index) const {
  CHECK(defined()) << "Cannot operate on an undefined bijective layout.";
  const BijectiveLayoutNode* self = operator->();
  CHECK_EQ(src_index.size(), self->src_layout->axes.size())
    << "Input mismatch with layout " << self->src_layout;
  return TransformIndex(src_index, self->src_layout->axes, self->forward_rule);
}


Array<Expr> BijectiveLayout::BackwardIndex(const Array<Expr>& dst_index) const {
  CHECK(defined()) << "Cannot operate on an undefined bijective layout.";
  const BijectiveLayoutNode* self = operator->();
  CHECK_EQ(dst_index.size(), self->dst_layout->axes.size())
    << "Output mismatch with layout " << self->dst_layout;
  return TransformIndex(dst_index, self->dst_layout->axes, self->backward_rule);
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
      if (orig_shape.defined()) {
        const auto* orig_shape_const = orig_shape.as<IntImm>();
        const auto* orig_axis_extent = orig_axis->dom->extent.as<IntImm>();
        CHECK_EQ(orig_shape_const->value, orig_axis_extent->value)
          << "Input shape mismatch at index " << i << ". Expected "
          << orig_axis->dom->extent << ", get " << orig_shape;
      }
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
  CHECK(defined()) << "Cannot operate on an undefined bijective layout.";
  const BijectiveLayoutNode* self = operator->();
  return TransformShape(shape, self->src_layout->axes,
                        self->dst_layout->axes, self->forward_rule);
}

Array<Expr> BijectiveLayout::BackwardShape(const Array<Expr>& shape) const {
  CHECK(defined()) << "Cannot operate on an undefined bijective layout.";
  const BijectiveLayoutNode* self = operator->();
  return TransformShape(shape, self->dst_layout->axes,
                        self->src_layout->axes, self->backward_rule);
}

BijectiveLayout BijectiveLayoutNode::make(const Layout& src_layout,
                                          const Layout& dst_layout) {
  auto n = make_node<BijectiveLayoutNode>();

  n->src_layout = src_layout;
  n->dst_layout = dst_layout;

  if (!GetStoreRule(n->forward_rule, n->src_layout, n->dst_layout)) {
    // not convertible
    return BijectiveLayout();
  }
  CHECK(GetStoreRule(n->backward_rule, n->dst_layout, n->src_layout));

  return BijectiveLayout(n);
}

}  // namespace tvm
