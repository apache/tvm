/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/lang/data_layout.cc
 * \brief Data Layout expression.
 */
#include <tvm/runtime/registry.h>
#include <tvm/tir/data_layout.h>
#include <tvm/tir/ir_pass.h>
#include <tvm/arith/analyzer.h>

#include <cctype>

namespace tvm {
namespace tir {
using tir::Var;
using tir::IterVar;
using tir::IterVarNode;

TVM_REGISTER_NODE_TYPE(LayoutNode);
TVM_REGISTER_NODE_TYPE(BijectiveLayoutNode);

const LayoutAxis LayoutAxis::UPPER_CASE[] = {
  LayoutAxis('A'), LayoutAxis('B'), LayoutAxis('C'), LayoutAxis('D'), LayoutAxis('E'),
  LayoutAxis('F'), LayoutAxis('G'), LayoutAxis('H'), LayoutAxis('I'), LayoutAxis('J'),
  LayoutAxis('K'), LayoutAxis('L'), LayoutAxis('M'), LayoutAxis('N'), LayoutAxis('O'),
  LayoutAxis('P'), LayoutAxis('Q'), LayoutAxis('R'), LayoutAxis('S'), LayoutAxis('T'),
  LayoutAxis('U'), LayoutAxis('V'), LayoutAxis('W'), LayoutAxis('X'), LayoutAxis('Y'),
  LayoutAxis('Z')
};

const LayoutAxis LayoutAxis::LOWER_CASE[] = {
  LayoutAxis('a'), LayoutAxis('b'), LayoutAxis('c'), LayoutAxis('d'), LayoutAxis('e'),
  LayoutAxis('f'), LayoutAxis('g'), LayoutAxis('h'), LayoutAxis('i'), LayoutAxis('j'),
  LayoutAxis('k'), LayoutAxis('l'), LayoutAxis('m'), LayoutAxis('n'), LayoutAxis('o'),
  LayoutAxis('p'), LayoutAxis('q'), LayoutAxis('r'), LayoutAxis('s'), LayoutAxis('t'),
  LayoutAxis('u'), LayoutAxis('v'), LayoutAxis('w'), LayoutAxis('x'), LayoutAxis('y'),
  LayoutAxis('z')
};

const LayoutAxis& LayoutAxis::Get(const char name) {
  CHECK((name >= 'A' && name <= 'Z') || (name >= 'a' && name <= 'z'))
    << "Invalid layout axis name: " << name << ". Has to be A-Z or a-z.";
  return (name >= 'A' && name <= 'Z') ?
         LayoutAxis::UPPER_CASE[name-'A'] :
         LayoutAxis::LOWER_CASE[name-'a'];
}

const LayoutAxis& LayoutAxis::Get(const IterVar& itvar) {
  const std::string axis = itvar->var.get()->name_hint;
  CHECK_EQ(axis.size(), 1) << "Invalid layout axis " << axis;
  return LayoutAxis::Get(axis[0]);
}

const LayoutAxis& LayoutAxis::make(const std::string& name) {
  CHECK_EQ(name.length(), 1) << "Invalid axis " << name;
  return LayoutAxis::Get(name[0]);
}

Layout::Layout(const Array<IterVar>& axes) {
  auto node = make_object<LayoutNode>();
  node->axes = axes;
  std::ostringstream repr;
  for (const IterVar& axis : axes) {
    if (const auto* factor = axis->dom->extent.as<IntImmNode>()) {
      CHECK_GT(factor->value, 0);
      repr << factor->value;
    }
    CHECK_EQ(axis->var.get()->name_hint.size(), 1) << "Invalid layout axis "
                                                   << axis->var.get()->name_hint;
    char c = axis->var.get()->name_hint[0];
    CHECK((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) << "Invalid layout axis " << c;
    repr << axis->var.get()->name_hint;
  }
  node->name = repr.str();
  data_ = std::move(node);
}

Layout::Layout(const std::string& name) { // NOLINT(*)
  if (name == "__undef__") return;

  auto node = make_object<LayoutNode>();
  node->name = name;

  if (name.empty()) return;  // scalar

  // parse layout string
  int32_t factor = 0;
  for (char c : name) {
    if (c >= 'A' && c <= 'Z') {
      CHECK_EQ(factor, 0) << "Invalid layout " << name
                          << ": invalid factor size " << factor
                          << " before dimension " << c;
      std::string shape_name("_shape");
      shape_name.insert(0, 1, c);
      IterVar axis = IterVarNode::make(Range(PrimExpr(0), Var(shape_name)),
                                       Var(std::string(1, c)), tir::kDataPar);
      node->axes.push_back(axis);
    } else if (c >= 'a' && c <= 'z') {
      CHECK_GT(factor, 0) << "Invalid layout " << name << ": invalid factor size "
                          << factor << " for dimension " << c;
      IterVar axis = IterVarNode::make(Range(PrimExpr(0), PrimExpr(factor)),
                                       Var(std::string(1, c)), tir::kDataPar);
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
                                      << std::toupper(axis);
    }
  }
  data_ = std::move(node);
}

Layout LayoutNode::make(const std::string& layout) {
  return Layout(layout);
}

Layout Layout::SubLayout(size_t pos, size_t len) const {
  if (!defined() || pos > ndim()) return Layout::Undef();
  if (len == 0) return Layout(Array<IterVar>());
  if (pos + len > ndim()) len = ndim() - pos;
  Array<IterVar> new_layout;
  const auto axes = operator->()->axes;
  for (size_t i = pos; i < pos + len; ++i) {
    new_layout.push_back(axes[i]);
  }
  return Layout(new_layout);
}

Layout Layout::Split(const LayoutAxis &axis, size_t target_pos, int32_t factor) const {
  if (!defined()) return Layout::Undef();
  const std::string& name = operator->()->name;
  const auto axes = operator->()->axes;
  CHECK(target_pos <= this->ndim()) << "Invalid split position "
                                    << target_pos << " for layout " << name;
  CHECK(axis.IsPrimal()) << "Cannot split a subordinate axis " << axis;
  CHECK(this->Contains(axis)) << "Axis " << axis << " does not exist in " << name;
  CHECK(!this->Contains(axis.ToSubordinate())) << "Axis " << axis
                                                << " has already been split in " << name;
  CHECK(factor > 0) << "Invalid split size " << factor;
  Array<IterVar> new_layout;
  for (size_t i = 0; i <= this->ndim(); ++i) {
    if (i == target_pos) {
      new_layout.push_back(IterVarNode::make(Range(PrimExpr(0), PrimExpr(factor)),
                                             Var(axis.ToSubordinate().name()), tir::kDataPar));
    }
    if (i == this->ndim()) break;
    new_layout.push_back(axes[i]);
  }
  return Layout(new_layout);
}

int32_t Layout::FactorOf(const LayoutAxis& axis) const {
  if (!defined()) return -1;
  const LayoutAxis& sub = axis.ToSubordinate();
  if (!this->defined()) return -1;
  for (const IterVar& itvar : operator->()->axes) {
    if (sub == LayoutAxis::Get(itvar)) {
      const auto* factor = itvar->dom->extent.as<IntImmNode>();
      CHECK(factor);
      return factor->value;
    }
  }
  return -1;
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<LayoutNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* l = static_cast<const LayoutNode*>(node.get());
    p->stream << "Layout(" << l->name << ")";
  });

inline bool GetStoreRule(Array<PrimExpr>* rule,
                         const Layout& src_layout,
                         const Layout& dst_layout) {
  if (!src_layout.defined() || src_layout.name().empty() ||
      !dst_layout.defined() || dst_layout.name().empty()) {
    return false;
  }
  for (size_t i = 0; i < dst_layout.ndim(); ++i) {
    const auto& store_axis = dst_layout[i];
    const IterVar& store_axis_impl = dst_layout->axes[i];
    PrimExpr store(0);

    for (size_t j = 0; j < src_layout.ndim(); ++j) {
      const auto& orig_axis = src_layout[j];
      const IterVar& orig_axis_impl = src_layout->axes[j];
      if (store_axis.ToPrimal() == orig_axis.ToPrimal()) {
        if (orig_axis.IsPrimal()) {
          PrimExpr orig_var = orig_axis_impl->var;
          const int32_t factor = src_layout.FactorOf(orig_axis);
          if (factor > 0) {
            orig_var = orig_var * PrimExpr(factor);
          }
          store = store + orig_var;
        } else {
          store = store + orig_axis_impl->var;
        }
      }
    }
    if (tir::is_zero(store)) {
      // Not convertible
      return false;
    }

    if (store_axis.IsPrimal()) {
      const int32_t factor = dst_layout.FactorOf(store_axis);
      if (factor > 0) {
        store = indexdiv(store, PrimExpr(factor));
      }
    } else {
      store = indexmod(store, store_axis_impl->dom->extent);
    }

    rule->push_back(store);
  }
  return true;
}

inline Array<PrimExpr> TransformIndex(const Array<PrimExpr>& src_index,
                                      const Array<IterVar>& src_axis,
                                      const Array<PrimExpr>& transform_rule) {
  arith::Analyzer ana;
  Array<PrimExpr> result;
  std::unordered_map<const tir::VarNode*, PrimExpr> bind_map;
  for (size_t i = 0; i < src_index.size(); ++i) {
    bind_map[src_axis[i]->var.get()] = src_index[i];
  }
  for (PrimExpr rule : transform_rule) {
    result.push_back(ana.Simplify(tir::Substitute(rule, bind_map)));
  }
  return result;
}

Array<PrimExpr> BijectiveLayout::ForwardIndex(const Array<PrimExpr>& src_index) const {
  CHECK(defined()) << "Cannot operate on an undefined bijective layout.";
  const BijectiveLayoutNode* self = operator->();
  CHECK_EQ(src_index.size(), self->src_layout->axes.size())
    << "Input mismatch with layout " << self->src_layout;
  return TransformIndex(src_index, self->src_layout->axes, self->forward_rule);
}


Array<PrimExpr> BijectiveLayout::BackwardIndex(const Array<PrimExpr>& dst_index) const {
  CHECK(defined()) << "Cannot operate on an undefined bijective layout.";
  const BijectiveLayoutNode* self = operator->();
  CHECK_EQ(dst_index.size(), self->dst_layout->axes.size())
    << "Output mismatch with layout " << self->dst_layout;
  return TransformIndex(dst_index, self->dst_layout->axes, self->backward_rule);
}

inline Array<PrimExpr> TransformShape(const Array<PrimExpr>& src_shape,
                                      const Array<IterVar>& src_axis,
                                      const Array<IterVar>& target_axis,
                                      const Array<PrimExpr>& transform_rule) {
  arith::Analyzer ana;
  CHECK_EQ(src_shape.size(), src_axis.size());
  // bind variables for original axes
  // for major-axis, bind the corresponding size
  // for minor-axis, simply bind it as 0, so that we can reuse forward/backward_rule,
  // e.g., (C * 16 + c) / 32
  std::unordered_map<const tir::VarNode*, PrimExpr> bind_map;
  std::unordered_set<size_t> symbolic_var_set;
  for (size_t i = 0; i < src_shape.size(); ++i) {
    PrimExpr orig_shape = src_shape[i];
    IterVar orig_axis = src_axis[i];
    if (orig_shape.as<tir::AnyNode>()) {
      symbolic_var_set.insert(i);
    }
    if (!LayoutAxis::Get(orig_axis).IsPrimal()) {
      if (orig_shape.defined()) {
        const auto* orig_shape_const = orig_shape.as<IntImmNode>();
        const auto* orig_axis_extent = orig_axis->dom->extent.as<IntImmNode>();
        if (orig_shape_const) {
          CHECK_EQ(orig_shape_const->value, orig_axis_extent->value)
            << "Input shape mismatch at index " << i << ". Expected "
            << orig_axis->dom->extent << ", get " << orig_shape;
        }
      }
      bind_map[orig_axis->var.get()] = PrimExpr(0);
    } else {
      bind_map[orig_axis->var.get()] = orig_shape;
    }
  }
  // infer the target shape,
  // for major-axis, use the forward/backward_rule directly,
  // for minor-axis, simply use the extent.
  Array<PrimExpr> result;
  CHECK_EQ(transform_rule.size(), target_axis.size());
  for (size_t i = 0; i < transform_rule.size(); ++i) {
    PrimExpr rule = transform_rule[i];
    IterVar axis = target_axis[i];
    if (!LayoutAxis::Get(axis).IsPrimal()) {
      result.push_back(axis->dom->extent);
    } else {
      if (symbolic_var_set.count(i)) {
        result.push_back(tir::AnyNode::make());
      } else {
        result.push_back(ana.Simplify(tir::Substitute(rule, bind_map)));
      }
    }
  }
  return result;
}

Array<PrimExpr> BijectiveLayout::ForwardShape(const Array<PrimExpr>& shape) const {
  CHECK(defined()) << "Cannot operate on an undefined bijective layout.";
  const BijectiveLayoutNode* self = operator->();
  return TransformShape(shape, self->src_layout->axes,
                        self->dst_layout->axes, self->forward_rule);
}

Array<PrimExpr> BijectiveLayout::BackwardShape(const Array<PrimExpr>& shape) const {
  CHECK(defined()) << "Cannot operate on an undefined bijective layout.";
  const BijectiveLayoutNode* self = operator->();
  return TransformShape(shape, self->dst_layout->axes,
                        self->src_layout->axes, self->backward_rule);
}

BijectiveLayout::BijectiveLayout(Layout src_layout, Layout dst_layout) {
  auto n = make_object<BijectiveLayoutNode>();

  n->src_layout = std::move(src_layout);
  n->dst_layout = std::move(dst_layout);

  // To be consistent with previous behavior, a nullptr layout is created
  // when argument is invalid.
  if (GetStoreRule(&n->forward_rule, n->src_layout, n->dst_layout)) {
    CHECK(GetStoreRule(&n->backward_rule, n->dst_layout, n->src_layout));
    data_ = std::move(n);
  }
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
.set_dispatch<BijectiveLayoutNode>([](const ObjectRef& node, ReprPrinter* p) {
    auto* b = static_cast<const BijectiveLayoutNode*>(node.get());
    p->stream << "BijectiveLayout(" << b->src_layout.name()
              << "->" << b->dst_layout.name() << ")";
  });

TVM_REGISTER_GLOBAL("tir.Layout")
.set_body_typed(LayoutNode::make);

TVM_REGISTER_GLOBAL("tir.LayoutIndexOf")
.set_body_typed([](Layout layout, std::string axis) -> int {
  return layout.IndexOf(LayoutAxis::make(axis));
});

TVM_REGISTER_GLOBAL("tir.LayoutFactorOf")
.set_body_typed([](Layout layout, std::string axis) -> int {
  return layout.FactorOf(LayoutAxis::make(axis));
});

TVM_REGISTER_GLOBAL("tir.LayoutNdim")
.set_body_typed([](Layout layout) -> int {
  return layout.ndim();
});

TVM_REGISTER_GLOBAL("tir.LayoutGetItem")
.set_body_typed([](Layout layout, int idx) -> std::string {
  const LayoutAxis& axis = layout[idx];
  return axis.name();
});

TVM_REGISTER_GLOBAL("tir.BijectiveLayout")
.set_body_typed([](Layout src_layout, Layout dst_layout) -> BijectiveLayout {
  return BijectiveLayout(src_layout, dst_layout);
});

TVM_REGISTER_GLOBAL("tir.BijectiveLayoutForwardIndex")
.set_body_method(&BijectiveLayout::ForwardIndex);

TVM_REGISTER_GLOBAL("tir.BijectiveLayoutBackwardIndex")
.set_body_method(&BijectiveLayout::BackwardIndex);

TVM_REGISTER_GLOBAL("tir.BijectiveLayoutForwardShape")
.set_body_method(&BijectiveLayout::ForwardShape);

TVM_REGISTER_GLOBAL("tir.BijectiveLayoutBackwardShape")
.set_body_method(&BijectiveLayout::BackwardShape);
}  // namespace tir
}  // namespace tvm
