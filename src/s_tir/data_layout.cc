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
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/data_layout.h>
#include <tvm/ir/expr.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/object.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/var.h>

#include <algorithm>
#include <cctype>

namespace tvm {
namespace tir {
using tir::IterVar;
using tir::IterVarNode;
using tir::Var;

TVM_FFI_STATIC_INIT_BLOCK() {
  LayoutNode::RegisterReflection();
  BijectiveLayoutNode::RegisterReflection();
}

const LayoutAxis LayoutAxis::UPPER_CASE[] = {
    LayoutAxis('A'), LayoutAxis('B'), LayoutAxis('C'), LayoutAxis('D'), LayoutAxis('E'),
    LayoutAxis('F'), LayoutAxis('G'), LayoutAxis('H'), LayoutAxis('I'), LayoutAxis('J'),
    LayoutAxis('K'), LayoutAxis('L'), LayoutAxis('M'), LayoutAxis('N'), LayoutAxis('O'),
    LayoutAxis('P'), LayoutAxis('Q'), LayoutAxis('R'), LayoutAxis('S'), LayoutAxis('T'),
    LayoutAxis('U'), LayoutAxis('V'), LayoutAxis('W'), LayoutAxis('X'), LayoutAxis('Y'),
    LayoutAxis('Z')};

const LayoutAxis LayoutAxis::LOWER_CASE[] = {
    LayoutAxis('a'), LayoutAxis('b'), LayoutAxis('c'), LayoutAxis('d'), LayoutAxis('e'),
    LayoutAxis('f'), LayoutAxis('g'), LayoutAxis('h'), LayoutAxis('i'), LayoutAxis('j'),
    LayoutAxis('k'), LayoutAxis('l'), LayoutAxis('m'), LayoutAxis('n'), LayoutAxis('o'),
    LayoutAxis('p'), LayoutAxis('q'), LayoutAxis('r'), LayoutAxis('s'), LayoutAxis('t'),
    LayoutAxis('u'), LayoutAxis('v'), LayoutAxis('w'), LayoutAxis('x'), LayoutAxis('y'),
    LayoutAxis('z')};

const LayoutAxis& LayoutAxis::Get(const char name) {
  ICHECK((name >= 'A' && name <= 'Z') || (name >= 'a' && name <= 'z'))
      << "Invalid layout axis name: " << name << ". Has to be A-Z or a-z.";
  return (name >= 'A' && name <= 'Z') ? LayoutAxis::UPPER_CASE[name - 'A']
                                      : LayoutAxis::LOWER_CASE[name - 'a'];
}

const LayoutAxis& LayoutAxis::Get(const IterVar& itvar) {
  const std::string axis = itvar->var.get()->name_hint;
  ICHECK_EQ(axis.size(), 1) << "Invalid layout axis " << axis;
  return LayoutAxis::Get(axis[0]);
}

const LayoutAxis& LayoutAxis::Get(const std::string& name) {
  ICHECK_EQ(name.length(), 1) << "Invalid axis " << name;
  return LayoutAxis::Get(name[0]);
}

Layout::Layout(const ffi::Array<IterVar>& axes) {
  auto node = ffi::make_object<LayoutNode>();
  node->axes = axes;
  std::ostringstream repr;

  for (const IterVar& packed_axis : axes) {
    auto unpacked_axes = UnpackIterVar(packed_axis);
    bool is_grouped = unpacked_axes.size() > 1;

    if (is_grouped) repr << "[";
    for (const IterVar& axis : unpacked_axes) {
      if (const auto* factor = axis->dom->extent.as<IntImmNode>()) {
        ICHECK_GT(factor->value, 0);
        repr << factor->value;
      } else {
        ICHECK(!is_grouped) << "Only Subordinate Axes with extent is allowed within a packed dim";
      }
      ICHECK_EQ(axis->var.get()->name_hint.size(), 1)
          << "Invalid layout axis " << axis->var.get()->name_hint;
      char c = axis->var.get()->name_hint.operator std::string()[0];
      ICHECK((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) << "Invalid layout axis " << c;
      repr << axis->var.get()->name_hint;
    }
    if (is_grouped) repr << "]";
  }

  node->name = repr.str();
  data_ = std::move(node);
}

Layout::Layout(const std::string& name, DataType dtype) {  // NOLINT(*)
  CHECK(dtype.is_int()) << "TypeError: The input dtype should be integer type";
  if (name == "__undef__") return;

  auto node = ffi::make_object<LayoutNode>();
  node->name = name;

  if (name.empty()) return;  // scalar

  // parse layout string
  int32_t factor = 0;
  bool in_packing = false;
  std::vector<IterVar> unpacked_axes;

  for (char c : name) {
    if (c >= 'A' && c <= 'Z') {
      ICHECK_EQ(factor, 0) << "Invalid layout " << name << ": invalid factor size " << factor
                           << " before dimension " << c;
      IterVar axis(Range(IntImm(dtype, 0), Var(std::string(1, c), dtype)),
                   Var(std::string(1, c), dtype), tir::kDataPar);
      if (!in_packing) {
        node->axes.push_back(axis);
      } else {
        unpacked_axes.push_back(axis);
      }
    } else if (c >= 'a' && c <= 'z') {
      ICHECK_GT(factor, 0) << "Invalid layout " << name << ": invalid factor size " << factor
                           << " for dimension " << c;
      std::stringstream name;
      name << factor << c;
      IterVar axis(Range(IntImm(dtype, 0), IntImm(dtype, factor)), Var(name.str(), dtype),
                   tir::kDataPar);
      if (!in_packing) {
        node->axes.push_back(axis);
      } else {
        unpacked_axes.push_back(axis);
      }
      factor = 0;
    } else if (c >= '0' && c <= '9') {
      ICHECK(factor >= 0) << "Invalid layout " << name << ": _ is adjacent to a number.";
      factor = factor * 10 + c - '0';
    } else if (c == '[') {
      ICHECK(!in_packing) << "Invalid layout " << name << ": can't do nested packing";
      in_packing = true;
    } else if (c == ']') {
      ICHECK(in_packing) << "Invalid layout " << name << ": encountered ] without matching bracket";
      ICHECK(unpacked_axes.size() > 1)
          << "Invalid layout " << name << ": found empty/single packed axis";
      std::stringstream ss;
      int64_t extent = 1;
      for (auto& axis : unpacked_axes) {
        ICHECK(axis->dom->extent.as<IntImmNode>())
            << "Invalid Layout " << name << ": can't have variable sized node("
            << axis->var->name_hint << ") within a packed axis";
        auto axis_name = axis->var->name_hint.operator std::string();
        auto factor = axis->dom->extent.as<IntImm>().value();
        ss << axis_name;
        extent = extent * factor->value;
      }
      std::string grouped_name = ss.str();
      IterVar grouped_axis(Range(IntImm(dtype, 0), IntImm(dtype, extent)), Var(grouped_name, dtype),
                           tir::kDataPar);
      node->axes.push_back(grouped_axis);

      in_packing = false;
      unpacked_axes.clear();
    } else {
      LOG(FATAL) << "Invalid layout " << name;
    }
  }
  ICHECK(in_packing == false) << "Invalid Layout " << name
                              << ": haven't terminated the packing sequence";

  // validate layout
  std::vector<int> axis_cnt(256, 0);
  for (const IterVar& pv : node->axes) {
    for (const IterVar& v : UnpackIterVar(pv)) {
      auto axis_str = v->var.get()->name_hint.operator std::string();
      ICHECK_EQ(axis_str.size(), 1);
      char axis = axis_str[0];
      ICHECK((axis >= 'a' && axis <= 'z') || (axis >= 'A' && axis <= 'Z'));
      axis_cnt[axis] += 1;
    }
  }
  for (const IterVar& pv : node->axes) {
    for (const IterVar& v : UnpackIterVar(pv)) {
      char axis = v->var.get()->name_hint.operator std::string()[0];
      if (axis >= 'a' && axis <= 'z') {
        ICHECK(axis_cnt[axis - 'a' + 'A'])
            << "Invalid layout " << name << ": missing axis " << std::toupper(axis);
        ICHECK(axis_cnt[axis] == 1) << "Invalid layout " << name
                                    << ": found more than one subordinate " << std::toupper(axis);
      }
    }
  }

  data_ = std::move(node);
}

Layout Layout::SubLayout(size_t pos, size_t len) const {
  if (!defined() || pos > ndim()) return Layout::Undef();
  if (len == 0) return Layout(ffi::Array<IterVar>());
  if (pos + len > ndim()) len = ndim() - pos;
  ffi::Array<IterVar> new_layout;
  const auto axes = operator->()->axes;
  for (size_t i = pos; i < pos + len; ++i) {
    new_layout.push_back(axes[i]);
  }
  return Layout(new_layout);
}

ffi::Array<IterVar> Layout::UnpackIterVar(IterVar packed_iter) {
  ffi::Array<IterVar> result;
  int64_t factor = 0, final_factor = 1;

  std::string name(packed_iter->var->name_hint.c_str());
  DataType dtype = packed_iter->var.dtype();

  for (auto ch : name) {
    if (ch >= '0' && ch <= '9') {
      factor = factor * 10 + (ch - '0');
    } else if (ch >= 'a' && ch <= 'z') {
      ICHECK(factor != 0) << "Invalid Factor Size";
      result.push_back(IterVar(Range(IntImm(dtype, 0), IntImm(dtype, factor)),
                               Var(std::string(1, ch), dtype), tir::kDataPar));
      final_factor *= factor;
      factor = 0;
    } else if (ch >= 'A' && ch <= 'Z') {
      ICHECK(factor == 0) << "Can't have non-zero factors for primal axis";
      result.push_back(IterVar(Range(IntImm(dtype, 0), Var(std::string(1, ch), dtype)),
                               Var(std::string(1, ch), dtype), tir::kDataPar));
    }
  }

  return result;
}

IterVar Layout::PackIterVar(ffi::Array<IterVar> iter_vars) {
  std::stringstream name;
  size_t extent = 1;

  DataType dtype = iter_vars[0]->dom->extent.as<PrimExpr>().value()->dtype;
  for (auto itvar : iter_vars) {
    ICHECK(itvar->dom->extent.as<IntImm>()) << "Packed Axis can contain only Subordinate Axes";
    name << itvar->dom->extent.as<IntImm>().value() << itvar->var->name_hint;
    extent = extent * itvar->dom->extent.as<IntImm>().value()->value;
  }

  return IterVar(Range(IntImm(dtype, 0), IntImm(dtype, extent)), Var(name.str(), dtype),
                 tir::kDataPar);
}

int32_t Layout::FactorOf(const LayoutAxis& axis) const {
  if (!defined()) return -1;
  const LayoutAxis& sub = axis.ToSubordinate();

  int32_t factor = 1;
  bool has_sub = false;
  for (const IterVar& packed_itvar : operator->()->axes) {
    for (auto itvar : UnpackIterVar(packed_itvar)) {
      if (sub == LayoutAxis::Get(itvar)) {
        has_sub = true;
        int32_t val = itvar->dom->extent.as<IntImmNode>()->value;
        factor *= val;
      }
    }
  }
  factor = has_sub ? factor : -1;

  return factor;
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<LayoutNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* l = static_cast<const LayoutNode*>(node.get());
      p->stream << "Layout(" << l->name << ")";
    });

inline bool GetStoreRule(ffi::Array<PrimExpr>* index_rule, ffi::Array<PrimExpr>* shape_rule,
                         const Layout& src_layout, const Layout& dst_layout) {
  if (!src_layout.defined() || src_layout.name().empty()) {
    LOG(WARNING) << "src layout '" << src_layout.name() << "' is invalid.";
    return false;
  }
  if (!dst_layout.defined() || dst_layout.name().empty()) {
    LOG(WARNING) << "dst layout '" << dst_layout.name() << "' is invalid.";
    return false;
  }

  std::vector<bool> exists(128, false);
  PrimExpr norm_indexes[128];
  for (auto& it : norm_indexes) it = PrimExpr(0);

  for (size_t i = 0; i < src_layout.ndim(); i++) {
    auto factor = src_layout.PackedAxisAt(i)->dom->extent;
    auto src_unpacked_axes = Layout::UnpackIterVar(src_layout.PackedAxisAt(i));

    if (src_unpacked_axes.size() == 1 && LayoutAxis::Get(src_unpacked_axes[0]).IsPrimal()) {
      const auto& prim_axis = LayoutAxis::Get(src_unpacked_axes[0]);
      int64_t offset = src_layout.FactorOf(prim_axis);
      if (offset == -1)
        norm_indexes[prim_axis.name()[0] - 'A'] =
            norm_indexes[prim_axis.name()[0] - 'A'] + src_layout.PackedAxisAt(i);
      else
        norm_indexes[prim_axis.name()[0] - 'A'] =
            norm_indexes[prim_axis.name()[0] - 'A'] +
            src_layout.PackedAxisAt(i) * src_layout.FactorOf(prim_axis);
      exists[prim_axis.name()[0]] = true;
    } else {
      int64_t value = 1;
      std::vector<int> index_divs(src_unpacked_axes.size());
      for (size_t j = 0; j < src_unpacked_axes.size(); j++) {
        index_divs[j] = value;
        const auto* extent = src_unpacked_axes[j]->dom->extent.as<IntImmNode>();
        ICHECK(extent) << "Expected Integer Extents for Offset Calculation";
        index_divs.push_back(value);
        value = value * extent->value;
      }
      std::reverse(index_divs.begin(), index_divs.end());

      for (size_t j = 0; j < src_unpacked_axes.size(); j++) {
        const int extent = src_unpacked_axes[j]->dom->extent.as<IntImmNode>()->value;
        const LayoutAxis& store_axis_impl = LayoutAxis::Get(src_unpacked_axes[j]);
        const LayoutAxis& sub_axis = store_axis_impl.ToSubordinate(); /* Not Needed */
        const LayoutAxis& prim_axis = store_axis_impl.ToPrimal();

        PrimExpr factor_ij = indexdiv(src_layout.PackedAxisAt(i), index_divs[j]);
        if (j != 0) factor_ij = indexmod(factor_ij, extent);

        for (size_t k = i; k < src_layout.ndim(); k++) {
          size_t l = 0;
          if (k == i) l = j + 1;

          auto inter_unpacked_axes = Layout::UnpackIterVar(src_layout.PackedAxisAt(k));
          for (; l < inter_unpacked_axes.size(); l++) {
            const LayoutAxis& axis = LayoutAxis::Get(inter_unpacked_axes[l]);
            if (axis == sub_axis) {
              const auto* sub_extent = inter_unpacked_axes[l]->dom->extent.as<IntImmNode>();
              ICHECK(sub_extent) << "Expected Integer Extents for Offset Calculation";
              factor_ij = factor_ij * IntImm(sub_extent->dtype, sub_extent->value);
            }
          }
        }

        norm_indexes[prim_axis.name()[0] - 'A'] =
            norm_indexes[prim_axis.name()[0] - 'A'] + factor_ij;
      }
    }
  }

  arith::Analyzer ana;

  for (size_t i = 0; i < dst_layout.ndim(); i++) {
    const auto dst_unpacked_axes = Layout::UnpackIterVar(dst_layout.PackedAxisAt(i));

    if (dst_unpacked_axes.size() == 1 && LayoutAxis::Get(dst_unpacked_axes[0]).IsPrimal()) {
      const auto& prim_axis = LayoutAxis::Get(dst_unpacked_axes[0]);
      if (!exists[prim_axis.name()[0]]) return false;
      int64_t offset = dst_layout.FactorOf(prim_axis);
      if (offset != -1) {
        index_rule->push_back(
            indexdiv(norm_indexes[prim_axis.name()[0] - 'A'], dst_layout.FactorOf(prim_axis)));
        shape_rule->push_back(
            indexdiv(norm_indexes[prim_axis.name()[0] - 'A'] + (dst_layout.FactorOf(prim_axis) - 1),
                     dst_layout.FactorOf(prim_axis)));
      } else {
        index_rule->push_back(norm_indexes[prim_axis.name()[0] - 'A']);
        shape_rule->push_back(norm_indexes[prim_axis.name()[0] - 'A']);
      }
    } else {
      PrimExpr factor(0);
      for (size_t j = 0; j < dst_unpacked_axes.size(); j++) {
        const auto& prim_axis = LayoutAxis::Get(dst_unpacked_axes[j]).ToPrimal();
        const auto& sub_axis = LayoutAxis::Get(dst_unpacked_axes[j]).ToSubordinate();
        const auto* extent = dst_unpacked_axes[j]->dom->extent.as<IntImmNode>();
        ICHECK(extent) << "Expected extent to be IntImmNode";

        size_t divfactor = 1;
        for (size_t k = i; k < dst_layout.ndim(); k++) {
          size_t l = 0;
          if (k == i) l = j + 1;

          const auto inter_unpacked_axes = Layout::UnpackIterVar(dst_layout.PackedAxisAt(k));
          for (; l < inter_unpacked_axes.size(); l++) {
            const auto& axis = LayoutAxis::Get(inter_unpacked_axes[l]);
            if (sub_axis == axis) {
              const auto* sub_extent = inter_unpacked_axes[l]->dom->extent.as<IntImmNode>();
              ICHECK(sub_extent) << "Expected Integer Extents for Offset Calculation";
              divfactor = divfactor * sub_extent->value;
            }
          }
        }

        factor = factor + indexmod(indexdiv(norm_indexes[prim_axis.name()[0] - 'A'], divfactor),
                                   extent->value);
        for (size_t k = j + 1; k < dst_unpacked_axes.size(); k++) {
          factor = factor * dst_unpacked_axes[k]->dom->extent.as<IntImm>().value();
        }
      }
      ana.Simplify(factor);
      index_rule->push_back(factor);
      shape_rule->push_back(factor);
    }
  }

  std::stringstream ss;
  ss << "index rule for " << src_layout.name() << "-->" << dst_layout.name() << ": [ ";
  for (const auto& r : *index_rule) {
    ss << r << ", ";
  }
  ss << "]" << std::endl;

  ss << "shape rule for " << src_layout.name() << "-->" << dst_layout.name() << ": [ ";
  for (const auto& r : *shape_rule) {
    ss << r << ", ";
  }
  ss << "]" << std::endl;
  VLOG(1) << ss.str() << std::endl;

  return true;
}

inline ffi::Array<PrimExpr> TransformIndex(const ffi::Array<PrimExpr>& src_index,
                                           const ffi::Array<IterVar>& src_axis,
                                           const ffi::Array<PrimExpr>& transform_rule) {
  arith::Analyzer ana;
  ffi::Array<PrimExpr> result;
  std::unordered_map<const tir::VarNode*, PrimExpr> bind_map;
  for (size_t i = 0; i < src_index.size(); ++i) {
    bind_map[src_axis[i]->var.get()] = src_index[i];
  }
  for (PrimExpr rule : transform_rule) {
    result.push_back(ana.Simplify(tir::Substitute(rule, bind_map)));
  }
  return result;
}

ffi::Array<PrimExpr> BijectiveLayout::ForwardIndex(const ffi::Array<PrimExpr>& src_index) const {
  ICHECK(defined()) << "Cannot operate on an undefined bijective layout.";
  const BijectiveLayoutNode* self = operator->();
  ICHECK_EQ(src_index.size(), self->src_layout->axes.size())
      << "Input mismatch with layout " << self->src_layout;
  return TransformIndex(src_index, self->src_layout->axes, self->index_forward_rule);
}

ffi::Array<PrimExpr> BijectiveLayout::BackwardIndex(const ffi::Array<PrimExpr>& dst_index) const {
  ICHECK(defined()) << "Cannot operate on an undefined bijective layout.";
  const BijectiveLayoutNode* self = operator->();
  ICHECK_EQ(dst_index.size(), self->dst_layout->axes.size())
      << "Output mismatch with layout " << self->dst_layout;
  return TransformIndex(dst_index, self->dst_layout->axes, self->index_backward_rule);
}

inline ffi::Array<PrimExpr> TransformShape(const ffi::Array<PrimExpr>& src_shape,
                                           const ffi::Array<IterVar>& src_axis,
                                           const ffi::Array<IterVar>& target_axis,
                                           const ffi::Array<PrimExpr>& transform_rule) {
  arith::Analyzer ana;
  ICHECK_EQ(src_shape.size(), src_axis.size())
      << "Input shape size " << src_shape.size() << " mismatch with the expected shape size "
      << src_axis.size();
  // bind variables for original axes
  // for major-axis, bind the corresponding size
  // for minor-axis, simply bind it as 0, so that we can reuse forward/backward_rule,
  // e.g., (C * 16 + c) / 32
  std::unordered_map<const tir::VarNode*, PrimExpr> bind_map;
  for (size_t i = 0; i < src_shape.size(); ++i) {
    PrimExpr orig_shape = src_shape[i];
    IterVar orig_axis = src_axis[i];
    auto layout = Layout::UnpackIterVar(orig_axis);
    if (layout.size() != 1 || !LayoutAxis::Get(layout[0]).IsPrimal()) {
      if (orig_shape.defined()) {
        const auto* orig_shape_const = orig_shape.as<IntImmNode>();
        const auto* orig_axis_extent = orig_axis->dom->extent.as<IntImmNode>();
        if (orig_shape_const) {
          ICHECK_EQ(orig_shape_const->value, orig_axis_extent->value)
              << "Input shape mismatch at index " << i << ". Expected " << orig_axis->dom->extent
              << ", get " << orig_shape;
        }
      }
      bind_map[orig_axis->var.get()] = IntImm(orig_axis->var->dtype, 0);
    } else {
      bind_map[orig_axis->var.get()] = orig_axis->var->dtype == orig_shape->dtype
                                           ? orig_shape
                                           : cast(orig_axis->var->dtype, orig_shape);
    }
  }
  // infer the target shape,
  // for major-axis, use the forward/backward_rule directly,
  // for minor-axis, simply use the extent.
  ffi::Array<PrimExpr> result;
  ICHECK_EQ(transform_rule.size(), target_axis.size());
  for (size_t i = 0; i < transform_rule.size(); ++i) {
    PrimExpr rule = transform_rule[i];
    IterVar axis = target_axis[i];
    auto layout = Layout::UnpackIterVar(axis);
    if (layout.size() != 1 || !LayoutAxis::Get(layout[0]).IsPrimal()) {
      result.push_back(axis->dom->extent);
    } else {
      result.push_back(ana.Simplify(tir::Substitute(rule, bind_map)));
    }
  }

  std::stringstream ss;
  ss << "shape rule for " << Layout(src_axis).name() << "-->" << Layout(target_axis).name()
     << ": [ ";
  for (const auto& r : transform_rule) {
    ss << r << ", ";
  }
  ss << "]" << std::endl;

  ss << "shape transform: [ ";
  for (const auto& s : src_shape) {
    ss << s << ", ";
  }
  ss << "] --> [ ";
  for (const auto& r : result) {
    ss << r << ", ";
  }
  ss << "]" << std::endl;
  VLOG(1) << std::endl << ss.str();

  return result;
}

ffi::Array<PrimExpr> BijectiveLayout::ForwardShape(const ffi::Array<PrimExpr>& shape) const {
  ICHECK(defined()) << "Cannot operate on an undefined bijective layout.";
  const BijectiveLayoutNode* self = operator->();
  return TransformShape(shape, self->src_layout->axes, self->dst_layout->axes,
                        self->shape_forward_rule);
}

ffi::Array<PrimExpr> BijectiveLayout::BackwardShape(const ffi::Array<PrimExpr>& shape) const {
  ICHECK(defined()) << "Cannot operate on an undefined bijective layout.";
  const BijectiveLayoutNode* self = operator->();
  return TransformShape(shape, self->dst_layout->axes, self->src_layout->axes,
                        self->shape_backward_rule);
}

BijectiveLayout::BijectiveLayout(Layout src_layout, Layout dst_layout) {
  auto n = ffi::make_object<BijectiveLayoutNode>();

  n->src_layout = std::move(src_layout);
  n->dst_layout = std::move(dst_layout);
  // To be consistent with previous behavior, a nullptr layout is created
  // when argument is invalid.
  if (GetStoreRule(&n->index_forward_rule, &n->shape_forward_rule, n->src_layout, n->dst_layout)) {
    ICHECK(GetStoreRule(&n->index_backward_rule, &n->shape_backward_rule, n->dst_layout,
                        n->src_layout));
    data_ = std::move(n);
  }
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<BijectiveLayoutNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* b = static_cast<const BijectiveLayoutNode*>(node.get());
      p->stream << "BijectiveLayout(" << b->src_layout.name() << "->" << b->dst_layout.name()
                << ")";
    });

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("s_tir.Layout", [](std::string name, DataType dtype) { return Layout(name, dtype); })
      .def("s_tir.LayoutIndexOf",
           [](Layout layout, std::string axis) -> int {
             return layout.IndexOf(axis);
           })
      .def("s_tir.LayoutFactorOf",
           [](Layout layout, std::string axis) -> int {
             return layout.FactorOf(LayoutAxis::Get(axis));
           })
      .def("s_tir.LayoutNdim", [](Layout layout) -> int { return layout.ndim(); })
      .def("s_tir.LayoutGetItem",
           [](Layout layout, int idx) -> std::string {
             const auto& axis = layout.PackedAxisAt(idx);
             return axis->var->name_hint;
           })
      .def("s_tir.BijectiveLayout",
           [](Layout src_layout, Layout dst_layout) -> BijectiveLayout {
             return BijectiveLayout(src_layout, dst_layout);
           })
      .def_method("s_tir.BijectiveLayoutForwardIndex", &BijectiveLayout::ForwardIndex)
      .def_method("s_tir.BijectiveLayoutBackwardIndex", &BijectiveLayout::BackwardIndex)
      .def_method("s_tir.BijectiveLayoutForwardShape", &BijectiveLayout::ForwardShape)
      .def_method("s_tir.BijectiveLayoutBackwardShape", &BijectiveLayout::BackwardShape);
}
}  // namespace tir
}  // namespace tvm
