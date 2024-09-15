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
 * \file src/contrib/msc/core/transform/layout_utils.cc
 */
#include "layout_utils.h"

#include <algorithm>
#include <set>
#include <string>

namespace tvm {
namespace contrib {
namespace msc {

NLayout LayoutUtils::InferNLayout(const Expr& expr, const VarLayoutMap& var_layout_map) {
  if (expr->IsInstance<VarNode>() && var_layout_map.count(Downcast<Var>(expr))) {
    return tvm::relax::GetNLayout(var_layout_map, expr);
  }
  return GetNLayout(expr);
}

LayoutDecision LayoutUtils::InferLayoutDecision(const Expr& expr,
                                                const VarLayoutMap& var_layout_map) {
  const auto& nlayout = InferNLayout(expr, var_layout_map);
  ICHECK(nlayout.IsLeaf()) << "Cannot get layout for " << expr;
  return nlayout.LeafValue();
}

LayoutDecision LayoutUtils::InferLayoutDecisionAt(const Expr& expr,
                                                  const VarLayoutMap& var_layout_map,
                                                  size_t index) {
  const auto& nlayouts = InferNLayout(expr, var_layout_map);
  if (nlayouts.IsLeaf()) {
    return index == 0 ? nlayouts.LeafValue() : LayoutDecision("");
  }
  const auto& nlayout = nlayouts.NestedArray()[0];
  ICHECK(nlayout.IsLeaf()) << "Cannot get output layout for " << expr;
  return nlayout.LeafValue();
}

bool LayoutUtils::LayoutInfered(const Expr& expr) {
  const String& layout = SpanUtils::GetAttr(expr->span, msc_attr::kLayout);
  return layout.size() > 0;
}

bool LayoutUtils::SetLayout(const Expr& expr, const NLayout& layout) {
  const String& saved_layout = SpanUtils::GetAttr(expr->span, msc_attr::kLayout);
  const auto& sinfo = GetStructInfo(expr);
  if (sinfo->IsInstance<TensorStructInfoNode>() || sinfo->IsInstance<ShapeStructInfoNode>()) {
    if (!layout.IsLeaf()) {
      return false;
    }
    const auto& l_layout = layout.LeafValue()->layout;
    if (!l_layout.defined()) {
      return false;
    }
    if (saved_layout == l_layout.name()) {
      return false;
    }
    expr->span = SpanUtils::SetAttr(expr->span, msc_attr::kLayout, l_layout.name());
  } else if (sinfo->IsInstance<TupleStructInfoNode>()) {
    if (layout.IsLeaf()) {
      return false;
    }
    String layout_str;
    Array<NLayout> nested_layouts = layout.NestedArray();
    for (size_t i = 0; i < nested_layouts.size(); i++) {
      if (!nested_layouts[i].IsLeaf()) {
        return false;
      }
      const auto& l_layout = nested_layouts[i].LeafValue()->layout;
      if (!l_layout.defined()) {
        return false;
      }
      layout_str = layout_str + l_layout.name() + (i < nested_layouts.size() - 1 ? "," : "");
    }
    if (saved_layout == layout_str) {
      return false;
    }
    expr->span = SpanUtils::SetAttr(expr->span, msc_attr::kLayout, layout_str);
  }
  return true;
}

const NLayout LayoutUtils::GetNLayout(const Expr& expr) {
  if (!LayoutInfered(expr)) {
    return LayoutDecision("");
  }
  auto sinfo = GetStructInfo(expr);
  if (sinfo->IsInstance<TensorStructInfoNode>()) {
    return LayoutDecision(SpanUtils::GetAttr(expr->span, msc_attr::kLayout));
  }
  if (sinfo->IsInstance<TupleStructInfoNode>()) {
    String layout_str = SpanUtils::GetAttr(expr->span, msc_attr::kLayout);
    std::vector<NLayout> output_layout;
    for (const auto& l : StringUtils::Split(layout_str, ",")) {
      output_layout.push_back(LayoutDecision(l));
    }
    return NLayout(output_layout);
  }
  return LayoutDecision("");
}

const LayoutDecision LayoutUtils::GetLayoutDecision(const Expr& expr) {
  NLayout nlayout = GetNLayout(expr);
  ICHECK(nlayout.IsLeaf()) << "Cannot get layout for " << expr;
  return nlayout.LeafValue();
}

bool LayoutUtils::HasUnknownDimTensor(const NLayout& nlayout) {
  bool find = false;
  auto fvisit = [&](const LayoutDecision& layout) {
    find = find | (NLayoutEqual()(layout, LayoutDecision::InitUnknownDim()));
  };
  ForEachLeaf<LayoutDecision>(nlayout, fvisit);
  return find;
}

bool LayoutUtils::HasUnknownDimTensor(const Array<Expr>& args) {
  for (const auto& arg : args) {
    if (IsNestedTensor(arg)) {
      if (HasUnknownDimTensor(GetNLayout(arg))) {
        return true;
      }
    }
  }
  return false;
}

const LayoutDecision LayoutUtils::ExpandLayout(const LayoutDecision& src_layout,
                                               const std::vector<size_t>& expand_axes) {
  if (!src_layout->layout.defined()) {
    return src_layout;
  }
  // sort expand axes
  std::vector<size_t> axes = expand_axes;
  std::sort(std::begin(axes), std::end(axes));
  std::string new_layout = src_layout.name();
  ICHECK_EQ(new_layout.size(), src_layout->layout.ndim())
      << "Only support normal layout, get " << src_layout->layout;
  std::set<std::string> used_axes;
  for (size_t i = 0; i < src_layout->layout.ndim(); i++) {
    used_axes.insert(src_layout->layout[i].name());
  }
  std::vector<std::string> prefer_axes{"N", "C", "H", "W", "D"};
  for (const auto& a : axes) {
    bool use_prefer = false;
    if (used_axes.size() < prefer_axes.size()) {
      use_prefer =
          std::all_of(prefer_axes.begin(), prefer_axes.begin() + used_axes.size(),
                      [&used_axes](const std::string& axis) { return used_axes.count(axis); });
    }
    std::string new_axis;
    char cur_axis = 'A';
    if (use_prefer) {
      new_axis = prefer_axes[used_axes.size()];
    } else {
      while (used_axes.count(std::string(1, cur_axis))) {
        cur_axis += 1;
      }
      new_axis = std::string(1, cur_axis);
    }
    used_axes.insert(new_axis);
    new_layout = new_layout.insert(a, new_axis);
  }
  return LayoutDecision(new_layout);
}

const LayoutDecision LayoutUtils::ReduceLayout(const LayoutDecision& src_layout,
                                               const std::vector<size_t>& reduce_axes) {
  if (!src_layout->layout.defined()) {
    return src_layout;
  }
  std::set<size_t> reduce_axes_set;
  for (const auto& a : reduce_axes) {
    reduce_axes_set.insert(a);
  }
  std::string new_layout = "";
  for (size_t i = 0; i < src_layout->layout.ndim(); i++) {
    if (reduce_axes_set.count(i)) {
      continue;
    }
    new_layout += src_layout->layout[i].name();
  }
  return LayoutDecision(new_layout);
}

const LayoutDecision LayoutUtils::PermuteLayout(const LayoutDecision& src_layout,
                                                const Array<Integer>& axes) {
  String layout_str;
  for (const auto& a : axes) {
    layout_str = layout_str + src_layout->layout[a->value].name();
  }
  return LayoutDecision(layout_str);
}

const LayoutDecision LayoutUtils::PermuteLayout(const LayoutDecision& src_layout,
                                                const std::vector<size_t>& axes) {
  String layout_str;
  for (const auto& a : axes) {
    layout_str = layout_str + src_layout->layout[a].name();
  }
  return LayoutDecision(layout_str);
}

int LayoutUtils::InferBatchDim(const LayoutDecision& layout) {
  if (!layout->layout.defined()) {
    return -1;
  }
  for (size_t i = 0; i < layout->layout.ndim(); i++) {
    if (layout->layout[i].name() == "N") {
      return static_cast<int>(i);
    }
  }
  return -1;
}

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
