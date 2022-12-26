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

#include "infer_layout_utils.h"

#include <tvm/relay/expr.h>
#include <tvm/relay/op_attr_types.h>
#include <tvm/tir/data_layout.h>

#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pattern_utils.h"
#include "tvm/runtime/logging.h"

namespace tvm {
namespace relay {

Layout AdjustSubordinateFactors(const Layout& src_layout, const Layout& old_layout,
                                const Array<tvm::PrimExpr>& old_shape) {
  // For each subordinate axis
  //   1) Find the corresponding dual axis.
  //   2) Find the Index of this dual axis in old_layout.
  //   3) Find the shape of the that axis in old_shape.
  //   4) a) Adjust factor to 1, if that shape is 1. b) Else retain the factor.
  DLOG(INFO) << "AdjustSubordinateFactors"
             << "src_layout: " << src_layout << " old_layout: " << old_layout
             << " old_shape: " << old_shape << std::endl;
  std::string new_layout;
  for (auto axis : src_layout->axes) {
    if (!LayoutAxis::Get(axis).IsPrimal()) {
      bool is_shape_one = false;
      // 1) Find the corresponding dual axis
      const auto& dual_axis = LayoutAxis::Get(axis).ToPrimal();

      // 2) Find the index of this dual axis in old_layout
      int old_axis = old_layout.IndexOf(dual_axis);

      if (old_axis == -1) {
        new_layout += "1";
        is_shape_one = true;
      } else {
        // 3) Find the shape of this index in old_shape
        auto shape_val = old_shape[old_axis];

        // 4) a) Check if this shape element is 1.
        if (auto* shape_int = shape_val.as<IntImmNode>()) {
          // We can treat 1 as broadcast only if axis was not split before
          if (shape_int->value == 1 && old_layout.IndexOf(LayoutAxis::Get(axis)) == -1) {
            new_layout += "1";
            is_shape_one = true;
          }
        }
      }

      // 4) b) If shape is not 1, retain the factor.
      if (!is_shape_one) {
        auto new_shape_val = src_layout.FactorOf(dual_axis);
        new_layout += std::to_string(new_shape_val);
      }
    }
    new_layout += LayoutAxis::Get(axis).name();
  }
  return new_layout != "" ? Layout(new_layout)
                          : Layout("H").SubLayout(0, 0);  // hack to create a scalar layout
}

bool Isomorphic(const Layout& lhs, const Layout& rhs) {
  DLOG(INFO) << "Isomorphic: "
             << "lhs: " << lhs << " rhs: " << rhs << std::endl;
  ICHECK(lhs.defined());
  ICHECK(rhs.defined());
  if (lhs->axes.size() != rhs->axes.size()) return false;
  std::map<std::string, std::string> map_to, map_back;
  for (size_t i = 0; i < lhs->axes.size(); ++i) {
    auto& lhs_axis = LayoutAxis::Get(lhs->axes[i]);
    auto& rhs_axis = LayoutAxis::Get(rhs->axes[i]);
    std::string name_lhs = lhs_axis.name();
    std::string name_rhs = rhs_axis.name();
    if (lhs_axis.IsPrimal() != rhs_axis.IsPrimal()) return false;

    auto it = map_to.find(name_lhs);
    if (it == map_to.end())
      map_to[name_lhs] = name_rhs;
    else if (it->second != name_rhs)
      return false;

    it = map_back.find(name_rhs);
    if (it == map_back.end())
      map_back[name_rhs] = name_lhs;
    else if (it->second != name_lhs)
      return false;
    if (!lhs_axis.IsPrimal() && lhs.FactorOf(lhs_axis) != rhs.FactorOf(rhs_axis)) return false;
  }
  return true;
}

Layout TryTransformLike(const Layout& old, const Layout& ref_old, const Layout& ref_new) {
  DLOG(INFO) << "transform_layout: old = " << old << ", ref_new = " << ref_new
             << ", ref_old = " << ref_old << std::endl;
  ICHECK(ref_old.defined());
  ICHECK(ref_new.defined());
  ICHECK(old.defined());

  {  // check if old and ref_old are similar enough such that it's
     // compatible for the transform ref_old -> ref_new
    const Layout& large = ref_old.ndim() > old.ndim() ? ref_old : old;
    const Layout& small = large == ref_old ? old : ref_old;
    Layout large_sublayout = large.SubLayout(large.ndim() - small.ndim(), small.ndim()),
           rest_sublayout = large.SubLayout(0, large.ndim() - small.ndim());
    bool orthorgonal = true;
    for (auto i : rest_sublayout->axes)
      if (large_sublayout.IndexOf(LayoutAxis::Get(i).ToPrimal()) != -1 ||
          large_sublayout.IndexOf(LayoutAxis::Get(i).ToSubordinate()) != -1) {
        orthorgonal = false;
        break;
      }
    if (!orthorgonal || !Isomorphic(large_sublayout, small))
      return Layout::Undef();  // For now this case is not supported.
  }

  // `old` is compatible. Now learn the axis name mapping between `old` and `ref_old`
  if (old.ndim() == 0) return old;  // an optmization for scalar: no-op
  std::vector<int> mapping(26, -1);
  std::vector<bool> used(26, false);

  auto find_unused = [&](char preference) -> char {
    if (!used[preference - 'A']) return preference;  // preference unused
    for (int i = 0; i < 26; ++i)
      if (!used[i]) return 'A' + i;
    LOG(FATAL) << "All letters are used";
  };

  for (int j = old->axes.size() - 1, i = ref_old->axes.size() - 1; j >= 0; --i, --j) {
    char name_ref = LayoutAxis::Get(ref_old->axes[i]).ToPrimal().name()[0];
    char name = LayoutAxis::Get(old->axes[j]).ToPrimal().name()[0];
    mapping[name_ref - 'A'] = name - 'A';
    used[name - 'A'] = true;
  }

  for (int i = ref_old->axes.size() - 1; i >= 0; --i) {
    char name_ref = LayoutAxis::Get(ref_old->axes[i]).ToPrimal().name()[0];
    int name = mapping[name_ref - 'A'];
    if (name == -1) {
      mapping[name_ref - 'A'] = find_unused(name_ref) - 'A';
      used[mapping[name_ref - 'A']] = true;
    }
  }

  // apply the mapping to rename `ref_new`
  std::string new_layout;
  for (auto c : std::string(ref_new->name)) {
    if (c >= 'A' && c <= 'Z') {
      ICHECK(mapping[c - 'A'] != -1);
      new_layout += mapping[c - 'A'] + 'A';
    } else if (c >= 'a' && c <= 'z') {
      ICHECK(mapping[c - 'a'] != -1);
      new_layout += mapping[c - 'a'] + 'a';
    } else {
      new_layout += c;
    }
  }

  DLOG(INFO) << "new_layout = " << new_layout << std::endl;
  return Layout(new_layout);
}

std::pair<Array<Layout>, Array<Layout>> BinaryBroadcastLayoutHelper(
    const Attrs& attrs, const Array<Layout>& new_in_layouts, const Array<Layout>& old_in_layouts,
    const Array<tvm::relay::Type>& old_in_types) {
  // Two steps. Step (2) only executes if the function is called after rewrite.
  // (1) infer input layouts before rewrite
  // (2) if some input layouts are changed by its producer after rewrite, rewrite the other
  // layout to make sure it's changed in the same way, so that they are still broadcastable.
  Array<Layout> layouts;
  Array<Array<IndexExpr>> old_in_shapes;
  for (auto old_in_t : old_in_types) {
    ICHECK(old_in_t.as<TensorTypeNode>());
    old_in_shapes.push_back(old_in_t.as<TensorTypeNode>()->shape);
  }
  int old_large_idx = old_in_shapes[0].size() >= old_in_shapes[1].size() ? 0 : 1;

  layouts.Assign(old_in_layouts.begin(), old_in_layouts.end());
  // always operate on the original layouts first for consistency

  std::pair<Array<Layout>, Array<Layout>> out,
      out_default{{Layout::Undef(), Layout::Undef()}, {Layout::Undef()}};

  if (!layouts[0].defined() && !layouts[1].defined()) {
    // both undefined, infer fails
    out = out_default;
  } else if (!layouts[0].defined() || !layouts[1].defined()) {
    // only one is defined, use shape information to help infer
    int defined_idx = layouts[0].defined() ? 0 : 1;
    int undef_idx = 1 - defined_idx;

    if (old_in_shapes[defined_idx].size() >= old_in_shapes[undef_idx].size()) {
      // TODO(lazycal): handle the case when the sublayout contains subcoordinate of factor one but
      // the other tensor has the corresponding dimension size other than one.
      // E.g. defined's shape = [x, x, x, x, 1] in NCHW1c and undefined's shape = [3]
      layouts.Set(undef_idx, layouts[defined_idx].SubLayout(old_in_shapes[defined_idx].size() -
                                                                old_in_shapes[undef_idx].size(),
                                                            old_in_shapes[undef_idx].size()));
      out = {layouts, {layouts[defined_idx]}};
    } else {
      // only know the tensor with smaller dimensions,
      // so we cannot infer the final broadcasted output.
      // fails in this case.
      out = out_default;
    }
  } else {
    // when both are defined, return the larger one
    out = {layouts, {layouts[old_large_idx]}};
  }
  if (!new_in_layouts.defined()) return out;
  // Step (2) rewrite the layouts to make them broadcastable again.
  Layout ret = new_in_layouts[old_large_idx];
  int large_idx = new_in_layouts[0].ndim_primal() >= new_in_layouts[1].ndim_primal() ? 0 : 1;
  int small_idx = 1 - large_idx;
  // start adjusting

  // Apply a greedy strategy that always transform the small layout in the same way as the
  // large layout is transformed, if possible.
  Layout tgt_layout =
      TryTransformLike(layouts[small_idx], layouts[large_idx], new_in_layouts[large_idx]);
  if (!tgt_layout.defined()) return out_default;  // fallback

  // Support scenarios where original operands were of type [N, H, W, C] and [N, H, W, 1]
  // In this case, we might have NCHW16c coming for 1 operand. However, the other operand does
  // not have enough C dimension. To reuse broadcasting, we would want to use NCHW1c for the
  // second operand. The following section of code walks through the layouts and shapes to
  // perform that operation.
  // a in NCHWC16c
  // b in NHW1
  // b = layout_transform(b) from NHW1 -> NCHW1c
  // add(a, b)
  auto old_small_shape = old_in_shapes[small_idx];
  auto old_small_layout = layouts[small_idx];
  auto new_small_layout = AdjustSubordinateFactors(tgt_layout, old_small_layout, old_small_shape);
  layouts.Set(large_idx, new_in_layouts[large_idx]);
  layouts.Set(small_idx, new_small_layout);
  return {layouts, {ret}};
}

}  //  namespace relay
}  //  namespace tvm
