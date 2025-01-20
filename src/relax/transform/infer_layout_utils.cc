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

#include "utils.h"

namespace tvm {
namespace relax {

using tir::IterVar;
using tir::Layout;

std::string TransposeSubLayoutStrLike(const std::string ref_str, const std::string& src_str,
                                      const std::string& desired_str) {
  std::string out;
  for (const char& c : desired_str) {
    if (std::isupper(c)) {
      auto res = src_str.find(c, 0);
      ICHECK(res != std::string::npos) << "Invalid Layout:"
                                       << "can't find " << c << " in source layout" << src_str;
      out.push_back(ref_str[res]);
    } else if (isdigit(c)) {
      out.push_back(c);
    } else if (std::islower(c)) {
      auto res = src_str.find(std::toupper(c), 0);
      ICHECK(res != std::string::npos) << "Invalid Layout:"
                                       << "can't find " << c << " in source layout" << src_str;
      out.push_back(std::tolower(ref_str[res]));
    }
  }
  return out;
}

Layout TransposeSubLayoutLike(const Layout& ref, const Layout& src, const Layout& desired) {
  std::string ref_str = ref.name();
  std::string src_str = src.name();
  std::string desired_str = desired.name();
  std::string out = TransposeSubLayoutStrLike(ref_str, src_str, desired_str);
  return Layout(out);
}

Layout TransposeLike(const Layout& input, const Layout& src, const Layout& dst) {
  ICHECK(src.ndim() == dst.ndim() && input.ndim() == src.ndim())
      << "Layouts must have the same size";
  std::vector<IterVar> axes;
  for (size_t i = 0; i < src.ndim(); ++i) {
    axes.push_back(input->axes[src.IndexOf(dst[i])]);
  }
  return Layout(axes);
}

String TransposeStrLike(const String& input, const Layout& src, const Layout& dst) {
  ICHECK(src.ndim() == dst.ndim() && input.size() == src.ndim())
      << "Layouts must have the same size";
  std::string axes;
  for (size_t i = 0; i < src.ndim(); ++i) {
    axes.push_back(input.at(src.IndexOf(dst[i])));
  }
  return axes;
}

int FindAxis(const Layout& dst, int axis) {
  axis = (axis + dst.ndim()) % dst.ndim();
  std::string layout_name = dst.name();
  layout_name.erase(std::remove_if(layout_name.begin(), layout_name.end(),
                                   [](unsigned char c) { return std::isdigit(c); }),
                    layout_name.end());
  return layout_name.find('A' + axis);
}

Layout InitialLayout(int ndim) {
  ICHECK(ndim >= 0 && ndim <= 26) << "Only support up to 26 dimensions, but got " << ndim;
  return Layout("ABCDEFGHIJKLMNOPQRSTUVWXYZ").SubLayout(0, ndim);
}

LayoutDecision InitialLayoutDecision(int ndim) {
  if (ndim == kUnknownNDim) {
    return LayoutDecision::InitUnknownDim();
  }
  ICHECK(ndim >= 0 && ndim <= 26) << "Only support up to 26 dimensions, but got " << ndim;
  return Layout("ABCDEFGHIJKLMNOPQRSTUVWXYZ").SubLayout(0, ndim);
}

NLayout InitialNLayout(const StructInfo& sinfo) {
  auto fmapleaf = [&](const StructInfo& sinfo) -> NLayout {
    if (const auto* tensor_sinfo = sinfo.as<TensorStructInfoNode>()) {
      return NLayout(InitialLayoutDecision(tensor_sinfo->ndim));
    }
    return LayoutDecision::InitUnknownDim();
  };
  return MapToNestedMsg<LayoutDecision>(sinfo, fmapleaf);
}

NLayout InitialNLayout(const Expr& expr) { return InitialNLayout(GetStructInfo(expr)); }

LayoutDecision GetLayoutDecision(const VarLayoutMap& var_layout_map, const Expr& arg) {
  NLayout nlayout = GetNLayout(var_layout_map, arg);
  ICHECK(nlayout.IsLeaf()) << "Cannot get layout for " << arg;
  return nlayout.LeafValue();
}

NLayout GetNLayout(const VarLayoutMap& var_layout_map, const Expr& arg) {
  auto fmapleaf = [&](const Expr& expr) -> NLayout {
    if (const auto* var = expr.as<VarNode>()) {
      auto it = var_layout_map.find(GetRef<Var>(var));
      if (it != var_layout_map.end()) {
        return (*it).second;
      } else {
        return InitialNLayout(expr);
      }
    } else if (const auto* constant = expr.as<ConstantNode>()) {
      return InitialLayoutDecision(constant->data.Shape().size());
    }
    return LayoutDecision::InitUnknownDim();
  };
  return MapToNestedMsg<LayoutDecision>(arg, fmapleaf);
}

bool NoDesiredLayout(const Call& call, const Map<String, Array<String>>& desired_layouts) {
  const OpNode* op_node = call->op.as<OpNode>();
  if (op_node == nullptr) return false;
  const auto& it = desired_layouts.find(op_node->name);
  return it == desired_layouts.end();
}

LayoutDecision FollowDecision(const LayoutDecision& src, int dst_ndim) {
  int src_ndim = src->layout.ndim();
  // broadcast case
  if (src_ndim == dst_ndim) {
    return src;
  } else {
    ICHECK_LT(src_ndim, dst_ndim) << "Cannot broadcast from " << src_ndim << " to " << dst_ndim;
    std::string layout = InitialLayout(dst_ndim - src_ndim).name();
    for (int i = 0; i < src_ndim; ++i) {
      layout.push_back(src->layout.name()[i] + dst_ndim - src_ndim);
    }
    return LayoutDecision(Layout(layout));
  }
}

}  // namespace relax
}  // namespace tvm
