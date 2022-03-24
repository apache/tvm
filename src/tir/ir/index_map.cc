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
 * \file index_map.cc
 */

#include "tvm/tir/index_map.h"

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <sstream>

namespace tvm {
namespace tir {

IndexMap::IndexMap(Array<Var> initial_indices, Array<PrimExpr> final_indices) {
  auto n = make_object<IndexMapNode>();
  n->initial_indices = std::move(initial_indices);
  n->final_indices = std::move(final_indices);
  data_ = std::move(n);
}

IndexMap IndexMap::FromFunc(int ndim, runtime::TypedPackedFunc<Array<PrimExpr>(Array<Var>)> func) {
  Array<Var> initial_indices;
  initial_indices.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    initial_indices.push_back(Var("i" + std::to_string(i), DataType::Int(32)));
  }
  return IndexMap(initial_indices, func(initial_indices));
}

IndexMap IndexMap::Inverse(Array<Range> initial_ranges) const {
  // Dummy variables to represent the inverse's inputs.
  Array<Var> output_vars;
  for (size_t i = 0; i < (*this)->final_indices.size(); i++) {
    PrimExpr index = (*this)->final_indices[i];
    // TODO(Lunderberg): Better names for these variables.  A variable
    // that is passed through unmodified (`index` is an element of
    // `initial_indices`) should use that input index's name.  A pair
    // of output indices variables split from a single input index
    // should be named (X.outer,X.inner).
    std::stringstream ss;
    ss << "axis" << i;
    Var var_index(ss.str(), index.dtype());
    output_vars.push_back(var_index);
  }

  // Dummy ranges for the extent of each input.
  Map<Var, Range> input_iters;
  ICHECK_EQ((*this)->initial_indices.size(), initial_ranges.size());
  for (size_t i = 0; i < initial_ranges.size(); i++) {
    input_iters.Set((*this)->initial_indices[i], initial_ranges[i]);
  }

  // Unpack the output indices into linear combinations of the initial
  // indices.
  arith::Analyzer analyzer;
  auto diagnostics = DiagnosticContext::Default(IRModule());
  auto iter_map =
      DetectIterMap((*this)->final_indices, input_iters, 1, true, &analyzer, diagnostics);
  CHECK(iter_map.size()) << "Index transformation was not bijective.";

  // Determine expressions for the input variables, in terms of the
  // output variables.
  Map<Var, PrimExpr> inverse_exprs_map =
      InverseAffineIterMap(iter_map, Array<PrimExpr>(output_vars.begin(), output_vars.end()));

  // Unpack the map to an array, maintaining the same parameter order.
  Array<PrimExpr> inverse_exprs;
  for (const auto& index : (*this)->initial_indices) {
    inverse_exprs.push_back(inverse_exprs_map.at(index));
  }

  return IndexMap(output_vars, inverse_exprs);
}

Array<PrimExpr> IndexMapNode::MapIndices(const Array<PrimExpr>& indices) const {
  ICHECK_EQ(indices.size(), initial_indices.size());

  arith::Analyzer analyzer;

  for (size_t i = 0; i < initial_indices.size(); i++) {
    analyzer.Bind(initial_indices[i], indices[i]);
  }

  Array<PrimExpr> output;
  for (const auto& output_dim : final_indices) {
    output.push_back(analyzer.Simplify(output_dim));
  }

  return output;
}

Array<Range> IndexMapNode::MapRanges(const Array<Range>& ranges) const {
  ICHECK_EQ(ranges.size(), initial_indices.size());

  Map<Var, Range> input_iters;
  for (size_t i = 0; i < initial_indices.size(); i++) {
    input_iters.Set(initial_indices[i], ranges[i]);
  }

  std::unordered_map<const VarNode*, arith::IntSet> dom_map;
  for (size_t i = 0; i < initial_indices.size(); i++) {
    dom_map[initial_indices[i].get()] = arith::IntSet::FromRange(ranges[i]);
  }

  Array<Range> output;
  for (const auto& final_index : final_indices) {
    auto int_set = arith::EvalSet(final_index, dom_map);
    output.push_back(Range::FromMinExtent(int_set.min(), int_set.max() - int_set.min() + 1));
  }

  return output;
}

Array<PrimExpr> IndexMapNode::MapShape(const Array<PrimExpr>& shape) const {
  ICHECK_EQ(shape.size(), initial_indices.size());

  Array<Range> ranges;
  for (auto& dim : shape) {
    ranges.push_back(Range(0, dim));
  }
  Array<Range> mapped = MapRanges(std::move(ranges));

  Array<PrimExpr> output;
  for (auto& range : mapped) {
    ICHECK(is_zero(range->min));
    output.push_back(range->extent);
  }

  return output;
}

String IndexMapNode::ToPythonString() const {
  std::unordered_set<std::string> used_names;
  Map<Var, PrimExpr> var_remap;
  for (const Var& initial_index : initial_indices) {
    if (used_names.count(initial_index->name_hint)) {
      std::string new_name = initial_index->name_hint + std::to_string(used_names.size());
      used_names.insert(new_name);
      var_remap.Set(initial_index, Var(new_name));
    } else {
      used_names.insert(initial_index->name_hint);
    }
  }
  std::ostringstream oss;
  oss << "lambda ";
  for (size_t i = 0; i < initial_indices.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    auto it = var_remap.find(initial_indices[i]);
    if (it != var_remap.end()) {
      oss << (*it).second;
    } else {
      oss << initial_indices[i];
    }
  }
  oss << ": (";
  for (size_t i = 0; i < final_indices.size(); ++i) {
    oss << Substitute(final_indices[i], var_remap);
    oss << ", ";
  }
  oss << ")";
  return String(oss.str());
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<IndexMapNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const IndexMapNode*>(node.get());
      p->stream << "index_map(" << op->ToPythonString() << ")";
    });

TVM_REGISTER_NODE_TYPE(IndexMapNode);

TVM_REGISTER_GLOBAL("tir.IndexMap")
    .set_body_typed([](Array<Var> initial_indices, Array<PrimExpr> final_indices) {
      return IndexMap(initial_indices, final_indices);
    });

}  // namespace tir
}  // namespace tvm
