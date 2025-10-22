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

#include <tvm/arith/analyzer.h>
#include <tvm/arith/int_set.h>
#include <tvm/arith/iter_affine_map.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/name_supply.h>
#include <tvm/tir/index_map.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>

#include <sstream>

namespace tvm {
namespace tir {

TVM_FFI_STATIC_INIT_BLOCK() { IndexMapNode::RegisterReflection(); }

IndexMap::IndexMap(ffi::Array<Var> initial_indices, ffi::Array<PrimExpr> final_indices,
                   ffi::Optional<IndexMap> inverse_index_map) {
  auto n = ffi::make_object<IndexMapNode>();
  n->initial_indices = std::move(initial_indices);
  n->final_indices = std::move(final_indices);
  n->inverse_index_map = std::move(inverse_index_map);
  data_ = std::move(n);
}

IndexMap IndexMap::FromFunc(int ndim,
                            ffi::TypedFunction<ffi::Array<PrimExpr>(ffi::Array<Var>)> func,
                            ffi::Optional<IndexMap> inverse_index_map) {
  ffi::Array<Var> initial_indices;
  initial_indices.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    initial_indices.push_back(Var("i" + std::to_string(i), DataType::Int(32)));
  }
  return IndexMap(initial_indices, func(initial_indices), std::move(inverse_index_map));
}

std::pair<IndexMap, PrimExpr> IndexMapInverseImpl(const IndexMap& self,
                                                  const ffi::Array<Range>& initial_ranges,
                                                  arith::IterMapLevel check_level,
                                                  arith::Analyzer* analyzer) {
  ICHECK(analyzer != nullptr);
  if (self->inverse_index_map.defined()) {
    // return the pre-defined inverse index map if exists.  In this
    // case, the user-defined inverse is assumed to be correct and
    // bijective.
    PrimExpr padding_predicate = Bool(false);
    return {Downcast<IndexMap>(self->inverse_index_map.value()), padding_predicate};
  }

  // Dummy variables to represent the inverse's inputs.
  ffi::Array<Var> output_vars;
  for (size_t i = 0; i < self->final_indices.size(); i++) {
    PrimExpr index = self->final_indices[i];
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
  ffi::Map<Var, Range> input_iters;
  ICHECK_EQ(self->initial_indices.size(), initial_ranges.size());
  for (size_t i = 0; i < initial_ranges.size(); i++) {
    input_iters.Set(self->initial_indices[i], initial_ranges[i]);
  }

  // Unpack the output indices into linear combinations of the initial
  // indices.
  auto padded_iter_map = DetectIterMap(self->final_indices, input_iters, /*predicate=*/1,
                                       /*check_level=*/check_level, analyzer,
                                       /*simplify_trivial_iterators=*/false);
  CHECK(padded_iter_map->errors.empty()) << "Could not parse mapping as sum of iterators.  "
                                         << "\nIndex map: " << self->initial_indices << " -> " << self->final_indices
                                         << "\nError: " << padded_iter_map->errors[0];

  // Determine expressions for the input variables, in terms of the
  // output variables.
  ffi::Map<Var, PrimExpr> inverse_exprs_map = InverseAffineIterMap(
      padded_iter_map->indices, ffi::Array<PrimExpr>(output_vars.begin(), output_vars.end()));

  // Unpack the map to an array, maintaining the same parameter order.
  ffi::Array<PrimExpr> inverse_exprs;
  for (int i = 0, n = self->initial_indices.size(); i < n; ++i) {
    Var index = self->initial_indices[i];
    PrimExpr expr;
    if (is_one(initial_ranges[i]->extent) && !inverse_exprs_map.count(index)) {
      expr = initial_ranges[i]->min;
    } else {
      expr = inverse_exprs_map.at(index);
    }
    inverse_exprs.push_back(analyzer->Simplify(expr));
  }

  PrimExpr padding_predicate = padded_iter_map->padding_predicate;
  padding_predicate = arith::NormalizeIterMapToExpr(padding_predicate);
  padding_predicate = Substitute(padding_predicate, inverse_exprs_map);

  auto output_ranges = self->MapRanges(initial_ranges, analyzer);
  {
    ICHECK_EQ(output_ranges.size(), output_vars.size());

    arith::Analyzer analyzer;
    for (size_t i = 0; i < output_vars.size(); ++i) {
      analyzer.Bind(output_vars[i], output_ranges[i]);
    }

    // Additional simplification steps required to unwrap nested floordiv/floormod
    padding_predicate = analyzer.Simplify(padding_predicate, 10);
  }

  return {IndexMap(output_vars, inverse_exprs), padding_predicate};
}

std::pair<IndexMap, PrimExpr> IndexMap::NonSurjectiveInverse(ffi::Array<Range> initial_ranges,
                                                             arith::Analyzer* analyzer) const {
  ICHECK(analyzer != nullptr);
  return IndexMapInverseImpl(*this, initial_ranges, arith::IterMapLevel::NoCheck, analyzer);
}

IndexMap IndexMap::Inverse(ffi::Array<Range> initial_ranges, arith::Analyzer* analyzer) const {
  ICHECK(analyzer != nullptr);
  auto [inverse, padding_predicate] =
      IndexMapInverseImpl(*this, initial_ranges, arith::IterMapLevel::Bijective, analyzer);
  CHECK(analyzer->CanProve(!padding_predicate))
      << "Bijective inverse should not contain padding, but inverse of " << *this << " over range "
      << initial_ranges << " resulted in a padding predicate of " << padding_predicate;
  return inverse;
}

ffi::Array<PrimExpr> IndexMapNode::MapIndices(const ffi::Array<PrimExpr>& indices,
                                              arith::Analyzer* analyzer) const {
  ICHECK(analyzer != nullptr);
  ICHECK_EQ(indices.size(), initial_indices.size());

  ffi::Map<Var, PrimExpr> vmap;

  for (size_t i = 0; i < initial_indices.size(); i++) {
    vmap.Set(initial_indices[i], indices[i]);
  }

  ffi::Array<PrimExpr> output = final_indices.Map([&](PrimExpr index) {
    PrimExpr result = SubstituteWithDataTypeLegalization(
        std::move(index), [&](const Var& var) { return vmap.Get(var); });
    return analyzer->Simplify(result);
  });
  return output;
}

ffi::Array<Range> IndexMapNode::MapRanges(const ffi::Array<Range>& ranges,
                                          arith::Analyzer* analyzer) const {
  ICHECK(analyzer != nullptr);
  ICHECK_EQ(ranges.size(), initial_indices.size());

  ffi::Map<Var, Range> input_iters;
  for (size_t i = 0; i < initial_indices.size(); i++) {
    input_iters.Set(initial_indices[i], ranges[i]);
  }
  auto iter_map = DetectIterMap(final_indices, input_iters, /* predicate = */ 1,
                                /*check_level=*/arith::IterMapLevel::NoCheck, analyzer,
                                /*simplify_trivial_iterators=*/false);
  ffi::Array<Range> output;
  if (iter_map->indices.size()) {
    // Preferred route, requires the map to be expressible as an
    // affine sum.  Since the terms are orthogonal, the extent of the
    // sum is the extent of the largest term.
    for (const auto& index : iter_map->indices) {
      ffi::Optional<PrimExpr> extent = std::nullopt;
      for (const auto& term : index->args) {
        PrimExpr term_extent = term->extent * term->scale;
        if (extent.defined()) {
          extent = tvm::max(extent.value(), term_extent);
        } else {
          extent = term_extent;
        }
      }
      output.push_back(Range::FromMinExtent(index->base, extent.value_or(1)));
    }

  } else {
    // Fall-back method, more general but can ignore intended padding.
    // For example, [N] mapped through i=>[i//4,i%4] should have shape
    // [ceildiv(N,4), 4].  However, for N<4, this method instead
    // results in a shape [1, N].
    std::unordered_map<const VarNode*, arith::IntSet> dom_map;
    for (size_t i = 0; i < initial_indices.size(); i++) {
      dom_map[initial_indices[i].get()] = arith::IntSet::FromRange(ranges[i]);
    }

    for (const auto& final_index : final_indices) {
      auto int_set = arith::EvalSet(final_index, dom_map);
      output.push_back(Range::FromMinExtent(analyzer->Simplify(int_set.min()),
                                            analyzer->Simplify(int_set.max() - int_set.min() + 1)));
    }
  }
  auto output_dtype = [&]() {
    int max_bits = ranges.empty() ? 32 : 0;
    for (const auto& range : ranges) {
      max_bits = std::max(max_bits, range->extent.dtype().bits());
    }
    return DataType::Int(max_bits);
  }();
  output.MutateByApply([&](const Range& range) {
    if (range->min.dtype() != output_dtype || range->extent.dtype() != output_dtype) {
      return Range::FromMinExtent(cast(output_dtype, range->min),
                                  cast(output_dtype, range->extent));
    } else {
      return range;
    }
  });
  return output;
}

ffi::Array<PrimExpr> IndexMapNode::MapShape(const ffi::Array<PrimExpr>& shape,
                                            arith::Analyzer* analyzer) const {
  ICHECK(analyzer != nullptr);
  ICHECK_EQ(shape.size(), initial_indices.size());

  ffi::Array<Range> ranges;
  for (auto& dim : shape) {
    ranges.push_back(Range(make_zero(dim.dtype()), dim));
  }
  ffi::Array<Range> mapped = MapRanges(std::move(ranges), analyzer);

  ffi::Array<PrimExpr> output;
  for (auto& range : mapped) {
    ICHECK(is_zero(range->min));
    output.push_back(range->extent);
  }

  return output;
}

runtime::Tensor IndexMapNode::MapTensor(runtime::Tensor arr_src) const {
  arith::Analyzer analyzer;
  auto shape = arr_src.Shape();
  ICHECK(shape.size() == initial_indices.size())
      << "The rank of the input array should be " << initial_indices.size() << " but got "
      << shape.size();
  size_t size_1d = 1;
  ffi::Array<PrimExpr> orig_shape;
  for (size_t i = 0; i < shape.size(); ++i) {
    size_1d *= shape[i];
    orig_shape.push_back(PrimExpr(static_cast<int>((shape[i]))));
  }
  auto dst_shape = MapShape(orig_shape, &analyzer);

  std::vector<int64_t> dst_shape_int;
  for (size_t i = 0; i < dst_shape.size(); ++i) {
    dst_shape_int.push_back(dst_shape[i].as<IntImmNode>()->value);
  }

  auto elem_bytes = (arr_src->dtype.bits / 8) * arr_src->dtype.lanes;
  std::vector<uint8_t> bytes_src(size_1d * elem_bytes);
  arr_src.CopyToBytes(bytes_src.data(), bytes_src.size());

  std::vector<uint8_t> bytes_dst(bytes_src.size());

  for (size_t i = 0; i < size_1d; ++i) {
    // Convert a linear coordinate to an N-d coordinate tuple
    // z * height * width + y * width + x -> (z, y, x)
    ffi::Array<PrimExpr> src_indices;
    auto div_factor = size_1d;
    auto src_linear_index = i;
    for (auto s : shape) {
      div_factor /= s;
      src_indices.push_back(PrimExpr(static_cast<int>((src_linear_index / div_factor))));
      src_linear_index %= div_factor;
    }
    auto dst_indices = MapIndices(src_indices, &analyzer);

    // Convert an N-d coordinate to a linear coordinate
    // (z, y, x) -> z * height * width + y * width + x
    size_t dst_linear_index = 0;
    auto mul_factor = size_1d;
    for (size_t j = 0; j < dst_indices.size(); ++j) {
      mul_factor /= dst_shape_int[j];
      dst_linear_index += dst_indices[j].as<IntImmNode>()->value * mul_factor;
    }
    std::copy(bytes_src.begin() + i * elem_bytes, bytes_src.begin() + (i + 1) * elem_bytes,
              bytes_dst.begin() + dst_linear_index * elem_bytes);
  }

  auto arr_dst = runtime::Tensor::Empty(dst_shape_int, arr_src->dtype, arr_src->device);
  arr_dst.CopyFromBytes(bytes_dst.data(), bytes_dst.size());
  return arr_dst;
}

IndexMap IndexMap::RenameVariables(
    const std::function<ffi::Optional<ffi::String>(const Var& var)>& f_name_map) const {
  std::unordered_set<std::string> used_names;
  ffi::Map<Var, Var> var_remap;
  NameSupply name_supply;
  const IndexMapNode* n = this->get();
  if (f_name_map != nullptr) {
    // Collect variables with pre-defined names provided by f_name_map.
    std::unordered_set<const Object*> visited;
    std::for_each(n->final_indices.begin(), n->final_indices.end(), [&](const PrimExpr& expr) {
      PostOrderVisit(expr, [&](const ObjectRef& obj) {
        if (!obj->IsInstance<VarNode>()) {
          return;
        }
        if (visited.count(obj.get())) {
          return;
        }
        visited.emplace(obj.get());
        Var var = Downcast<Var>(obj);
        if (ffi::Optional<ffi::String> opt_name = f_name_map(var); opt_name.has_value()) {
          ffi::String name = opt_name.value();
          ICHECK(!name_supply->ContainsName(name, /*add_prefix=*/false));
          name_supply->ReserveName(name, /*add_prefix=*/false);
          var_remap.Set(var, Var(name, var->dtype));
        }
      });
    });
  }

  for (const Var& initial_index : n->initial_indices) {
    if (var_remap.count(initial_index)) {
      // The name of the variable is pre-defined.
      continue;
    }
    ffi::String unique_name =
        name_supply->FreshName(initial_index->name_hint, /*add_prefix=*/false);
    if (unique_name != initial_index->name_hint) {
      var_remap.Set(initial_index, Var(unique_name));
    }
  }

  auto new_initial_indices = n->initial_indices.Map(
      [&](const Var& var) { return Downcast<Var>(Substitute(var, var_remap)); });
  auto new_final_indices =
      n->final_indices.Map([&](const PrimExpr& expr) { return Substitute(expr, var_remap); });
  ffi::Optional<IndexMap> new_inverse_index_map = std::nullopt;
  if (n->inverse_index_map.defined()) {
    new_inverse_index_map = Downcast<IndexMap>(n->inverse_index_map).RenameVariables(f_name_map);
  }
  return IndexMap(new_initial_indices, new_final_indices, new_inverse_index_map);
}

/*!
 * \brief Auxilarry function to convert an index map to lambda expression in Python.
 * \param initial_indices The initial indices in the index map.
 * \param final_indices The final indices in the index map.
 * \return The lambda expression string.
 */
std::string IndexMap2PythonLambdaExpr(const ffi::Array<Var>& initial_indices,
                                      const ffi::Array<PrimExpr>& final_indices) {
  std::unordered_set<std::string> used_names;
  ffi::Map<Var, PrimExpr> var_remap;
  std::ostringstream oss;
  oss << "lambda ";
  for (size_t i = 0; i < initial_indices.size(); ++i) {
    if (i != 0) {
      oss << ", ";
    }
    oss << initial_indices[i];
  }
  oss << ": (";
  for (size_t i = 0; i < final_indices.size(); ++i) {
    if (i != 0) {
      oss << " ";
    }
    oss << final_indices[i];
    oss << ",";
  }
  oss << ")";
  return oss.str();
}

ffi::String IndexMapNode::ToPythonString(
    const std::function<ffi::Optional<ffi::String>(const Var& var)>& f_name_map) const {
  auto index_map = ffi::GetRef<IndexMap>(this).RenameVariables(f_name_map);
  std::string lambda_expr =
      IndexMap2PythonLambdaExpr(index_map->initial_indices, index_map->final_indices);
  if (!index_map->inverse_index_map.defined()) {
    return ffi::String(lambda_expr);
  }
  // Also convert the inverse index map.
  IndexMap inverse = Downcast<IndexMap>(index_map->inverse_index_map.value());
  std::string inverse_lambda_expr =
      IndexMap2PythonLambdaExpr(inverse->initial_indices, inverse->final_indices);
  std::ostringstream oss;
  oss << "tvm.tir.IndexMap.from_func(" << lambda_expr
      << ", inverse_index_map=" << inverse_lambda_expr << ")";
  return ffi::String(oss.str());
}

IndexMap Substitute(const IndexMap& index_map,
                    std::function<ffi::Optional<PrimExpr>(const Var& var)> f_subst) {
  ffi::Array<PrimExpr> new_output =
      index_map->final_indices.Map([&](const PrimExpr& expr) { return Substitute(expr, f_subst); });
  ffi::Optional<IndexMap> new_inverse_map = std::nullopt;
  if (index_map->inverse_index_map.defined()) {
    new_inverse_map = Substitute(Downcast<IndexMap>(index_map->inverse_index_map.value()), f_subst);
  }
  return IndexMap{index_map->initial_indices, new_output, new_inverse_map};
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tir.IndexMap",
           [](ffi::Array<Var> initial_indices, ffi::Array<PrimExpr> final_indices,
              ffi::Optional<IndexMap> inverse_index_map) {
             return IndexMap(initial_indices, final_indices, inverse_index_map);
           })
      .def("tir.IndexMapMapIndices",
           [](IndexMap map, ffi::Array<PrimExpr> indices) {
             arith::Analyzer analyzer;
             return map->MapIndices(indices, &analyzer);
           })
      .def("tir.IndexMapMapShape",
           [](IndexMap map, ffi::Array<PrimExpr> shape) {
             arith::Analyzer analyzer;
             return map->MapShape(shape, &analyzer);
           })
      .def("tir.IndexMapInverse",
           [](IndexMap map, ffi::Array<Range> initial_ranges) {
             arith::Analyzer analyzer;
             return map.Inverse(initial_ranges, &analyzer);
           })
      .def("tir.IndexMapMapTensor",
           [](IndexMap map, runtime::Tensor arr) { return map->MapTensor(arr); })
      .def("tir.IndexMapNonSurjectiveInverse",
           [](IndexMap forward, ffi::Array<Range> initial_ranges) {
             arith::Analyzer analyzer;
             auto result = forward.NonSurjectiveInverse(initial_ranges, &analyzer);
             return ffi::Array<ObjectRef>{result.first, result.second};
           });
}

}  // namespace tir
}  // namespace tvm
