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
 * \file buffer.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/ir/prim/op.h>
#include <tvm/sym/analyzer.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op/memory.h>
#include <tvm/tirx/stmt.h>

#include <iterator>
#include <list>
#include <stack>
#include <utility>

#include "../../sym/pattern_match.h"

namespace tvm {
namespace tirx {
using namespace tvm::prim;

namespace {

using SubscriptSlice = ffi::Array<ffi::Variant<
    ffi::Tuple<ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>>,
    PrimExpr>>;

ffi::ObjectRef RealizeBufferSubscript(
    Expr value,
    ffi::Array<ffi::Variant<
        ffi::Tuple<ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>>,
        PrimExpr>>
        slice,
    Location loc) {
  TensorVar buffer = value.as_or_throw<TensorVar>();
  TensorType buffer_ty = buffer.type();
  TVM_FFI_CHECK_LE(slice.size(), buffer_ty->shape.size(), IndexError)
      << "Too many indices for a " << buffer_ty->shape.size() << "-dimensional buffer";

  bool all_points = slice.size() == buffer_ty->shape.size();
  for (const auto& item : slice) {
    if (auto descriptor = item.as<ffi::Tuple<ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>,
                                             ffi::Optional<PrimExpr>>>()) {
      all_points = false;
      ffi::Optional<PrimExpr> step = descriptor.value().get<2>();
      TVM_FFI_CHECK(!step.has_value() || IsOne(step.value()), ValueError)
          << "Buffer slices with a non-unit step are not supported";
    }
  }

  if (all_points) {
    ffi::Array<PrimExpr> indices;
    indices.reserve(slice.size());
    for (const auto& item : slice) {
      indices.push_back(item.as<PrimExpr>().value());
    }
    return MakeTensorLoad(buffer, indices, loc);
  }

  // Any slice or omitted trailing dimension denotes a region.  Rejecting
  // steps makes the old behavior, where a stride could be silently dropped,
  // unrepresentable rather than giving it dimension-dependent semantics.
  sym::Analyzer analyzer;
  ffi::Array<Range> region;
  region.reserve(buffer_ty->shape.size());
  for (size_t i = 0; i < slice.size(); ++i) {
    if (auto point = slice[i].as<PrimExpr>()) {
      region.push_back(Range::FromMinExtent(point.value(), IntImm(point.value().ty(), 1)));
    } else {
      auto descriptor = slice[i]
                            .as<ffi::Tuple<ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>,
                                           ffi::Optional<PrimExpr>>>()
                            .value();
      PrimExpr start = descriptor.get<0>().value_or(IntImm(buffer_ty->shape[i].ty(), 0));
      PrimExpr stop = descriptor.get<1>().value_or(buffer_ty->shape[i]);
      // Preserve the sole simplification performed by the former Python path.
      region.push_back(Range::FromMinExtent(start, analyzer->Simplify(stop - start)));
    }
  }
  for (size_t i = slice.size(); i < buffer_ty->shape.size(); ++i) {
    region.push_back(
        Range::FromMinExtent(IntImm(buffer_ty->shape[i].ty(), 0), buffer_ty->shape[i]));
  }
  return BufferRegion(buffer, region, loc);
}

ffi::ObjectRef RealizeBufferRegionSubscript(Expr value, SubscriptSlice slice, Location loc) {
  TensorRegion source = value.as_or_throw<TensorRegion>();
  TVM_FFI_CHECK_LE(slice.size(), source->region.size(), IndexError)
      << "Too many indices for a " << source->region.size() << "-dimensional buffer region";

  bool all_points = slice.size() == source->region.size();
  for (const auto& item : slice) {
    if (auto descriptor = item.as<ffi::Tuple<ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>,
                                             ffi::Optional<PrimExpr>>>()) {
      all_points = false;
      ffi::Optional<PrimExpr> step = descriptor.value().get<2>();
      TVM_FFI_CHECK(!step.has_value() || IsOne(step.value()), ValueError)
          << "TensorRegion slices with a non-unit step are not supported";
    }
  }

  if (all_points) {
    ffi::Array<PrimExpr> indices;
    indices.reserve(slice.size());
    for (size_t i = 0; i < slice.size(); ++i) {
      indices.push_back(source->region[i]->min + slice[i].as<PrimExpr>().value());
    }
    return MakeTensorLoad(source->source.as_or_throw<TensorVar>(), indices, loc);
  }

  sym::Analyzer analyzer;
  ffi::Array<Range> region;
  region.reserve(source->region.size());
  for (size_t i = 0; i < slice.size(); ++i) {
    const Range& old_range = source->region[i];
    if (auto point = slice[i].as<PrimExpr>()) {
      PrimExpr new_min = old_range->min + point.value();
      region.push_back(Range::FromMinExtent(new_min, IntImm(point.value().ty(), 1)));
    } else {
      auto descriptor = slice[i]
                            .as<ffi::Tuple<ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>,
                                           ffi::Optional<PrimExpr>>>()
                            .value();
      PrimExpr start = descriptor.get<0>().value_or(IntImm(old_range->extent.ty(), 0));
      PrimExpr stop = descriptor.get<1>().value_or(old_range->extent);
      region.push_back(
          Range::FromMinExtent(old_range->min + start, analyzer->Simplify(stop - start)));
    }
  }
  for (size_t i = slice.size(); i < source->region.size(); ++i) {
    region.push_back(source->region[i]);
  }
  return BufferRegion(source->source.as_or_throw<TensorVar>(), region, loc);
}

}  // namespace

using IndexMod = prim::FloorModNode;
using IndexDiv = prim::FloorDivNode;

TensorRegion BufferRegion(TensorVar buffer, ffi::Array<Range> region, Location loc) {
  TVM_FFI_ICHECK_EQ(buffer->shape.size(), region.size())
      << "Buffer rank and region dimension mismatch";
  return TensorRegion(std::move(buffer), std::move(region), TensorRegionType(), std::move(loc));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.BufferRegion", [](TensorVar buffer, ffi::Array<Range> region) {
    return BufferRegion(buffer, region);
  });
}

TensorRegion FullBufferRegion(TensorVar buffer) {
  ffi::Array<Range> region;
  for (PrimExpr extent : buffer->shape) {
    region.push_back(Range::FromMinExtent(0, extent));
  }
  return BufferRegion(buffer, region);
}

TensorRegion BufferRegionFromPoint(TensorVar buffer, ffi::Array<PrimExpr> indices) {
  ffi::Array<Range> region;
  for (const PrimExpr& index : indices) {
    if (const prim::RampNode* ramp_index = index.as<prim::RampNode>()) {
      region.push_back(
          Range::FromMinExtent(ramp_index->base, ramp_index->stride * ramp_index->lanes));
    } else {
      region.push_back(Range::FromMinExtent(index, MakeConst(index.ty(), 1)));
    }
  }
  return BufferRegion(buffer, region);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::TypeAttrDef<TensorTypeNode>().def(tvm::type_attr::kSubscriptExprRealize,
                                          RealizeBufferSubscript);
  refl::TypeAttrDef<TensorRegionTypeNode>().def(tvm::type_attr::kSubscriptExprRealize,
                                                RealizeBufferRegionSubscript);
}

ffi::Array<PrimExpr> SimplifyArray(sym::AnalyzerObj* ana, ffi::Array<PrimExpr> array) {
  for (size_t i = 0; i < array.size(); ++i) {
    array.Set(i, ana->Simplify(array[i]));
  }
  return array;
}

TensorVar decl_tensor(ffi::Array<PrimExpr> shape, PrimType dtype, ffi::String name,
                      ffi::String storage_scope, Location loc) {
  return TensorVar(name, TensorType(storage_scope, dtype, shape, {}, std::nullopt, 0, 0), loc);
}

// Split the given expression w.r.t the add operator
inline std::vector<const PrimExpr*> ExprSplitAddition(const PrimExpr& expr) {
  using namespace tirx;
  std::vector<const PrimExpr*> ret;
  std::stack<const PrimExpr*> split_buffer;
  split_buffer.push(&expr);
  while (!split_buffer.empty()) {
    const PrimExpr* top_ele = split_buffer.top();
    split_buffer.pop();
    auto expr_add_match = top_ele->as<prim::AddNode>();
    if (expr_add_match) {
      split_buffer.push(&expr_add_match->b);
      split_buffer.push(&expr_add_match->a);
    } else {
      ret.emplace_back(top_ele);
    }
  }
  return ret;
}

// Searches for the following types of expr:
//   mult_expr = (a1 + a2 + ... + aj + c1 / (k1 * k2 * ... * ki) * k1 * ... * kt-1 ) * kt * ... * ki
//   mod_l_expr = c2
//   mod_r_expr = k1 * k2 * ... * ki
//   where c1 ~= c2 mod k1 * k2 * ... * ki
// If it can be optimized, returns (true, (a1 + a2 + ... + aj) * kt * ... * ki + c1)
// Currently the we will not search the add/mult combinations exhaustively
//   as it will take too much computation.
inline ffi::Optional<PrimExpr> MergeMulModInner(sym::AnalyzerObj* analyzer,
                                                const PrimExpr& mult_expr,
                                                const PrimExpr& mod_l_expr,
                                                const PrimExpr& mod_r_expr) {
  using namespace tirx;
  const prim::MulNode* mult_ptr = mult_expr.as<prim::MulNode>();
  if (!mult_ptr) return std::nullopt;
  PrimExpr mult_outer = mult_ptr->b;
  const PrimExpr* inner = &(mult_ptr->a);
  // 1. Calculate the outer multiplier
  while (true) {
    mult_ptr = inner->as<prim::MulNode>();
    if (mult_ptr) {
      inner = &(mult_ptr->a);
      mult_outer = mult_ptr->b * mult_outer;
    } else {
      break;
    }
  }
  // 2. Search for the pattern c / (...) * (...) + c % (...)
  // We match the search element with Add, Mul and Div.
  //   If Add is found, we need to continue our search for the rhs
  //   If Mult is found, we will expand the inner multiplication factor
  //   If Div is found, we will go on testing whether lhs matches the lhs of mod expr
  //      and returns the optimization result.
  const PrimExpr* search_ptr = inner;
  ffi::Optional<PrimExpr> mult_inner;  // The inner multiplication factor
  ffi::Optional<PrimExpr> no_opt_sum;  // Sum of the exprs that cannot be optimized
  prim::ExprDeepEqual expr_equal;

  while (true) {
    auto inner_div_ptr = search_ptr->as<IndexDiv>();
    auto inner_mult_ptr = search_ptr->as<prim::MulNode>();
    auto inner_add_ptr = search_ptr->as<prim::AddNode>();
    if (!inner_div_ptr && !inner_mult_ptr && !inner_add_ptr) {
      return std::nullopt;
    } else if (inner_div_ptr) {
      PrimExpr overall_mult = mult_inner.has_value() ? mult_inner.value() * mult_outer : mult_outer;
      if (expr_equal(overall_mult, inner_div_ptr->b) && expr_equal(overall_mult, mod_r_expr) &&
          analyzer->CanProveEqual(floormod(inner_div_ptr->a - mod_l_expr, mod_r_expr), 0)) {
        // Found!
        PrimExpr ret = no_opt_sum.has_value() ? no_opt_sum.value() * mult_outer + inner_div_ptr->a
                                              : inner_div_ptr->a;
        return ret;
      } else {
        return std::nullopt;
      }
    } else if (inner_mult_ptr) {
      mult_inner =
          mult_inner.has_value() ? inner_mult_ptr->b * mult_inner.value() : inner_mult_ptr->b;
      search_ptr = &(inner_mult_ptr->a);
    } else if (inner_add_ptr) {
      if (mult_inner.has_value()) {
        return std::nullopt;
      }
      no_opt_sum =
          no_opt_sum.has_value() ? no_opt_sum.value() + inner_add_ptr->a : inner_add_ptr->a;
      search_ptr = &(inner_add_ptr->b);
    } else {
      TVM_FFI_THROW(InternalError) << "Unexpected search result!";
      break;
    }
  }
  return std::nullopt;
}

// Insert the elements into the corresponding mult_exprs and mod_exprs.
// If the element is found to match Mul, it will be pushed to the mult_exprs.
// If the element it found to match Mod, it will be pused to the mod_exprs.
// Otherwise, the elements will be added to the no_opt_sum variable
inline void MergeMulModInsertElements(const std::vector<const PrimExpr*>& eles,
                                      std::list<PrimExpr>* mult_exprs,
                                      std::list<std::pair<PrimExpr, PrimExpr>>* mod_exprs,
                                      ffi::Optional<PrimExpr>* no_opt_sum, bool* has_mult,
                                      bool* has_mod) {
  using namespace tirx;
  *has_mult = false;
  *has_mod = false;
  for (const PrimExpr* ele : eles) {
    auto mod_ptr = ele->as<IndexMod>();
    auto mult_ptr = ele->as<prim::MulNode>();
    if (mod_ptr) {
      *has_mod = true;
      mod_exprs->emplace_back(std::make_pair(std::move(mod_ptr->a), std::move(mod_ptr->b)));
    } else if (mult_ptr) {
      *has_mult = true;
      mult_exprs->emplace_back(*ele);
    } else {
      *no_opt_sum = no_opt_sum->has_value() ? no_opt_sum->value() + *ele : *ele;
    }
  }
}

// Searches for this types of expr:
//   (a1 + a2 + ... + aj + c / (k1 * k2 * ... * ki) * k1 * ... * kt-1 ) * kt * ... * ki
//   + c % (k1 * k2 * ... * ki)
// and simplifies to (a1 + a2 + ... + aj) * kt * ... * ki + c
// The search will be performed repeatively until no pattern is found.
// Return the simplified expression, retaining the input when no merge is possible.
inline PrimExpr MergeMulMod(sym::AnalyzerObj* analyzer, const PrimExpr& base) {
  using namespace tirx;
  // 1. Prepare the lists.
  // We store two lists, a list that contain all the elements that match Mul and
  //                     a list that contain all the elements that match Mod.
  // The elements in the Mod will be used to match against the elements in Mul.
  // The result will then be split and pushed back to these two lists.
  PrimExpr simplified_base = base;
  sym::PVar<PrimExpr> x, y;
  if ((floordiv(x, y) * y + floormod(x, y)).Match(simplified_base)) {
    simplified_base = x.Eval();
  }
  simplified_base = analyzer->Simplify(simplified_base);
  std::vector<const PrimExpr*> eles = ExprSplitAddition(simplified_base);
  std::list<PrimExpr> mult_exprs;
  std::list<std::pair<PrimExpr, PrimExpr>> mod_exprs;
  ffi::Optional<PrimExpr> no_opt_sum;
  bool has_mult;
  bool has_mod;
  MergeMulModInsertElements(eles, &mult_exprs, &mod_exprs, &no_opt_sum, &has_mult, &has_mod);
  bool find_opt = false;
  std::list<std::pair<PrimExpr, PrimExpr>>::iterator search_mod_it = mod_exprs.begin();
  // 2. Exhaustive Search
  while (search_mod_it != mod_exprs.end()) {
    std::list<PrimExpr>::iterator mult_it = mult_exprs.begin();
    bool inner_find_opt = false;
    while (mult_it != mult_exprs.end()) {
      ffi::Optional<PrimExpr> ret =
          MergeMulModInner(analyzer, *mult_it, search_mod_it->first, search_mod_it->second);
      if (ret.has_value()) {
        inner_find_opt = true;
        auto temp_mod_it = search_mod_it;
        ++search_mod_it;
        mod_exprs.erase(temp_mod_it);
        mult_exprs.erase(mult_it);
        std::vector<const PrimExpr*> ret_eles = ExprSplitAddition(ret.value());
        MergeMulModInsertElements(ret_eles, &mult_exprs, &mod_exprs, &no_opt_sum, &has_mult,
                                  &has_mod);
        if (has_mult) {
          search_mod_it = mod_exprs.begin();
        } else if (has_mod && search_mod_it == mod_exprs.end()) {
          search_mod_it--;
        }
        break;
      } else {
        ++mult_it;
      }
    }
    find_opt = find_opt || inner_find_opt;
    if (!inner_find_opt) {
      ++search_mod_it;
    }
  }
  if (!find_opt) {
    return simplified_base;
  }
  for (std::list<PrimExpr>::iterator it = mult_exprs.begin(); it != mult_exprs.end(); ++it) {
    no_opt_sum = no_opt_sum.has_value() ? no_opt_sum.value() + *it : *it;
  }
  for (std::list<std::pair<PrimExpr, PrimExpr>>::iterator it = mod_exprs.begin();
       it != mod_exprs.end(); ++it) {
    no_opt_sum = no_opt_sum.has_value() ? no_opt_sum.value() + indexmod(it->first, it->second)
                                        : indexmod(it->first, it->second);
  }
  return no_opt_sum.value();
}

// The buffer offset in convention of number of elements of
// original data ignoring number of lanes.
// We also perform optimization to simplify the indexing expression.
ffi::Array<PrimExpr> TensorTypeNode::ElemOffset(ffi::Array<PrimExpr> input_indices,
                                                bool inner) const {
  TVM_FFI_ICHECK_EQ(shape.size(), input_indices.size())
      << "TensorType is " << shape.size() << "-dimensional, cannot be indexed with the "
      << input_indices.size() << "-dimensional indices provided.";

  if (strides.size()) {
    TVM_FFI_ICHECK_EQ(this->strides.size(), input_indices.size())
        << "If strides are defined, "
        << "the index's dimensionality must match the dimensionality of the index given.";
  }

  PrimExpr output_index = 0;
  sym::Analyzer ana;

  for (size_t i = 0; i < input_indices.size(); i++) {
    if (strides.size()) {
      output_index = output_index + input_indices[i] * strides[i];
    } else {
      output_index = output_index * this->shape[i] + input_indices[i];
    }

    if (i > 0) {
      output_index = MergeMulMod(ana.get(), output_index);
    }
  }

  if (elem_offset.defined() && !IsZero(elem_offset) && !inner) {
    output_index = output_index + elem_offset;
  }

  return SimplifyArray(ana.get(), {output_index});
}

TensorVar::TensorVar(ffi::String name, TensorType type, Location loc)
    : Var(Var(std::move(name), std::move(type), std::move(loc))) {}

tirx::TensorVar TensorWithOffsetAlignment(ffi::Array<PrimExpr> shape, PrimType dtype,
                                          std::string name, int data_alignment, int offset_factor,
                                          std::string memory_scope) {
  ffi::Optional<PrimExpr> elem_offset;
  if (offset_factor != 0) {
    elem_offset = PrimVar(name + "_elem_offset", shape[0].ty());
  }

  return tirx::TensorVar(
      name, TensorType(memory_scope, dtype, shape, {}, elem_offset, data_alignment, offset_factor));
}

bool TensorVar::IsScalar(bool alloc_or_decl) const { return type()->IsScalar(alloc_or_decl); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.TensorVar",
           [](ffi::String name, TensorType type, Location loc) {
             return TensorVar(std::move(name), std::move(type), std::move(loc));
           })
      .def_method("tirx.TensorStorageScope", &TensorVar::scope)
      .def_method("tirx.TensorIsScalar", &TensorVar::IsScalar)
      .def_method("tirx.TensorData", &TensorVar::data);
}

}  // namespace tirx
}  // namespace tvm
