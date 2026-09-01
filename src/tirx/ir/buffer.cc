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
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/prim/builtin.h>
#include <tvm/ir/prim/expr.h>
#include <tvm/runtime/device_api.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>

#include <iterator>
#include <list>
#include <stack>

#include "../../arith/pattern_match.h"

namespace tvm {
namespace tirx {

namespace {

ffi::ObjectRef RealizeBufferSubscript(
    Expr value,
    ffi::Array<ffi::Variant<
        ffi::Tuple<ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>>,
        PrimExpr>>
        slice,
    Span span) {
  BufferVar buffer = value.as_or_throw<BufferVar>();
  BufferType buffer_ty = buffer.type();
  TVM_FFI_CHECK_LE(slice.size(), buffer_ty->shape.size(), IndexError)
      << "Too many indices for a " << buffer_ty->shape.size() << "-dimensional buffer";

  bool all_points = slice.size() == buffer_ty->shape.size();
  for (const auto& item : slice) {
    if (auto descriptor = item.as<ffi::Tuple<ffi::Optional<PrimExpr>, ffi::Optional<PrimExpr>,
                                             ffi::Optional<PrimExpr>>>()) {
      all_points = false;
      ffi::Optional<PrimExpr> step = descriptor.value().get<2>();
      TVM_FFI_CHECK(!step.has_value() || is_one(step.value()), ValueError)
          << "Buffer slices with a non-unit step are not supported";
    }
  }

  if (all_points) {
    ffi::Array<PrimExpr> indices;
    indices.reserve(slice.size());
    for (const auto& item : slice) {
      indices.push_back(item.as<PrimExpr>().value());
    }
    return BufferLoad(buffer, indices, span);
  }

  // Any slice or omitted trailing dimension denotes a region.  Rejecting
  // steps makes the old behavior, where a stride could be silently dropped,
  // unrepresentable rather than giving it dimension-dependent semantics.
  arith::Analyzer analyzer;
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
  return BufferRegion(buffer, region);
}

}  // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  BufferTypeNode::RegisterReflection();
  refl::TypeAttrDef<BufferTypeNode>().def("__subscript_expr_realize__", RealizeBufferSubscript);
}

using IndexMod = prim::FloorModNode;
using IndexDiv = prim::FloorDivNode;

BufferType::BufferType(ffi::String storage_scope, PrimType dtype, ffi::Array<PrimExpr> shape,
                       ffi::Array<PrimExpr> strides, PrimExpr elem_offset, int data_alignment,
                       int offset_factor, ffi::Optional<Layout> layout,
                       ffi::Array<PrimExpr> allocated_addr, Span span)
    : Type(ffi::UnsafeInit{}) {
  auto n = ffi::make_object<BufferTypeNode>();
  n->dtype = std::move(dtype);
  n->storage_scope = storage_scope.empty() ? ffi::String("global") : std::move(storage_scope);
  n->shape = std::move(shape);
  n->strides = std::move(strides);
  if (!elem_offset.defined()) {
    elem_offset = IntImm(PrimType(n->DefaultIndexType()), 0);
  }
  n->elem_offset = std::move(elem_offset);
  n->data_alignment =
      data_alignment <= 0 ? static_cast<int>(runtime::kAllocAlignment) : data_alignment;
  n->offset_factor = offset_factor == 0 ? 1 : offset_factor;
  n->layout = std::move(layout);
  n->allocated_addr = std::move(allocated_addr);
  n->span = std::move(span);
  data_ = std::move(n);
}

namespace {

BufferVar RebuildBufferVarFromType(const BufferVar& buffer, BufferType type,
                                   ffi::String name_suffix = "") {
  return BufferVar(buffer.name() + name_suffix, std::move(type), buffer.span());
}

}  // namespace

ffi::Array<PrimExpr> SimplifyArray(arith::AnalyzerObj* ana, ffi::Array<PrimExpr> array) {
  for (size_t i = 0; i < array.size(); ++i) {
    array.Set(i, ana->Simplify(array[i]));
  }
  return array;
}

BufferVar decl_buffer(ffi::Array<PrimExpr> shape, PrimType dtype, ffi::String name,
                      ffi::String storage_scope, Span span) {
  return BufferVar(name, BufferType(storage_scope, dtype, shape, {}, PrimExpr(), 0, 0), span);
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
inline std::pair<bool, PrimExpr> MergeMulModInner(arith::AnalyzerObj* analyzer,
                                                  const PrimExpr& mult_expr,
                                                  const PrimExpr& mod_l_expr,
                                                  const PrimExpr& mod_r_expr) {
  using namespace tirx;
  const prim::MulNode* mult_ptr = mult_expr.as<prim::MulNode>();
  if (!mult_ptr) return std::make_pair(false, PrimExpr());
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
  PrimExpr mult_inner;  // The inner multiplication factor
  PrimExpr no_opt_sum;  // Sum of the exprs that cannot be optimized
  tirx::ExprDeepEqual expr_equal;

  while (true) {
    auto inner_div_ptr = search_ptr->as<IndexDiv>();
    auto inner_mult_ptr = search_ptr->as<prim::MulNode>();
    auto inner_add_ptr = search_ptr->as<prim::AddNode>();
    if (!inner_div_ptr && !inner_mult_ptr && !inner_add_ptr) {
      return std::make_pair(false, PrimExpr());
    } else if (inner_div_ptr) {
      PrimExpr overall_mult = mult_inner.get() ? mult_inner * mult_outer : mult_outer;
      if (expr_equal(overall_mult, inner_div_ptr->b) && expr_equal(overall_mult, mod_r_expr) &&
          analyzer->CanProveEqual(floormod(inner_div_ptr->a - mod_l_expr, mod_r_expr), 0)) {
        // Found!
        PrimExpr ret =
            no_opt_sum.get() ? no_opt_sum * mult_outer + inner_div_ptr->a : inner_div_ptr->a;
        return std::make_pair(true, ret);
      } else {
        return std::make_pair(false, PrimExpr());
      }
    } else if (inner_mult_ptr) {
      mult_inner = mult_inner.get() ? inner_mult_ptr->b * mult_inner : inner_mult_ptr->b;
      search_ptr = &(inner_mult_ptr->a);
    } else if (inner_add_ptr) {
      if (mult_inner.get()) {
        return std::make_pair(false, PrimExpr());
      }
      no_opt_sum = no_opt_sum.get() ? no_opt_sum + inner_add_ptr->a : inner_add_ptr->a;
      search_ptr = &(inner_add_ptr->b);
    } else {
      TVM_FFI_THROW(InternalError) << "Unexpected search result!";
      break;
    }
  }
  return std::make_pair(false, PrimExpr());
}

// Insert the elements into the corresponding mult_exprs and mod_exprs.
// If the element is found to match Mul, it will be pushed to the mult_exprs.
// If the element it found to match Mod, it will be pused to the mod_exprs.
// Otherwise, the elements will be added to the no_opt_sum variable
inline void MergeMulModInsertElements(const std::vector<const PrimExpr*>& eles,
                                      std::list<PrimExpr>* mult_exprs,
                                      std::list<std::pair<PrimExpr, PrimExpr>>* mod_exprs,
                                      PrimExpr* no_opt_sum, bool* has_mult, bool* has_mod) {
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
      *no_opt_sum = no_opt_sum->get() ? *no_opt_sum + *ele : *ele;
    }
  }
}

// Searches for this types of expr:
//   (a1 + a2 + ... + aj + c / (k1 * k2 * ... * ki) * k1 * ... * kt-1 ) * kt * ... * ki
//   + c % (k1 * k2 * ... * ki)
// and simplifies to (a1 + a2 + ... + aj) * kt * ... * ki + c
// The search will be performed repeatively until no pattern is found.
// Return: a pair with (false, Expr()) if cannot be optimized.
//         a pair with (true, optimized_expr) if can be optimized
inline PrimExpr MergeMulMod(arith::AnalyzerObj* analyzer, const PrimExpr& base) {
  using namespace tirx;
  // 1. Prepare the lists.
  // We store two lists, a list that contain all the elements that match Mul and
  //                     a list that contain all the elements that match Mod.
  // The elements in the Mod will be used to match against the elements in Mul.
  // The result will then be split and pushed back to these two lists.
  PrimExpr simplified_base = base;
  arith::PVar<PrimExpr> x, y;
  if ((floordiv(x, y) * y + floormod(x, y)).Match(simplified_base)) {
    simplified_base = x.Eval();
  }
  simplified_base = analyzer->Simplify(simplified_base);
  std::vector<const PrimExpr*> eles = ExprSplitAddition(simplified_base);
  std::list<PrimExpr> mult_exprs;
  std::list<std::pair<PrimExpr, PrimExpr>> mod_exprs;
  PrimExpr no_opt_sum;
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
      std::pair<bool, PrimExpr> ret =
          MergeMulModInner(analyzer, *mult_it, search_mod_it->first, search_mod_it->second);
      if (ret.first) {
        inner_find_opt = true;
        auto temp_mod_it = search_mod_it;
        ++search_mod_it;
        mod_exprs.erase(temp_mod_it);
        mult_exprs.erase(mult_it);
        std::vector<const PrimExpr*> ret_eles = ExprSplitAddition(ret.second);
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
    no_opt_sum = no_opt_sum.get() ? no_opt_sum + *it : *it;
  }
  for (std::list<std::pair<PrimExpr, PrimExpr>>::iterator it = mod_exprs.begin();
       it != mod_exprs.end(); ++it) {
    no_opt_sum = no_opt_sum.get() ? no_opt_sum + indexmod(it->first, it->second)
                                  : indexmod(it->first, it->second);
  }
  return no_opt_sum;
}

ffi::Array<PrimExpr> BufferVar::OffsetOf(ffi::Array<PrimExpr> input_indices) const {
  return (*this)->ElemOffset(std::move(input_indices));
}

// The buffer offset in convention of number of elements of
// original data ignoring number of lanes.
// We also perform optimization to simplify the indexing expression.
ffi::Array<PrimExpr> BufferTypeNode::ElemOffset(ffi::Array<PrimExpr> input_indices,
                                                bool inner) const {
  TVM_FFI_ICHECK_EQ(shape.size(), input_indices.size())
      << "BufferType is " << shape.size() << "-dimensional, cannot be indexed with the "
      << input_indices.size() << "-dimensional indices provided.";

  if (strides.size()) {
    TVM_FFI_ICHECK_EQ(this->strides.size(), input_indices.size())
        << "If strides are defined, "
        << "the index's dimensionality must match the dimensionality of the index given.";
  }

  PrimExpr output_index = 0;
  arith::Analyzer ana;

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

  if (elem_offset.defined() && !is_zero(elem_offset) && !inner) {
    output_index = output_index + elem_offset;
  }

  return SimplifyArray(ana.get(), {output_index});
}

inline ffi::Array<PrimExpr> BufferOffset(const BufferTypeNode* n, ffi::Array<PrimExpr> index,
                                         PrimType dtype) {
  ffi::Array<PrimExpr> offsets = n->ElemOffset(index);
  // If the BufferVar has element type with more than one lane, scale to
  // get the offset in number of scalars.
  if (PrimType(n->dtype).lanes() != 1) {
    PrimExpr last_offset = offsets[offsets.size() - 1];
    offsets.Set(offsets.size() - 1, last_offset * MakeConst(last_offset.ty(), dtype.lanes()));
  }

  // If the requested type has more than one lane, make a RampNode at
  // that offset.
  if (dtype.lanes() != 1) {
    PrimExpr last_offset = offsets[offsets.size() - 1];
    PrimExpr stride = MakeConst(last_offset.ty(), 1);
    offsets.Set(offsets.size() - 1, prim::Ramp(last_offset, stride, dtype.lanes()));
  }

  return offsets;
}

BufferVar BufferVar::GetFlattenedBuffer() const {
  auto self = operator->();

  ffi::Array<PrimExpr> output_shape{1};
  if (self->strides.size()) {
    // If strides are defined, the flattened extent is the span of the
    // outermost input axis.
    TVM_FFI_ICHECK_EQ(self->shape.size(), self->strides.size());
    output_shape.Set(0, self->strides[0] * self->shape[0]);
  } else {
    // Otherwise, the flattened extent is the product of the input extents.
    // This also flattens rank-0 tensors to a rank-1 buffer of shape [1].
    for (size_t i = 0; i < self->shape.size(); i++) {
      output_shape.Set(0, output_shape[0] * self->shape[i]);
    }
  }

  if (output_shape.size() == self->shape.size() && self->strides.empty()) {
    return *this;
  } else {
    // Keep `layout` in sync with `shape`. The old layout describes the
    // pre-flatten N-D shape (e.g. `S[(16,16):(16,1)]`); after collapsing
    // shape to 1-D, that layout no longer matches the buffer's rank and
    // structural compares against a freshly-decl'd 1-D buffer would diff
    // (see test_tir_transform_flatten_buffer). Reset to the default layout
    // for the new shape so the buffer stays internally consistent.
    return RebuildBufferVarFromType(
        *this, BufferType(self->storage_scope, self->dtype, output_shape, {}, self->elem_offset,
                          self->data_alignment, self->offset_factor,
                          TileLayoutNode::DefaultLayout(output_shape), self->allocated_addr));
  }
}

PrimExpr BufferVar::vload(ffi::Array<PrimExpr> begin, PrimType value_dtype) const {
  const BufferTypeNode* n = operator->();
  TVM_FFI_ICHECK(n != nullptr);
  PrimType buffer_dtype(n->dtype);
  int value_lanes =
      value_dtype.IsScalableVector() ? value_dtype.VScaleFactor() : value_dtype.lanes();
  int buffer_lanes =
      buffer_dtype.IsScalableVector() ? buffer_dtype.VScaleFactor() : buffer_dtype.lanes();
  TVM_FFI_ICHECK(value_dtype.WithLanes(1)->dtype == buffer_dtype.WithLanes(1)->dtype &&
                 value_lanes % buffer_lanes == 0)
      << "Cannot load " << value_dtype << " from buffer of " << n->dtype;

  ffi::Array<PrimExpr> indices = begin;
  PrimExpr base = indices[indices.size() - 1];
  if (value_dtype.IsFixedLengthVector()) {
    int factor = value_dtype.lanes() / buffer_dtype.lanes();
    PrimType base_ty = base.ty();
    if (factor > 1 && !base_ty.IsFixedLengthVector() && !base_ty.IsScalableVector()) {
      indices.Set(indices.size() - 1, prim::Ramp(base, 1, factor));
    }
  }
  return BufferLoad(*this, indices);
}

Stmt BufferVar::vstore(ffi::Array<PrimExpr> begin, PrimExpr value) const {
  const BufferTypeNode* n = operator->();
  TVM_FFI_ICHECK(n != nullptr);
  PrimType value_dtype = value.ty();
  PrimType buffer_dtype(n->dtype);
  int value_lanes =
      value_dtype.IsScalableVector() ? value_dtype.VScaleFactor() : value_dtype.lanes();
  int buffer_lanes =
      buffer_dtype.IsScalableVector() ? buffer_dtype.VScaleFactor() : buffer_dtype.lanes();
  TVM_FFI_ICHECK(value_dtype.WithLanes(1)->dtype == buffer_dtype.WithLanes(1)->dtype &&
                 value_lanes % buffer_lanes == 0)
      << "Cannot store " << value_dtype << " to buffer of " << n->dtype;

  ffi::Array<PrimExpr> indices = begin;
  PrimExpr base = indices[indices.size() - 1];
  if (value_dtype.IsFixedLengthVector()) {
    int factor = value_dtype.lanes() / buffer_dtype.lanes();
    PrimType base_ty = base.ty();
    if (factor > 1 && !base_ty.IsFixedLengthVector() && !base_ty.IsScalableVector()) {
      indices.Set(indices.size() - 1, prim::Ramp(base, 1, factor));
    }
  }
  return BufferStore(*this, value, indices);
}

ffi::String BufferVar::scope() const { return (*this)->storage_scope; }

BufferVar BufferVar::MakeStrideView() const {
  if ((*this)->strides.size() != 0) return *this;
  if ((*this)->shape.size() == 0) return *this;
  const BufferTypeNode* self = operator->();
  TVM_FFI_ICHECK(self != nullptr);
  PrimExpr acc = IntImm(PrimType(self->DefaultIndexType()), 1);
  std::vector<PrimExpr> temp;
  for (size_t i = self->shape.size(); i != 0; --i) {
    temp.push_back(acc);
    acc = acc * self->shape[i - 1];
  }
  ffi::Array<PrimExpr> strides;
  for (size_t i = temp.size(); i != 0; --i) {
    strides.push_back(temp[i - 1]);
  }
  return RebuildBufferVarFromType(
      *this, BufferType(self->storage_scope, self->dtype, self->shape, std::move(strides),
                        self->elem_offset, self->data_alignment, self->offset_factor, self->layout,
                        self->allocated_addr));
}

BufferVar BufferVar::MakeSlice(ffi::Array<PrimExpr> begins, ffi::Array<PrimExpr> extents) const {
  const BufferTypeNode* n = operator->();
  TVM_FFI_ICHECK(n != nullptr);
  arith::Analyzer ana;
  begins = SimplifyArray(ana.get(), begins);
  ffi::Array<PrimExpr> elem_offset =
      n->ElemOffset(begins).Map([&](const PrimExpr& expr) { return ana->Simplify(expr); });

  ffi::Array<PrimExpr> strides = n->strides;
  if (strides.size() == 0) {
    bool can_relax = true;
    bool need_stride = false;
    // check if stride is needed.
    for (size_t i = 0; i < extents.size(); ++i) {
      if (!can_relax) {
        if (!is_zero(begins[i]) || !is_zero(ana->Simplify(extents[i] - n->shape[i]))) {
          need_stride = true;
        }
      }
      if (!is_one(extents[i])) can_relax = false;
    }
    // make stride.
    if (need_stride) {
      return MakeStrideView().MakeSlice(begins, extents);
    }
  }
  return RebuildBufferVarFromType(
      *this,
      BufferType(n->storage_scope, n->dtype, extents, strides, elem_offset[0], n->data_alignment, 0,
                 TileLayoutNode::DefaultLayout(extents)),
      "_slice");
}

Expr BufferVar::access_ptr(int access_mask, PointerType ptr_type, int content_lanes,
                           PrimExpr offset, ffi::Optional<PrimExpr> input_extent) const {
  const BufferTypeNode* self = operator->();
  TVM_FFI_ICHECK(self != nullptr);
  // An access pointer addresses the same allocation as the buffer data.  The
  // requested type controls its pointee, while the buffer controls its address
  // space (for example, shared or local memory).
  ptr_type = PointerType(ptr_type->element_type, self->storage_scope);
  PrimExpr e_dtype;
  PrimExpr extent;
  if (self->shape.size() == 0) {
    extent = IntImm(PrimType(self->DefaultIndexType()), 1);
  } else if (self->strides.size() == self->shape.size()) {
    int highest_dim = 0;
    extent = self->strides[highest_dim] * self->shape[highest_dim] - offset;
  } else {
    extent = foldl([](PrimExpr a, PrimExpr b, Span span) { return mul(a, b, span); },
                   IntImm::Int32(1), self->shape) -
             offset;
  }
  PrimExpr elem_offset = self->elem_offset + offset;
  if (content_lanes > 1) {
    e_dtype = tirx::TypeAnnotation(PrimType(self->dtype).WithLanes(content_lanes));
    extent = extent / MakeConst(self->elem_offset.ty(), content_lanes);
    elem_offset = self->elem_offset / MakeConst(self->elem_offset.ty(), content_lanes);
  } else {
    e_dtype = tirx::TypeAnnotation(self->dtype);
  }

  if (input_extent.has_value()) {
    extent = input_extent.value();
  }
  ffi::Array<Expr> acc_args{e_dtype, data(), elem_offset, extent, IntImm::Int32(access_mask)};
  return Call(ptr_type, tirx::builtin::tvm_access_ptr(), acc_args);
}

BufferVar::BufferVar(ffi::String name, BufferType type, Span span)
    : Var(Var(std::move(name), std::move(type), std::move(span))) {}

Expr BufferVar::data() const { return Call(DataPointerType(), builtin::buffer_data(), {var()}); }

tirx::BufferVar BufferWithOffsetAlignment(ffi::Array<PrimExpr> shape, PrimType dtype,
                                          std::string name, int data_alignment, int offset_factor,
                                          std::string memory_scope) {
  PrimExpr elem_offset;
  if (offset_factor != 0) {
    elem_offset = tirx::PrimVar(name + "_elem_offset", shape[0].ty());
  } else {
    elem_offset = PrimExpr();
  }

  return tirx::BufferVar(
      name, BufferType(memory_scope, dtype, shape, {}, elem_offset, data_alignment, offset_factor));
}

BufferVar BufferVar::with_allocated_addr(ffi::Array<PrimExpr> allocated_addr) const {
  const auto* self = operator->();
  return RebuildBufferVarFromType(
      *this, BufferType(self->storage_scope, self->dtype, self->shape, self->strides,
                        self->elem_offset, self->data_alignment, self->offset_factor, self->layout,
                        std::move(allocated_addr)));
}

BufferVar BufferVar::with_dtype(PrimType dtype) const {
  const auto* self = operator->();
  return RebuildBufferVarFromType(
      *this, BufferType(self->storage_scope, std::move(dtype), self->shape, self->strides,
                        self->elem_offset, self->data_alignment, self->offset_factor, self->layout,
                        self->allocated_addr));
}

PrimExpr BufferVar::OffsetOf_p(const Array<PrimExpr>& indices) const {
  return Call(PrimType::Int(32), tirx::builtin::buffer_offset(), {BufferLoad(*this, indices)})
      .as_or_throw<PrimExpr>();
}

bool BufferVar::IsScalar(bool alloc_or_decl) const {
  // TODO(@bohan): logical scope is not considered
  return (*this)->shape.size() == 1 && is_one((*this)->shape[0]) && (*this)->strides.size() == 0 &&
         (!alloc_or_decl || tirx::is_zero((*this)->elem_offset)) && (*this)->data_alignment == 64 &&
         (*this)->offset_factor == 1 && (*this)->allocated_addr.size() == 0 &&
         (*this)->layout.has_value() &&
         ffi::StructuralEqual()((*this)->layout.value(), TileLayoutNode::DefaultLayout({1}));
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.BufferVar",
           [](ffi::String name, BufferType type, Span span) {
             return BufferVar(std::move(name), std::move(type), std::move(span));
           })
      .def_method(
          "tirx.BufferAccessPtr",
          static_cast<Expr (BufferVar::*)(int, PointerType, int, PrimExpr, ffi::Optional<PrimExpr>)
                          const>(&BufferVar::access_ptr))
      .def_method("tirx.BufferGetFlattenedBuffer", &BufferVar::GetFlattenedBuffer)
      .def_method("tirx.BufferOffsetOf", &BufferVar::OffsetOf)
      .def_method("tirx.BufferOffsetOfp", &BufferVar::OffsetOf_p)
      .def_method("tirx.BufferVLoad", &BufferVar::vload)
      .def_method("tirx.BufferVStore", &BufferVar::vstore)
      .def_method("tirx.BufferStorageScope", &BufferVar::scope)
      .def_method("tirx.BufferWithAllocatedAddr", &BufferVar::with_allocated_addr)
      .def_method("tirx.BufferWithDtype", &BufferVar::with_dtype)
      .def_method("tirx.BufferIsScalar", &BufferVar::IsScalar)
      .def_method("tirx.BufferData", &BufferVar::data)
      .def_method("tirx.BufferDataPointerType", &BufferVar::DataPointerType)
      .def("tirx.BufferType", [](ffi::String storage_scope, PrimType dtype,
                                 ffi::Array<PrimExpr> shape, ffi::Array<PrimExpr> strides,
                                 PrimExpr elem_offset, int data_alignment, int offset_factor,
                                 ffi::Optional<Layout> layout, ffi::Array<PrimExpr> allocated_addr,
                                 Span span) {
        return BufferType(std::move(storage_scope), std::move(dtype), std::move(shape),
                          std::move(strides), std::move(elem_offset), data_alignment, offset_factor,
                          std::move(layout), std::move(allocated_addr), std::move(span));
      });
}

}  // namespace tirx
}  // namespace tvm
