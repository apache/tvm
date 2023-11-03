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
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/buffer.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

#include <iterator>
#include <stack>

#include "../../arith/pattern_match.h"

namespace tvm {
namespace tir {

using IndexMod = tir::FloorModNode;
using IndexDiv = tir::FloorDivNode;

Array<PrimExpr> SimplifyArray(arith::Analyzer* ana, Array<PrimExpr> array) {
  for (size_t i = 0; i < array.size(); ++i) {
    array.Set(i, ana->Simplify(array[i]));
  }
  return array;
}

Buffer decl_buffer(Array<PrimExpr> shape, DataType dtype, String name, String storage_scope,
                   Array<IntImm> axis_separators, Span span) {
  DataType storage_dtype = (dtype == DataType::Bool() ? DataType::Int(8) : dtype);
  return Buffer(Var(name, PointerType(PrimType(storage_dtype), storage_scope), span), dtype, shape,
                Array<PrimExpr>(), PrimExpr(), name, 0, 0, kDefault, axis_separators, span);
}

// Split the given expression w.r.t the add operator
inline std::vector<const PrimExpr*> ExprSplitAddition(const PrimExpr& expr) {
  using namespace tir;
  std::vector<const PrimExpr*> ret;
  std::stack<const PrimExpr*> split_buffer;
  split_buffer.push(&expr);
  while (!split_buffer.empty()) {
    const PrimExpr* top_ele = split_buffer.top();
    split_buffer.pop();
    auto expr_add_match = top_ele->as<AddNode>();
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
inline std::pair<bool, PrimExpr> MergeMulModInner(arith::Analyzer* analyzer,
                                                  const PrimExpr& mult_expr,
                                                  const PrimExpr& mod_l_expr,
                                                  const PrimExpr& mod_r_expr) {
  using namespace tir;
  const MulNode* mult_ptr = mult_expr.as<MulNode>();
  if (!mult_ptr) return std::make_pair(false, PrimExpr());
  PrimExpr mult_outer = mult_ptr->b;
  const PrimExpr* inner = &(mult_ptr->a);
  // 1. Calculate the outer multiplier
  while (true) {
    mult_ptr = inner->as<MulNode>();
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
  tir::ExprDeepEqual expr_equal;

  while (true) {
    auto inner_div_ptr = search_ptr->as<IndexDiv>();
    auto inner_mult_ptr = search_ptr->as<MulNode>();
    auto inner_add_ptr = search_ptr->as<AddNode>();
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
      LOG(FATAL) << "Unexpected search result!";
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
  using namespace tir;
  *has_mult = false;
  *has_mod = false;
  for (const PrimExpr* ele : eles) {
    auto mod_ptr = ele->as<IndexMod>();
    auto mult_ptr = ele->as<MulNode>();
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
inline PrimExpr MergeMulMod(arith::Analyzer* analyzer, const PrimExpr& base) {
  using namespace tir;
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

Array<PrimExpr> Buffer::OffsetOf(Array<PrimExpr> input_indices) const {
  return (*this)->ElemOffset(std::move(input_indices));
}

// The buffer offset in convention of number of elements of
// original data ignoring number of lanes.
// We also perform optimization to simplify the indexing expression.
Array<PrimExpr> BufferNode::ElemOffset(Array<PrimExpr> input_indices) const {
  ICHECK_EQ(shape.size(), input_indices.size())
      << "Buffer " << this->name << " is " << shape.size()
      << "-dimensional, cannot be indexed with the " << input_indices.size()
      << "-dimensional indices provided.";

  if (strides.size()) {
    ICHECK_EQ(this->strides.size(), input_indices.size())
        << "If strides are defined, "
        << "the index's dimensionality must match the dimensionality of the index given.";
  }

  // TODO(Lunderberg): Better handling for cases where there is more
  // than one output index.  Currently, this only allows elem_offset
  // to be non-zero for flat memory allocations.
  Array<PrimExpr> elem_offsets = {};
  if (elem_offset.defined() && !is_zero(elem_offset)) {
    elem_offsets = {elem_offset};
  }

  if (elem_offsets.size()) {
    ICHECK_EQ(elem_offsets.size(), axis_separators.size() + 1)
        << "If element offsets are defined, "
        << "there must be one element offset for each output index.";
  }

  Array<PrimExpr> output_indices(axis_separators.size() + 1, 0);

  size_t current_output_axis = 0;

  arith::Analyzer ana;

  for (size_t i = 0; i < input_indices.size(); i++) {
    if ((current_output_axis < axis_separators.size()) &&
        (i == size_t(axis_separators[current_output_axis]->value))) {
      current_output_axis++;
    }

    PrimExpr output_index = output_indices[current_output_axis];
    if (strides.size()) {
      output_index = output_index + input_indices[i] * strides[i];
    } else {
      output_index = output_index * this->shape[i] + input_indices[i];
    }

    if (i > 0) {
      output_index = MergeMulMod(&ana, output_index);
    }

    output_indices.Set(current_output_axis, output_index);
  }

  if (elem_offsets.size()) {
    for (size_t i = 0; i < output_indices.size(); i++) {
      output_indices.Set(i, output_indices[i] + elem_offsets[i]);
    }
  }

  return SimplifyArray(&ana, output_indices);
}

inline Array<PrimExpr> BufferOffset(const BufferNode* n, Array<PrimExpr> index, DataType dtype) {
  Array<PrimExpr> offsets = n->ElemOffset(index);
  // If the Buffer has element type with more than one lane, scale to
  // get the offset in number of scalars.
  if (n->dtype.lanes() != 1) {
    PrimExpr last_offset = offsets[offsets.size() - 1];
    offsets.Set(offsets.size() - 1, last_offset * make_const(last_offset.dtype(), dtype.lanes()));
  }

  // If the requested type has more than one lane, make a RampNode at
  // that offset.
  if (dtype.lanes() != 1) {
    PrimExpr last_offset = offsets[offsets.size() - 1];
    PrimExpr stride = make_const(last_offset.dtype(), 1);
    offsets.Set(offsets.size() - 1, tir::Ramp(last_offset, stride, dtype.lanes()));
  }

  return offsets;
}

Buffer Buffer::GetFlattenedBuffer() const {
  auto self = operator->();

  // These checks ensure that all output axes contain at least one
  // input axis.
  for (size_t i = 0; (i + 1) < self->axis_separators.size(); i++) {
    auto sep = self->axis_separators[i]->value;
    auto next_sep = self->axis_separators[i + 1]->value;
    ICHECK_LT(sep, next_sep) << "Axis separators must be in strictly increasing order.";
  }
  if (self->axis_separators.size()) {
    auto first_sep = self->axis_separators[0]->value;
    ICHECK_GT(first_sep, 0) << "First axis separator must be strictly greater than 0, "
                            << "so that first output axis contains at least one input axis";
    auto last_sep = self->axis_separators[self->axis_separators.size() - 1]->value;
    ICHECK_LT(last_sep, self->shape.size())
        << "Last output axis must contain at least one input axis.";
  }

  Array<PrimExpr> output_shape;
  if (self->strides.size()) {
    // If strides are defined, then the extent of each flattened
    // buffer is the stride*size for the first input axis used for
    // each output axis.
    ICHECK_EQ(self->shape.size(), self->strides.size());
    output_shape.push_back(self->strides[0] * self->shape[0]);
    for (const auto& sep : self->axis_separators) {
      output_shape.push_back(self->strides[sep->value] * self->shape[sep->value]);
    }

  } else {
    // Otherwise, the extent of each flattened buffer is the product
    // of the extents of each input axis used to generate that output
    // axis.  This also "flattens" rank-0 tensors to a rank-1 buffer
    // of shape [1].
    output_shape = Array<PrimExpr>(self->axis_separators.size() + 1, 1);
    size_t current_output_index = 0;
    for (size_t i = 0; i < self->shape.size(); i++) {
      if ((current_output_index < self->axis_separators.size()) &&
          (i == size_t(self->axis_separators[current_output_index]->value))) {
        current_output_index += 1;
      }
      output_shape.Set(current_output_index, output_shape[current_output_index] * self->shape[i]);
    }
  }

  // The axis_separators for the output buffer.
  Array<IntImm> output_axis_separators;
  for (size_t i = 0; i < self->axis_separators.size(); i++) {
    auto dtype = self->axis_separators[i]->dtype;
    output_axis_separators.push_back(IntImm(dtype, i + 1));
  }

  if (output_shape.size() == self->shape.size() && self->strides.empty()) {
    return *this;
  } else {
    Buffer output = *this;
    auto writer = output.CopyOnWrite();
    writer->shape = output_shape;
    writer->axis_separators = output_axis_separators;
    writer->strides = {};
    return output;
  }
}

PrimExpr Buffer::vload(Array<PrimExpr> begin, DataType value_dtype) const {
  // specially handle bool, stored as DataType::Int(8)
  const BufferNode* n = operator->();
  ICHECK(n != nullptr);
  ICHECK(value_dtype.element_of() == n->dtype.element_of() &&
         value_dtype.lanes() % n->dtype.lanes() == 0)
      << "Cannot load " << value_dtype << " from buffer of " << n->dtype;

  Array<PrimExpr> indices = begin;
  int factor = value_dtype.lanes() / n->dtype.lanes();
  if (factor > 1) {
    indices.Set(indices.size() - 1, Ramp(indices[indices.size() - 1], 1, factor));
  }
  return BufferLoad(*this, indices);
}

Stmt Buffer::vstore(Array<PrimExpr> begin, PrimExpr value) const {
  // specially handle bool, stored as DataType::Int(8)
  const BufferNode* n = operator->();
  ICHECK(n != nullptr);
  DataType value_dtype = value.dtype();
  ICHECK(value_dtype.element_of() == n->dtype.element_of() &&
         value_dtype.lanes() % n->dtype.lanes() == 0)
      << "Cannot store " << value_dtype << " to buffer of " << n->dtype;

  Array<PrimExpr> indices = begin;
  int factor = value_dtype.lanes() / n->dtype.lanes();
  if (factor > 1) {
    indices.Set(indices.size() - 1, Ramp(indices[indices.size() - 1], 1, factor));
  }
  return BufferStore(*this, value, indices);
}

String Buffer::scope() const {
  const auto* ptr_type = (*this)->data->type_annotation.as<PointerTypeNode>();
  ICHECK(ptr_type) << "Buffer variable is not of pointer type";
  if (ptr_type->storage_scope.empty()) {
    return "global";
  }
  return ptr_type->storage_scope;
}

Buffer Buffer::MakeStrideView() const {
  if ((*this)->strides.size() != 0) return *this;
  if ((*this)->shape.size() == 0) return *this;
  std::vector<PrimExpr> temp;
  const BufferNode* self = operator->();
  ICHECK(self != nullptr);
  auto n = make_object<BufferNode>(*self);
  PrimExpr acc = make_const(n->DefaultIndexType(), 1);
  for (size_t i = n->shape.size(); i != 0; --i) {
    temp.push_back(acc);
    acc = acc * n->shape[i - 1];
  }
  for (size_t i = temp.size(); i != 0; --i) {
    n->strides.push_back(temp[i - 1]);
  }
  return Buffer(n);
}

Buffer Buffer::MakeSlice(Array<PrimExpr> begins, Array<PrimExpr> extents) const {
  const BufferNode* n = operator->();
  ICHECK(n != nullptr);
  arith::Analyzer ana;
  begins = SimplifyArray(&ana, begins);
  Array<PrimExpr> elem_offset =
      n->ElemOffset(begins).Map([&](const PrimExpr& expr) { return ana.Simplify(expr); });

  Array<PrimExpr> strides = n->strides;
  if (strides.size() == 0) {
    bool can_relax = true;
    bool need_stride = false;
    // check if stride is needed.
    for (size_t i = 0; i < extents.size(); ++i) {
      if (!can_relax) {
        if (!is_zero(begins[i]) || !is_zero(ana.Simplify(extents[i] - n->shape[i]))) {
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
  Buffer slice(n->data, n->dtype, extents, strides, elem_offset[0], n->name + "_slice",
               n->data_alignment, 0, n->buffer_type);

  // Buffer must be constructed with a singular element offset which means there is no
  // support for n-dimensional buffers where n > 1.  Insert sentinel value for
  // ArgBinder::BindBuffer to state that any usage of element offset is invalid
  // in this case.  This allows for construction of a Buffer with multiple element offsets
  // but disallows any usage of those element offsets.  See PR #10816 for discussion on
  // supporting multiple element offsets in TIR Buffer.
  // TODO(Lunderberg): Remove if/when TIR supports multiple element offsets in TIR Buffer
  if (elem_offset.size() != 1) {
    slice.CopyOnWrite()->elem_offset = PrimExpr();
  }
  return slice;
}

PrimExpr Buffer::access_ptr(int access_mask, DataType ptr_type, int content_lanes, PrimExpr offset,
                            Optional<PrimExpr> input_extent) const {
  const BufferNode* self = operator->();
  ICHECK(self != nullptr);
  PrimExpr e_dtype;
  PrimExpr extent;
  if (self->shape.size() == 0) {
    extent = make_const(self->DefaultIndexType(), 1);
  } else if (self->strides.size() == self->shape.size()) {
    int highest_dim = 0;
    extent = self->strides[highest_dim] * self->shape[highest_dim] - offset;
  } else {
    extent = foldl([](PrimExpr a, PrimExpr b, Span span) { return mul(a, b, span); },
                   make_const(DataType::Int(32), 1), self->shape) -
             offset;
  }
  PrimExpr elem_offset = self->elem_offset + offset;
  if (content_lanes > 1) {
    e_dtype = tir::TypeAnnotation(self->dtype.with_lanes(content_lanes));
    extent = extent / make_const(self->elem_offset.dtype(), content_lanes);
    elem_offset = self->elem_offset / make_const(self->elem_offset.dtype(), content_lanes);
  } else {
    e_dtype = tir::TypeAnnotation(self->dtype);
  }

  if (input_extent.defined()) {
    extent = input_extent.value();
  }
  Array<PrimExpr> acc_args{e_dtype, self->data, elem_offset, extent,
                           make_const(DataType::Int(32), access_mask)};
  return tir::Call(ptr_type, tir::builtin::tvm_access_ptr(), acc_args);
}

Buffer::Buffer(Var data, DataType dtype, Array<PrimExpr> shape, Array<PrimExpr> strides,
               PrimExpr elem_offset, String name, int data_alignment, int offset_factor,
               BufferType buffer_type, Array<IntImm> axis_separators, Span span) {
  DataType storage_dtype = dtype;
  // specially handle bool
  if (storage_dtype == DataType::Bool()) {
    storage_dtype = DataType::Int(8);
  }
  // The buffer dtype may differ from the dtype of the underlying
  // allocation, such as a single allocation that backs multiple
  // tensors without a common datatype.  Therefore, we check that the
  // data pointer is a pointer, but not the exact type of the
  // pointed-to values.

  // TODO(Lunderberg): Use an explicit pointer cast for the data
  // pointer.  Should be done alongside extensions to StmtExprMutator
  // to more easily handle buffer/buffer_var updates.
  ICHECK(data->type_annotation.defined())
      << "Variable " << data->name_hint << " is missing a type annotation.";
  ICHECK(data->type_annotation.as<PointerTypeNode>())
      << "Variable " << data->name_hint << " is not a pointer.";
  ICHECK(data->type_annotation.as<PointerTypeNode>()->element_type.as<PrimTypeNode>())
      << "Variable " << data->name_hint << " does not point to a primitive.";

  auto n = make_object<BufferNode>();
  n->data = std::move(data);
  n->dtype = dtype;

  n->shape = std::move(shape);
  n->strides = std::move(strides);
  n->axis_separators = std::move(axis_separators);
  n->name = std::move(name);
  if (!elem_offset.defined()) {
    elem_offset = make_const(n->DefaultIndexType(), 0);
  }
  if (data_alignment <= 0) {
    data_alignment = runtime::kAllocAlignment;
  }
  if (offset_factor == 0) {
    offset_factor = 1;
  }
  n->elem_offset = std::move(elem_offset);
  n->data_alignment = data_alignment;
  n->offset_factor = offset_factor;
  n->buffer_type = buffer_type;
  if (n->buffer_type == kAutoBroadcast && n->shape.size() > 0 && n->strides.empty()) {
    for (size_t i = 0; i < n->shape.size(); ++i) {
      n->strides.push_back(Var("stride", n->shape[i].dtype()));
    }
  }
  n->span = std::move(span);
  data_ = std::move(n);
}

tir::Buffer BufferWithOffsetAlignment(Array<PrimExpr> shape, DataType dtype, std::string name,
                                      int data_alignment, int offset_factor, bool compact,
                                      std::string memory_scope) {
  DataType storage_dtype = (dtype == DataType::Bool() ? DataType::Int(8) : dtype);
  auto data = tir::Var(name, PointerType(PrimType(storage_dtype), memory_scope));
  bool has_any = false;
  if (!compact) {
    for (const auto& it : shape) {
      if (it.as<tir::VarNode>()) {
        has_any = true;
        break;
      }
    }
  }
  tir::BufferType buffer_type = has_any ? tir::kAutoBroadcast : tir::kDefault;

  PrimExpr elem_offset;
  if (offset_factor != 0) {
    elem_offset = tir::Var(name + "_elem_offset", shape[0].dtype());
  } else {
    elem_offset = PrimExpr();
  }

  return tir::Buffer(data, dtype, shape, Array<PrimExpr>(), elem_offset, name, data_alignment,
                     offset_factor, buffer_type);
}

TVM_REGISTER_NODE_TYPE(BufferNode);

TVM_REGISTER_GLOBAL("tir.Buffer").set_body([](TVMArgs args, TVMRetValue* ret) {
  ICHECK_EQ(args.size(), 11);
  auto buffer_type = args[8].operator String();
  BufferType type = (buffer_type == "auto_broadcast") ? kAutoBroadcast : kDefault;
  *ret = Buffer(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], type,
                args[9], args[10]);
});

TVM_REGISTER_GLOBAL("tir.BufferAccessPtr").set_body_method(&Buffer::access_ptr);

TVM_REGISTER_GLOBAL("tir.BufferGetFlattenedBuffer").set_body_method(&Buffer::GetFlattenedBuffer);

TVM_REGISTER_GLOBAL("tir.BufferOffsetOf").set_body_method(&Buffer::OffsetOf);

TVM_REGISTER_GLOBAL("tir.BufferVLoad").set_body_method(&Buffer::vload);

TVM_REGISTER_GLOBAL("tir.BufferVStore").set_body_method(&Buffer::vstore);

TVM_REGISTER_GLOBAL("tir.BufferStorageScope").set_body_method(&Buffer::scope);

}  // namespace tir
}  // namespace tvm
