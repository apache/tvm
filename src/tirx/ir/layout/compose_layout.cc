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
#include <algorithm>
#include <limits>
#include <optional>
#include <vector>

#include "tile_internal.h"

namespace tvm {
namespace tirx {

namespace {

PrimExpr ApplyFullSwizzle(const ComposeLayoutNode* layout, const PrimExpr& m) {
  auto swizzle = [&](const PrimExpr& x) -> PrimExpr {
    if (layout->swizzle_inner) {
      return x ^ ((x & layout->outer_mask) >> layout->atom_len);
    }
    return x ^ ((x & layout->inner_mask) << layout->atom_len);
  };
  int base = 1 << layout->per_element;
  arith::Analyzer analyzer;
  PrimVar m_once("compose_m", m.ty());
  PrimExpr quotient = floordiv(m_once, base);
  PrimVar quotient_once("compose_q", quotient.ty());
  PrimExpr body =
      analyzer->Simplify((swizzle(quotient_once) << layout->per_element) + floormod(m_once, base));
  return Let(m_once, m, Let(quotient_once, quotient, body));
}

void AddExpr(std::optional<PrimExpr>* sum, const PrimExpr& term, const arith::Analyzer& analyzer) {
  if (is_zero(term)) return;
  if (sum->has_value()) {
    *sum = analyzer->Simplify(sum->value() + term);
  } else {
    *sum = term;
  }
}

bool AddWithoutOverflow(int64_t lhs, int64_t rhs, int64_t* result) {
  if (rhs > 0 && lhs > std::numeric_limits<int64_t>::max() - rhs) return false;
  if (rhs < 0 && lhs < std::numeric_limits<int64_t>::min() - rhs) return false;
  *result = lhs + rhs;
  return true;
}

bool MulWithoutOverflow(int64_t lhs, int64_t rhs, int64_t* result) {
  if (lhs == 0 || rhs == 0) {
    *result = 0;
    return true;
  }
  if (lhs < 0 || rhs < 0) return false;
  if (lhs > std::numeric_limits<int64_t>::max() / rhs) return false;
  *result = lhs * rhs;
  return true;
}

void CollectOffsetTerms(const PrimExpr& expr, int sign, std::vector<PrimExpr>* dynamic_terms,
                        int64_t* constant, bool* valid, const arith::Analyzer& analyzer) {
  if (!*valid) return;
  PrimExpr simplified = analyzer->Simplify(expr);
  if (const auto* imm = simplified.as<IntImmNode>()) {
    int64_t signed_value;
    if (!MulWithoutOverflow(imm->value, sign, &signed_value) ||
        !AddWithoutOverflow(*constant, signed_value, constant)) {
      *valid = false;
    }
    return;
  }
  if (const auto* add = simplified.as<AddNode>()) {
    CollectOffsetTerms(add->a, sign, dynamic_terms, constant, valid, analyzer);
    CollectOffsetTerms(add->b, sign, dynamic_terms, constant, valid, analyzer);
    return;
  }
  if (const auto* sub = simplified.as<SubNode>()) {
    CollectOffsetTerms(sub->a, sign, dynamic_terms, constant, valid, analyzer);
    CollectOffsetTerms(sub->b, -sign, dynamic_terms, constant, valid, analyzer);
    return;
  }
  if (sign < 0) {
    *valid = false;
    return;
  }
  dynamic_terms->push_back(simplified);
}

std::optional<PrimExpr> DivideExactTerm(const PrimExpr& term, int64_t divisor,
                                        const arith::Analyzer& analyzer) {
  PrimExpr simplified = analyzer->Simplify(term);
  if (const auto* imm = simplified.as<IntImmNode>()) {
    if (imm->value % divisor != 0) return std::nullopt;
    return IntImm(simplified.ty(), imm->value / divisor);
  }

  if (const auto* mul = simplified.as<MulNode>()) {
    const IntImmNode* factor = mul->a.as<IntImmNode>();
    PrimExpr value = mul->b;
    if (factor == nullptr) {
      factor = mul->b.as<IntImmNode>();
      value = mul->a;
    }
    if (factor != nullptr && factor->value >= 0 && factor->value % divisor == 0) {
      return analyzer->Simplify(value * IntImm(value.ty(), factor->value / divisor));
    }
  }

  PrimExpr scale = IntImm(simplified.ty(), divisor);
  if (!analyzer->CanProveEqual(floormod(simplified, scale), IntImm(simplified.ty(), 0))) {
    return std::nullopt;
  }
  return analyzer->Simplify(floordiv(simplified, scale));
}

ffi::Map<ffi::String, PrimExpr> ApplyStructured(const ComposeLayoutNode* layout,
                                                const TileLayout& tile,
                                                ffi::Array<PrimExpr> coord) {
  // Normalize the bounded TileLayout terms as m = B + D + K, where B is a
  // multiple of one atom, D is dynamic and contained in that atom, and K is
  // the constant remainder.  If D + K is provably carry-free, the swizzle is
  //
  //   phase = (B / atom) & mask
  //   P = B + (D ^ (phase << per_element))
  //   address = P ^ K.
  //
  // Every division used for B / atom is derived from its individual proven
  // term.  The full address sum is never divided speculatively.
  TVM_FFI_ICHECK_EQ(coord.size(), tile->shard.size())
      << "Coordinate size must match the number of shard axes";

  arith::Analyzer analyzer;
  for (size_t i = 0; i < tile->shard.size(); ++i) {
    if (analyzer->CanProveEqual(tile->shard[i]->extent, 1)) {
      coord.Set(i, IntImm(coord[i].ty(), 0));
    }
  }

  auto mapped = tile->Apply(coord);
  TVM_FFI_ICHECK(mapped.size() == 1 && mapped.find("m") != mapped.end());
  PrimExpr m = mapped["m"];
  auto fallback = [&]() {
    return ffi::Map<ffi::String, PrimExpr>{{"m", ApplyFullSwizzle(layout, m)}};
  };

  if (layout->swizzle_len == 0) return mapped;
  if (!layout->swizzle_inner) return fallback();
  int scale_bits = layout->per_element + layout->atom_len;
  if (scale_bits < 0 || scale_bits >= 62) return fallback();
  int64_t atom = int64_t{1} << scale_bits;

  std::optional<PrimExpr> high;
  std::optional<PrimExpr> high_in_atoms;
  std::optional<PrimExpr> low;
  int64_t low_max = 0;
  int64_t constant = 0;

  auto add_constant = [&](int64_t value) -> bool {
    if (value < 0) return false;
    return AddWithoutOverflow(constant, value, &constant);
  };
  auto add_high = [&](const PrimExpr& term, const PrimExpr& quotient) {
    AddExpr(&high, term, analyzer);
    AddExpr(&high_in_atoms, quotient, analyzer);
  };
  auto add_low = [&](const PrimExpr& term, int64_t term_max) -> bool {
    if (term_max < 0 || term_max >= atom) return false;
    int64_t next_max;
    if (!AddWithoutOverflow(low_max, term_max, &next_max) || next_max >= atom) return false;
    low_max = next_max;
    AddExpr(&low, term, analyzer);
    return true;
  };

  for (size_t i = 0; i < tile->shard.size(); ++i) {
    const Iter& iter = tile->shard[i];
    if (analyzer->CanProveEqual(iter->extent, 1)) continue;
    PrimExpr simplified_stride = analyzer->Simplify(iter->stride);
    const int64_t* stride = as_const_int(simplified_stride);
    if (stride == nullptr || *stride < 0) return fallback();
    PrimExpr term = analyzer->Simplify(coord[i] * iter->stride);
    if (const auto* imm = term.as<IntImmNode>()) {
      if (!add_constant(imm->value)) return fallback();
      continue;
    }
    if (auto quotient = DivideExactTerm(term, atom, analyzer); quotient.has_value()) {
      add_high(term, quotient.value());
      continue;
    }
    PrimExpr simplified_extent = analyzer->Simplify(iter->extent);
    const int64_t* extent = as_const_int(simplified_extent);
    if (extent == nullptr || *extent <= 0) return fallback();
    int64_t term_max;
    if (!MulWithoutOverflow(*extent - 1, *stride, &term_max)) return fallback();

    // A single structured coordinate may span multiple atoms even when its
    // stride is smaller than one atom.  If the stride divides the atom, split
    // that term exactly instead of dividing the complete address:
    //
    //   coord * stride = (coord / n) * atom + (coord % n) * stride,
    //   n = atom / stride.
    //
    // The first term is atom-aligned and the second is independently bounded
    // below one atom, so it can continue through the existing carry proof.
    if (term_max >= atom && *stride > 0 && *stride < atom && atom % *stride == 0) {
      int64_t n = atom / *stride;
      PrimExpr n_expr = IntImm(coord[i].ty(), n);
      PrimExpr high_coord = analyzer->Simplify(floordiv(coord[i], n_expr));
      PrimExpr low_coord = analyzer->Simplify(floormod(coord[i], n_expr));
      PrimExpr high_term = analyzer->Simplify(high_coord * IntImm(high_coord.ty(), atom));
      PrimExpr low_term = analyzer->Simplify(low_coord * iter->stride);
      add_high(high_term, high_coord);

      int64_t low_extent = std::min(*extent, n);
      int64_t split_low_max;
      if (!MulWithoutOverflow(low_extent - 1, *stride, &split_low_max) ||
          !add_low(low_term, split_low_max)) {
        return fallback();
      }
      continue;
    }

    if (!add_low(term, term_max)) {
      return fallback();
    }
  }

  for (const auto& [axis, offset] : tile->offset) {
    if (axis->name != "m") return fallback();
    std::vector<PrimExpr> dynamic_terms;
    bool valid = true;
    CollectOffsetTerms(offset, 1, &dynamic_terms, &constant, &valid, analyzer);
    if (!valid) return fallback();
    for (const PrimExpr& term : dynamic_terms) {
      if (auto quotient = DivideExactTerm(term, atom, analyzer); quotient.has_value()) {
        add_high(term, quotient.value());
        continue;
      }
      arith::ConstIntBound bound = analyzer->const_int_bound(term);
      if (bound->min_value < 0 || bound->max_value == arith::ConstIntBound::kPosInf ||
          !add_low(term, bound->max_value)) {
        return fallback();
      }
    }
  }

  int64_t constant_low = constant % atom;
  int64_t constant_high = constant - constant_low;
  if (constant_high != 0) {
    PrimExpr high_term = IntImm(m.ty(), constant_high);
    AddExpr(&high, high_term, analyzer);
    int64_t constant_phase = (constant_high / atom) & layout->inner_mask;
    if (constant_phase != 0) {
      AddExpr(&high_in_atoms, IntImm(m.ty(), constant_phase), analyzer);
    }
  }

  int64_t low_and_constant_max;
  if (!AddWithoutOverflow(low_max, constant_low, &low_and_constant_max) ||
      low_and_constant_max >= atom) {
    return fallback();
  }
  if (constant_low != 0 && low_max != 0) {
    int64_t lowest_constant_bit = constant_low & -constant_low;
    if (low_max >= lowest_constant_bit) return fallback();
  }

  PrimExpr zero = IntImm(m.ty(), 0);
  PrimExpr high_expr = high.value_or(zero);
  PrimExpr low_expr = low.value_or(zero);
  PrimExpr high_atoms_expr = high_in_atoms.value_or(zero);
  PrimExpr phase = high_atoms_expr & layout->inner_mask;
  PrimExpr permuted_low = low_expr ^ (phase << layout->per_element);
  PrimExpr base = analyzer->Simplify(high_expr + permuted_low);
  PrimExpr address =
      constant_low == 0 ? base : analyzer->Simplify(base ^ IntImm(base.ty(), constant_low));
  return {{"m", address}};
}

}  // namespace

/**************** ComposeLayout ****************/
ComposeLayout::ComposeLayout(int per_element, int swizzle_len, int atom_len, TileLayout tile_layout,
                             bool swizzle_inner) {
  auto n = ffi::make_object<ComposeLayoutNode>();
  n->per_element = per_element;
  n->swizzle_len = swizzle_len;
  n->atom_len = atom_len;
  n->swizzle_inner = swizzle_inner;
  n->tile_layout = std::move(tile_layout);
  TVM_FFI_ICHECK(n->VerifyWellFormed()) << "ValueError: The compose layout is not well-formed";
  int swizzle_mask = (1 << swizzle_len) - 1;
  n->inner_mask = swizzle_mask;
  n->outer_mask = swizzle_mask << atom_len;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.ComposeLayout", [](int per_element, int swizzle_len, int atom_len,
                                                 TileLayout tile_layout, bool swizzle_inner) {
    return ComposeLayout(per_element, swizzle_len, atom_len, tile_layout, swizzle_inner);
  });
}

bool ComposeLayoutNode::CompatibleWithShape(const Array<PrimExpr>& shape) const { return true; }

bool ComposeLayoutNode::VerifyWellFormed() const {
  if (!(per_element >= 0 && swizzle_len >= 0 && atom_len >= swizzle_len)) {
    return false;
  }
  return tile_layout->VerifyWellFormed();
}

PrimExpr ComposeLayoutNode::GetSize(ffi::Optional<ffi::String> axis_name) const {
  TVM_FFI_ICHECK(!axis_name.has_value())
      << "ValueError: axis_name is not supported for compose layout";
  return tile_layout->GetSize(axis_name);
}

PrimExpr ComposeLayoutNode::GetSpan(ffi::Optional<ffi::String> axis_name) const {
  TVM_FFI_ICHECK(!axis_name.has_value())
      << "ValueError: axis_name is not supported for compose layout";
  return tile_layout->GetSpan(axis_name);
}

ffi::Map<ffi::String, PrimExpr> ComposeLayoutNode::Apply(ffi::Array<PrimExpr> coord) const {
  return ApplyStructured(this, tile_layout, std::move(coord));
}

ffi::Map<ffi::String, PrimExpr> ComposeLayoutNode::Apply(PrimExpr coord) const {
  auto res = tile_layout->Apply(coord);
  TVM_FFI_ICHECK(res.size() == 1 && res.find("m") != res.end());
  auto m = res["m"];
  // Inline the swizzle XOR (formerly SwizzleLayoutNode::Apply): the swizzle
  // operates on the tile-mapped coordinate ``m``.
  auto f = [&](const PrimExpr& x) -> PrimExpr {
    if (swizzle_inner) {
      return x ^ ((x & outer_mask) >> atom_len);
    } else {
      return x ^ ((x & inner_mask) << atom_len);
    }
  };
  auto base = 1 << per_element;
  arith::Analyzer analyzer;
  return {{"m", analyzer->Simplify((f(floordiv(m, base)) << per_element) + floormod(m, base))}};
}

ffi::Map<ffi::String, PrimExpr> ComposeLayoutNode::Apply(const ffi::Array<PrimExpr>& coord,
                                                         const ffi::Array<PrimExpr>& shape) const {
  TVM_FFI_ICHECK_EQ(coord.size(), shape.size())
      << "ValueError: The size of coord and shape should be equal";
  auto grouped = TryGroup(tile_layout, shape);
  if (!grouped.has_value()) {
    auto mapped = tile_layout->Apply(coord, shape);
    TVM_FFI_ICHECK(mapped.size() == 1 && mapped.find("m") != mapped.end());
    return {{"m", ApplyFullSwizzle(this, mapped["m"])}};
  }

  const auto& [grouped_tile, separators] = grouped.value();
  ffi::Array<PrimExpr> shard_coord;
  shard_coord.reserve(grouped_tile->shard.size());
  for (size_t i = 0; i < grouped_tile->shard.size(); ++i) {
    shard_coord.push_back(IntImm::Int32(0));
  }
  for (size_t d = 0; d < shape.size(); ++d) {
    int64_t start = separators[d];
    int64_t end = separators[d + 1];
    if (start == end) continue;
    ffi::Array<PrimExpr> extents;
    for (int64_t i = start; i < end; ++i) {
      extents.push_back(grouped_tile->shard[i]->extent);
    }
    ffi::Array<PrimExpr> split = SplitCoord(coord[d], extents);
    for (int64_t i = start, j = 0; i < end; ++i, ++j) {
      shard_coord.Set(i, split[j]);
    }
  }
  return ApplyStructured(this, grouped_tile, std::move(shard_coord));
}

Layout ComposeLayoutNode::Canonicalize() const {
  auto tile_normalized = tile_layout->Canonicalize().as<TileLayout>().value();
  return ComposeLayout(per_element, swizzle_len, atom_len, tile_normalized, swizzle_inner);
}

Layout ComposeLayoutNode::Tile(const TileLayout& outer, const ffi::Array<PrimExpr>& outer_shape,
                               const ffi::Array<PrimExpr>& inner_shape) const {
  // A bare swizzle (ComposeLayout with a trivial tile) carries only the swizzle
  // period, not a tile matching `inner_shape`; substitute an identity tile over
  // the inner product first, matching the former SwizzleLayoutNode::Tile.
  TileLayout base = this->tile_layout;
  if (base->IsTrivial()) {
    base = IdentityTileLayout(inner_shape);
  }
  auto tiled_B = base->Tile(outer, outer_shape, inner_shape).as<TileLayout>().value();
  return ComposeLayout(per_element, swizzle_len, atom_len, tiled_B, swizzle_inner);
}

ffi::Optional<TileLayout> ComposeLayoutNode::IsTileInner(
    const Layout& tile_layout, const ffi::Array<PrimExpr>& tiled_shape,
    const ffi::Array<PrimExpr>& inner_shape) const {
  if (auto comp = tile_layout.as<ComposeLayout>()) {
    if (comp.value()->per_element == this->per_element &&
        comp.value()->swizzle_len == this->swizzle_len &&
        comp.value()->atom_len == this->atom_len &&
        comp.value()->swizzle_inner == this->swizzle_inner) {
      // A bare swizzle (ComposeLayout with a trivial tile) has no real tile to
      // compare; its "tile" is the identity over the inner product, and a bare
      // `tile_layout` argument contributes the identity over the tiled product.
      // Substitute those identities so TileLayoutNode::IsTileInner sees the same
      // inputs the former SwizzleLayoutNode::IsTileInner produced.
      TileLayout this_tile = this->tile_layout;
      if (this->tile_layout->IsTrivial()) {
        this_tile = IdentityTileLayout(inner_shape);
      }
      TileLayout arg_tile = comp.value()->tile_layout;
      if (comp.value()->tile_layout->IsTrivial()) {
        arg_tile = IdentityTileLayout(tiled_shape);
      }
      return this_tile->IsTileInner(arg_tile, tiled_shape, inner_shape);
    }
  }
  return std::nullopt;
}

ffi::Optional<Layout> ComposeLayoutNode::IsTileOuter(
    const Layout& tile_layout, const ffi::Array<PrimExpr>& tiled_shape,
    const ffi::Array<PrimExpr>& outer_shape) const {
  return std::nullopt;
}

ffi::Optional<Layout> ComposeLayoutNode::Slice(const ffi::Array<PrimExpr>& shape,
                                               const Region& region) const {
  // A bare swizzle (ComposeLayout with a trivial tile) carries only the swizzle
  // period, not a tile matching `shape`; substitute an identity tile over the
  // buffer shape first, matching the former SwizzleLayoutNode::Slice.
  TileLayout base = this->tile_layout;
  if (base->IsTrivial()) {
    base = IdentityTileLayout(shape);
  }
  auto sliced_opt = base->Slice(shape, region);
  if (!sliced_opt.has_value()) return std::nullopt;
  auto sliced = sliced_opt.value().as<TileLayout>().value();
  return ComposeLayout(per_element, swizzle_len, atom_len, sliced, swizzle_inner);
}

}  // namespace tirx
}  // namespace tvm
