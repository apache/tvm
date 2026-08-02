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

/*
 * Tiling operations and helpers for TileLayout.
 */
#include <algorithm>

#include "tile_internal.h"

namespace tvm {
namespace tirx {

std::pair<TileLayout, std::vector<int64_t>> Group(TileLayout layout,
                                                  const ffi::Array<PrimExpr>& shape) {
  arith::Analyzer analyzer;
  size_t shape_idx = 0;
  PrimExpr prod = 1;

  std::vector<Iter> new_shard;
  std::vector<int64_t> seps{0};

  for (size_t i = 0; i < layout->shard.size(); ++i) {
    auto extent_i = layout->shard[i]->extent;
    auto stride_i = layout->shard[i]->stride;
    prod *= extent_i;
    while (shape_idx < shape.size() &&
           analyzer->CanProveEqual(floormod(prod, shape[shape_idx]), 0)) {
      // Simplify ``c``, ``floordiv(extent_i, c)`` and ``stride_i * c`` —
      // without this, splitting an iter whose extent contains a symbolic
      // dim that algebraically cancels (e.g. ``floordiv(batch_size,
      // batch_size) == 1``) leaves dead ``a // a`` factors in the new
      // iter's stride that ``int(stride)`` can't unwrap downstream.
      PrimExpr c = analyzer->Simplify(floordiv(prod, shape[shape_idx]));
      TVM_FFI_ICHECK(analyzer->CanProveEqual(floormod(extent_i, c), 0))
          << "layout " << layout << " can not be grouped by shape " << shape;
      new_shard.push_back(Iter(analyzer->Simplify(floordiv(extent_i, c)),
                               analyzer->Simplify(stride_i * c), layout->shard[i]->axis));
      extent_i = c;
      prod = c;
      shape_idx++;
      seps.push_back(new_shard.size());
    }
    extent_i = analyzer->Simplify(extent_i);
    if (!is_one(extent_i)) {
      TVM_FFI_ICHECK(shape_idx < shape.size())
          << "layout " << layout << " can not be grouped by shape " << shape;
      new_shard.push_back(Iter(extent_i, analyzer->Simplify(stride_i), layout->shard[i]->axis));
    }
  }

  TVM_FFI_ICHECK(shape_idx == shape.size())
      << "layout " << layout << " can not be grouped by shape " << shape;

  auto* n = layout.CopyOnWrite();
  n->shard = new_shard;
  return {ffi::GetRef<TileLayout>(n), seps};
}

std::pair<TileLayout, std::vector<std::vector<int64_t>>> GroupMany(
    TileLayout layout, const ffi::Array<ffi::Array<PrimExpr>>& shapes) {
  struct BoundaryClass {
    PrimExpr value;
    std::vector<size_t> counts;
  };

  arith::Analyzer analyzer;
  TVM_FFI_ICHECK(!shapes.empty()) << "group_many requires at least one shape";

  std::vector<std::vector<PrimExpr>> boundary_sequences;
  boundary_sequences.reserve(shapes.size() + 1);
  for (size_t shape_idx = 0; shape_idx < shapes.size(); ++shape_idx) {
    const auto& shape = shapes[shape_idx];
    TVM_FFI_ICHECK(!shape.empty()) << "group_many shape " << shape_idx << " must not be empty";

    std::vector<PrimExpr> boundaries{PrimExpr(1)};
    PrimExpr product = 1;
    for (size_t dim_idx = 0; dim_idx < shape.size(); ++dim_idx) {
      PrimExpr previous = product;
      product = analyzer->Simplify(product * shape[dim_idx]);
      TVM_FFI_ICHECK(analyzer->CanProve(previous <= product))
          << "group_many cannot prove cumulative boundary order for shape " << shape_idx
          << " at dimension " << dim_idx << ": " << previous << " <= " << product;
      TVM_FFI_ICHECK(analyzer->CanProveEqual(floormod(product, previous), 0))
          << "group_many cannot prove cumulative boundary divisibility for shape " << shape_idx
          << " at dimension " << dim_idx << ": " << previous << " divides " << product;
      boundaries.push_back(product);
    }
    boundary_sequences.push_back(std::move(boundaries));
  }

  std::vector<PrimExpr> layout_boundaries{PrimExpr(1)};
  PrimExpr layout_product = 1;
  for (size_t iter_idx = 0; iter_idx < layout->shard.size(); ++iter_idx) {
    PrimExpr previous = layout_product;
    layout_product = analyzer->Simplify(layout_product * layout->shard[iter_idx]->extent);
    TVM_FFI_ICHECK(analyzer->CanProve(previous <= layout_product))
        << "group_many cannot prove cumulative boundary order for layout iter " << iter_idx << ": "
        << previous << " <= " << layout_product;
    TVM_FFI_ICHECK(analyzer->CanProveEqual(floormod(layout_product, previous), 0))
        << "group_many cannot prove cumulative boundary divisibility for layout iter " << iter_idx
        << ": " << previous << " divides " << layout_product;
    layout_boundaries.push_back(layout_product);
  }

  const PrimExpr& total = boundary_sequences[0].back();
  for (size_t shape_idx = 1; shape_idx < boundary_sequences.size(); ++shape_idx) {
    TVM_FFI_ICHECK(analyzer->CanProveEqual(total, boundary_sequences[shape_idx].back()))
        << "group_many cannot prove equal total products for shape 0 and shape " << shape_idx
        << ": " << total << " vs " << boundary_sequences[shape_idx].back();
  }
  TVM_FFI_ICHECK(analyzer->CanProveEqual(total, layout_boundaries.back()))
      << "group_many cannot prove that the layout and shapes have equal total products: "
      << layout_boundaries.back() << " vs " << total;
  boundary_sequences.push_back(std::move(layout_boundaries));

  std::vector<BoundaryClass> classes;
  for (size_t sequence_idx = 0; sequence_idx < boundary_sequences.size(); ++sequence_idx) {
    for (const PrimExpr& boundary : boundary_sequences[sequence_idx]) {
      size_t class_idx = classes.size();
      for (size_t idx = 0; idx < classes.size(); ++idx) {
        if (analyzer->CanProveEqual(boundary, classes[idx].value)) {
          class_idx = idx;
          break;
        }
      }

      if (class_idx == classes.size()) {
        size_t insert_at = classes.size();
        for (size_t idx = 0; idx < classes.size(); ++idx) {
          if (analyzer->CanProve(boundary < classes[idx].value)) {
            insert_at = idx;
            break;
          }
          TVM_FFI_ICHECK(analyzer->CanProve(classes[idx].value < boundary))
              << "group_many cannot prove the order or equality of cumulative boundaries "
              << boundary << " and " << classes[idx].value;
        }
        classes.insert(classes.begin() + insert_at,
                       BoundaryClass{boundary, std::vector<size_t>(boundary_sequences.size(), 0)});
        class_idx = insert_at;
      }
      classes[class_idx].counts[sequence_idx]++;
    }
  }

  std::vector<size_t> class_starts;
  std::vector<size_t> class_multiplicities;
  std::vector<PrimExpr> common_boundaries;
  class_starts.reserve(classes.size());
  class_multiplicities.reserve(classes.size());
  for (const BoundaryClass& boundary_class : classes) {
    size_t multiplicity =
        *std::max_element(boundary_class.counts.begin(), boundary_class.counts.end());
    class_starts.push_back(common_boundaries.size());
    class_multiplicities.push_back(multiplicity);
    for (size_t idx = 0; idx < multiplicity; ++idx) {
      common_boundaries.push_back(boundary_class.value);
    }
  }

  std::vector<PrimExpr> refined_extents;
  refined_extents.reserve(common_boundaries.size() - 1);
  for (size_t idx = 1; idx < common_boundaries.size(); ++idx) {
    const PrimExpr& previous = common_boundaries[idx - 1];
    const PrimExpr& current = common_boundaries[idx];
    TVM_FFI_ICHECK(analyzer->CanProve(previous <= current))
        << "group_many cannot prove common boundary order: " << previous << " <= " << current;
    TVM_FFI_ICHECK(analyzer->CanProveEqual(floormod(current, previous), 0))
        << "group_many has incompatible cumulative boundaries: " << previous << " does not divide "
        << current;
    refined_extents.push_back(analyzer->Simplify(floordiv(current, previous)));
  }

  auto boundary_positions = [&](const std::vector<PrimExpr>& sequence,
                                size_t sequence_idx) -> std::vector<size_t> {
    std::vector<size_t> seen(classes.size(), 0);
    std::vector<size_t> positions;
    positions.reserve(sequence.size());
    size_t previous_position = 0;
    for (size_t boundary_idx = 0; boundary_idx < sequence.size(); ++boundary_idx) {
      size_t class_idx = classes.size();
      for (size_t idx = 0; idx < classes.size(); ++idx) {
        if (analyzer->CanProveEqual(sequence[boundary_idx], classes[idx].value)) {
          class_idx = idx;
          break;
        }
      }
      TVM_FFI_ICHECK(class_idx < classes.size())
          << "group_many lost cumulative boundary " << sequence[boundary_idx];

      size_t position = class_starts[class_idx] + seen[class_idx]++;
      if (boundary_idx == 0) {
        position = class_starts[class_idx];
      }
      if (boundary_idx + 1 == sequence.size()) {
        position = class_starts[class_idx] + class_multiplicities[class_idx] - 1;
      }
      TVM_FFI_ICHECK(boundary_idx == 0 || position > previous_position)
          << "group_many could not preserve repeated boundary " << sequence[boundary_idx]
          << " for sequence " << sequence_idx;
      previous_position = position;
      positions.push_back(position);
    }
    TVM_FFI_ICHECK_EQ(positions.front(), 0);
    TVM_FFI_ICHECK_EQ(positions.back(), common_boundaries.size() - 1);
    return positions;
  };

  const auto& layout_sequence = boundary_sequences.back();
  std::vector<size_t> layout_positions =
      boundary_positions(layout_sequence, boundary_sequences.size() - 1);
  std::vector<Iter> refined_shard;
  refined_shard.reserve(refined_extents.size());
  for (size_t iter_idx = 0; iter_idx < layout->shard.size(); ++iter_idx) {
    size_t start = layout_positions[iter_idx];
    size_t end = layout_positions[iter_idx + 1];
    TVM_FFI_ICHECK_LT(start, end) << "group_many cannot preserve layout iter " << iter_idx;
    TVM_FFI_ICHECK(analyzer->CanProveEqual(
        common_boundaries[end], common_boundaries[start] * layout->shard[iter_idx]->extent))
        << "group_many refinement changed layout iter " << iter_idx << " extent "
        << layout->shard[iter_idx]->extent;
    for (size_t position = start; position < end; ++position) {
      PrimExpr inner_product =
          analyzer->Simplify(floordiv(common_boundaries[end], common_boundaries[position + 1]));
      refined_shard.push_back(
          Iter(refined_extents[position],
               analyzer->Simplify(layout->shard[iter_idx]->stride * inner_product),
               layout->shard[iter_idx]->axis));
    }
  }

  auto* n = layout.CopyOnWrite();
  n->shard = refined_shard;
  TileLayout grouped = ffi::GetRef<TileLayout>(n);

  std::vector<std::vector<int64_t>> separators;
  separators.reserve(shapes.size());
  for (size_t shape_idx = 0; shape_idx < shapes.size(); ++shape_idx) {
    std::vector<size_t> positions = boundary_positions(boundary_sequences[shape_idx], shape_idx);
    separators.emplace_back(positions.begin(), positions.end());
  }
  return {grouped, separators};
}

std::optional<std::pair<TileLayout, std::vector<int64_t>>> TryGroup(
    TileLayout layout, const ffi::Array<PrimExpr>& shape) {
  // Same algorithm as Group but returns std::nullopt instead of ICHECK-failing
  // on regroup impossibility. Used by Apply(coord, shape) to opportunistically
  // pick the group-first path with a fallback to flatten+split.
  arith::Analyzer analyzer;
  size_t shape_idx = 0;
  PrimExpr prod = 1;

  std::vector<Iter> new_shard;
  std::vector<int64_t> seps{0};

  for (size_t i = 0; i < layout->shard.size(); ++i) {
    auto extent_i = layout->shard[i]->extent;
    auto stride_i = layout->shard[i]->stride;
    prod *= extent_i;
    while (shape_idx < shape.size() &&
           analyzer->CanProveEqual(floormod(prod, shape[shape_idx]), 0)) {
      PrimExpr c = analyzer->Simplify(floordiv(prod, shape[shape_idx]));
      if (!analyzer->CanProveEqual(floormod(extent_i, c), 0)) return std::nullopt;
      new_shard.push_back(Iter(analyzer->Simplify(floordiv(extent_i, c)),
                               analyzer->Simplify(stride_i * c), layout->shard[i]->axis));
      extent_i = c;
      prod = c;
      shape_idx++;
      seps.push_back(new_shard.size());
    }
    extent_i = analyzer->Simplify(extent_i);
    if (!is_one(extent_i)) {
      if (shape_idx >= shape.size()) return std::nullopt;
      new_shard.push_back(Iter(extent_i, analyzer->Simplify(stride_i), layout->shard[i]->axis));
    }
  }

  if (shape_idx != shape.size()) return std::nullopt;

  auto* n = layout.CopyOnWrite();
  n->shard = new_shard;
  return std::make_pair(ffi::GetRef<TileLayout>(n), seps);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tirx.TileLayoutGroup",
           [](const TileLayout& layout, const Array<PrimExpr>& shape) {
             auto [res, seps] = Group(layout, shape);
             return Tuple<TileLayout, Array<int64_t>>{res,
                                                      Array<int64_t>(seps.begin(), seps.end())};
           })
      .def("tirx.TileLayoutGroupMany",
           [](const TileLayout& layout, const Array<Array<PrimExpr>>& shapes) {
             auto [res, separators] = GroupMany(layout, shapes);
             Array<Array<int64_t>> ffi_separators;
             ffi_separators.reserve(separators.size());
             for (const auto& seps : separators) {
               ffi_separators.push_back(Array<int64_t>(seps.begin(), seps.end()));
             }
             return Tuple<TileLayout, Array<Array<int64_t>>>{res, ffi_separators};
           });
}

Layout TileLayoutNode::Tile(const TileLayout& outer_in, const Array<PrimExpr>& outer_shape,
                            const Array<PrimExpr>& inner_shape) const {
  auto outer = outer_in->Canonicalize().as<TileLayout>().value();
  auto inner = ffi::GetRef<TileLayout>(this)->Canonicalize().as<TileLayout>().value();

  TVM_FFI_ICHECK_EQ(outer_shape.size(), inner_shape.size())
      << "Outer and inner shape size must match";

  auto [grouped_outer, outer_seps] = Group(outer, outer_shape);
  auto [grouped_inner, inner_seps] = Group(inner, inner_shape);

  outer = grouped_outer;
  inner = grouped_inner;

  arith::Analyzer analyzer;

  {
    // Scale outer axis strides by inner span on matching axes
    auto inner_span_map = BuildSpanMap(inner);
    std::vector<Iter> new_shard;
    for (size_t i = 0; i < outer->shard.size(); ++i) {
      auto it = inner_span_map.find(outer->shard[i]->axis->name);
      if (it != inner_span_map.end()) {
        new_shard.push_back(Iter(outer->shard[i]->extent, outer->shard[i]->stride * (*it).second,
                                 outer->shard[i]->axis));
      } else {
        new_shard.push_back(outer->shard[i]);
      }
    }
    outer = TileLayout(new_shard, outer->replica, outer->offset);
  }

  TVM_FFI_ICHECK(!outer_seps.empty())
      << "Outer layout must only use split/reorder from logical scope";
  TVM_FFI_ICHECK(!inner_seps.empty())
      << "Inner layout must only use split/reorder from logical scope";

  std::vector<Iter> tile_shard;
  for (size_t i = 0; i < outer_shape.size(); ++i) {
    tile_shard.insert(tile_shard.end(), outer->shard.begin() + outer_seps[i],
                      outer->shard.begin() + outer_seps[i + 1]);

    tile_shard.insert(tile_shard.end(), inner->shard.begin() + inner_seps[i],
                      inner->shard.begin() + inner_seps[i + 1]);
  }

  std::vector<Iter> tile_rep{inner->replica.begin(), inner->replica.end()};
  tile_rep.insert(tile_rep.end(), outer->replica.begin(), outer->replica.end());

  ffi::Map<Axis, PrimExpr> tile_offset;
  for (const auto& [axis, off] : inner->offset) {
    tile_offset.Set(axis, off);
  }
  for (const auto& [axis, off] : outer->offset) {
    auto it = tile_offset.find(axis);
    if (it != tile_offset.end()) {
      tile_offset.Set(axis, (*it).second + off);
    } else {
      tile_offset.Set(axis, off);
    }
  }

  return TileLayout(tile_shard, tile_rep, tile_offset)->Canonicalize();
}

// Tiles a logical shape by a given factor array.
ffi::Array<PrimExpr> TileShape(ffi::Array<PrimExpr> shape, ffi::Array<PrimExpr> factor,
                               bool is_inner) {
  TVM_FFI_ICHECK_EQ(shape.size(), factor.size()) << "Shape and factor dimension must match.";
  arith::Analyzer analyzer;

  ffi::Array<PrimExpr> new_shape;
  for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
    TVM_FFI_ICHECK(analyzer->CanProveEqual(floormod(shape[i], factor[i]), 0))
        << "Shape[i] must be divisible by factor[i]";

    if (is_inner) {
      new_shape.push_back(floordiv(shape[i], factor[i]));
      new_shape.push_back(factor[i]);
    } else {
      new_shape.push_back(factor[i]);
      new_shape.push_back(floordiv(shape[i], factor[i]));
    }
  }
  return new_shape;
}

ffi::Array<PrimExpr> DivideShape(ffi::Array<PrimExpr> shape, ffi::Array<PrimExpr> factor) {
  ffi::Array<PrimExpr> new_shape;
  for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
    new_shape.push_back(floordiv(shape[i], factor[i]));
  }
  return new_shape;
}

// Extract every even index from seps
std::vector<int64_t> EvenSeparatorIndices(std::vector<int64_t> seps) {
  std::vector<int64_t> even;
  for (size_t i = 0; i < seps.size(); i += 2) {
    even.push_back(seps[i]);
  }
  return even;
}

// Split axes according to a split scope on the target.
TileLayout SplitAxesByScope(TileLayout layout, const ffi::String& split_scope) {
  Target target = Target::Current();
  if (!target.defined()) {
    return layout;
  }
  auto split_iter = [&](const Iter& iter) -> ffi::Array<Iter> {
    const auto& splitter = iter->axis->GetSplitter();
    if (splitter.has_value()) {
      return splitter.value()(target, split_scope, iter);
    }
    return {iter};
  };

  std::vector<Iter> shard, replica;
  ffi::Map<Axis, PrimExpr> offset;

  for (const auto& iter : layout->shard) {
    auto split_iters = split_iter(iter);
    shard.insert(shard.end(), split_iters.begin(), split_iters.end());
  }

  for (const auto& iter : layout->replica) {
    auto split_iters = split_iter(iter);
    replica.insert(replica.end(), split_iters.begin(), split_iters.end());
  }

  for (const auto& [axis, off] : layout->offset) {
    auto split_iters = split_iter(Iter(1, off, axis));
    if (split_iters.size() == 1) {
      offset.Set(split_iters[0]->axis, split_iters[0]->stride);
    } else {
      auto coord = SplitCoord(off, {split_iters[0]->extent, split_iters[1]->extent});
      TVM_FFI_ICHECK(coord.size() == 2) << "Split coord size must be 2";
      offset.Set(split_iters[0]->axis, coord[0] * split_iters[0]->stride);
      offset.Set(split_iters[1]->axis, coord[1] * split_iters[1]->stride);
    }
  }

  return TileLayout(shard, replica, offset);
}

ffi::Optional<TileLayout> TileLayoutNode::IsTileInner(
    const Layout& tile_layout, const ffi::Array<PrimExpr>& tiled_shape,
    const ffi::Array<PrimExpr>& inner_shape) const {
  auto maybe_tile = tile_layout.as<TileLayout>();
  if (!maybe_tile) return std::nullopt;

  TileLayout tiled = maybe_tile.value()->Canonicalize().as<TileLayout>().value();
  TileLayout layout = ffi::GetRef<TileLayout>(this)->Canonicalize().as<TileLayout>().value();

  auto tiled_scope = tiled->GetScope();
  auto inner_scope = layout->GetScope();
  if (tiled_scope.has_value() && inner_scope.has_value()) {
    if (tiled_scope.value().get<0>()->kind != inner_scope.value().get<0>()->kind ||
        ScopeKindHigher(inner_scope.value().get<1>()->kind, tiled_scope.value().get<1>()->kind)) {
      return std::nullopt;
    }
    if (ScopeKindHigher(tiled_scope.value().get<1>()->kind, inner_scope.value().get<1>()->kind)) {
      tiled = SplitAxesByScope(tiled, inner_scope.value().get<1>()->name());
    }
  }

  arith::Analyzer analyzer;
  // Get the span map of the inner layout of each axis
  auto inner_span_map = BuildSpanMap(layout);
  auto rescale_by_inner_span = [&](const Iter& iter) -> ffi::Optional<Iter> {
    auto it = inner_span_map.find(iter->axis->name);
    if (it != inner_span_map.end() && !is_one(iter->extent)) {
      if (!analyzer->CanProveEqual(floormod(iter->stride, (*it).second), 0)) {
        return std::nullopt;
      }
      return Iter(iter->extent, floordiv(iter->stride, (*it).second), iter->axis);
    }
    return iter;
  };

  TVM_FFI_ICHECK_EQ(tiled_shape.size(), inner_shape.size())
      << "Tiled shape size must match inner shape size";

  auto factored = TileShape(tiled_shape, inner_shape, true);
  auto [grouped_tiled, tiled_seps] = Group(tiled, factored);
  TVM_FFI_ICHECK(grouped_tiled.defined() && !tiled_seps.empty())
      << "tile layout group by shape failed, layout is " << tiled << " and shape is " << factored;
  auto [grouped_layout, inner_seps] = Group(layout, inner_shape);
  TVM_FFI_ICHECK(grouped_layout.defined() && !inner_seps.empty())
      << "tile layout group by shape failed, layout is " << layout << " and shape is "
      << inner_shape;

  auto tiled_seps_even = EvenSeparatorIndices(tiled_seps);

  // Gather outer shards
  std::vector<Iter> outer_shard;
  for (size_t i = 0; i < tiled_shape.size(); ++i) {
    int inner_count = inner_seps[i + 1] - inner_seps[i];
    int tiled_count = tiled_seps_even[i + 1] - tiled_seps_even[i];
    if (inner_count > tiled_count) return std::nullopt;

    // Compare extents (and stride/axis if extent is not 1).
    for (int j = 0; j < inner_count; ++j) {
      Iter inner_iter = grouped_layout->shard[inner_seps[i] + j];
      Iter tiled_iter = grouped_tiled->shard[tiled_seps_even[i + 1] - inner_count + j];
      if (!analyzer->CanProveEqual(inner_iter->extent, tiled_iter->extent) ||
          (!is_one(inner_iter->extent) &&
           !(analyzer->CanProveEqual(inner_iter->stride, tiled_iter->stride) &&
             inner_iter->axis.same_as(tiled_iter->axis)))) {
        return std::nullopt;
      }
    }
    for (int j = 0; j < tiled_count - inner_count; ++j) {
      auto outer_iter = rescale_by_inner_span(grouped_tiled->shard[tiled_seps_even[i] + j]);
      if (!outer_iter.has_value()) return std::nullopt;
      outer_shard.push_back(outer_iter.value());
    }
  }

  // Gather outer replicate
  std::vector<Iter> outer_replicate;
  for (const auto& tiled_iter : tiled->replica) {
    if (std::none_of(layout->replica.begin(), layout->replica.end(), [&](const Iter& inner_iter) {
          return StructuralEqual()(tiled_iter, inner_iter);
        })) {
      auto outer_iter = rescale_by_inner_span(tiled_iter);
      if (!outer_iter.has_value()) return std::nullopt;
      outer_replicate.push_back(outer_iter.value());
    }
  }
  // Gather outer offset
  ffi::Map<Axis, PrimExpr> outer_exclude;
  for (const auto& [axis, off] : tiled->offset) {
    auto it = layout->offset.find(axis);
    if (it != layout->offset.end()) {
      outer_exclude.Set(axis, analyzer->Simplify(off - (*it).second));
    } else {
      outer_exclude.Set(axis, off);
    }
  }
  return TileLayout(outer_shard, outer_replicate, outer_exclude);
}

ffi::Optional<Layout> TileLayoutNode::IsTileOuter(const Layout& tile_layout,
                                                  const ffi::Array<PrimExpr>& tiled_shape,
                                                  const ffi::Array<PrimExpr>& outer_shape) const {
  auto maybe_tile = tile_layout.as<TileLayout>();
  if (!maybe_tile) {
    if (auto comp = tile_layout.as<ComposeLayout>()) {
      auto inner_layout = IsTileOuter(comp.value()->tile_layout, tiled_shape, outer_shape);
      if (!inner_layout) return std::nullopt;
      return ComposeLayout(comp.value()->per_element, comp.value()->swizzle_len,
                           comp.value()->atom_len, inner_layout.value().as<TileLayout>().value(),
                           comp.value()->swizzle_inner);
    }
    return std::nullopt;
  }
  TileLayout tiled = maybe_tile.value()->Canonicalize().as<TileLayout>().value();
  TileLayout layout = ffi::GetRef<TileLayout>(this)->Canonicalize().as<TileLayout>().value();

  auto tiled_scope = tiled->GetScope();
  auto outer_scope = layout->GetScope();
  if (tiled_scope.has_value() && outer_scope.has_value()) {
    if (tiled_scope.value().get<1>()->kind != outer_scope.value().get<1>()->kind ||
        ScopeKindHigher(tiled_scope.value().get<0>()->kind, outer_scope.value().get<0>()->kind)) {
      return std::nullopt;
    }
    if (ScopeKindHigher(outer_scope.value().get<0>()->kind, tiled_scope.value().get<0>()->kind)) {
      tiled = SplitAxesByScope(tiled, outer_scope.value().get<0>()->name());
    }
  }

  arith::Analyzer analyzer;
  TVM_FFI_ICHECK_EQ(tiled_shape.size(), outer_shape.size())
      << "Tiled shape size must match outer shape size";

  auto factored = TileShape(tiled_shape, outer_shape, false);
  auto [grouped_tiled, tiled_seps] = Group(tiled, factored);
  TVM_FFI_ICHECK(grouped_tiled.defined() && !tiled_seps.empty())
      << "tile layout group by shape failed, layout is " << tiled << " and shape is " << factored;
  auto [grouped_layout, outer_seps] = Group(layout, outer_shape);
  TVM_FFI_ICHECK(grouped_layout.defined() && !outer_seps.empty())
      << "tile layout group by shape failed, layout is " << layout << " and shape is "
      << outer_shape;

  auto tiled_seps_even = EvenSeparatorIndices(tiled_seps);

  std::vector<Iter> inner_shard;
  for (size_t i = 0; i < tiled_shape.size(); ++i) {
    int outer_count = outer_seps[i + 1] - outer_seps[i];
    int tiled_count = tiled_seps_even[i + 1] - tiled_seps_even[i];
    if (outer_count > tiled_count) return std::nullopt;

    for (int j = 0; j < outer_count; ++j) {
      Iter outer_iter = grouped_layout->shard[outer_seps[i] + j];
      Iter tiled_iter = grouped_tiled->shard[tiled_seps_even[i] + j];
      if (!analyzer->CanProveEqual(outer_iter->extent, tiled_iter->extent) ||
          (!is_one(outer_iter->extent) && !outer_iter->axis.same_as(tiled_iter->axis))) {
        return std::nullopt;
      }
    }

    for (int j = 0; j < tiled_count - outer_count; ++j) {
      Iter inner_iter = grouped_tiled->shard[tiled_seps_even[i] + outer_count + j];
      inner_shard.push_back(inner_iter);
    }
  }

  std::vector<Iter> inner_replicate;
  for (const auto& tiled_iter : tiled->replica) {
    if (std::none_of(layout->replica.begin(), layout->replica.end(), [&](const Iter& inner_iter) {
          return StructuralEqual()(tiled_iter, inner_iter);
        })) {
      inner_replicate.push_back(tiled_iter);
    }
  }
  ffi::Map<Axis, PrimExpr> inner_exclude;
  for (const auto& [axis, off] : tiled->offset) {
    auto it = layout->offset.find(axis);
    if (it != layout->offset.end()) {
      inner_exclude.Set(axis, analyzer->Simplify(off - (*it).second));
    } else {
      inner_exclude.Set(axis, off);
    }
  }

  auto inner_layout = TileLayout(inner_shard, inner_replicate, inner_exclude);
  auto try_tile = inner_layout->Tile(layout, outer_shape, DivideShape(tiled_shape, outer_shape));
  if (StructuralEqual()(try_tile->Canonicalize(), tiled->Canonicalize())) {
    return inner_layout;
  }
  return std::nullopt;
}

}  // namespace tirx
}  // namespace tvm
