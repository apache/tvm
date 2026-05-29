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
 * Direct-sum operations (unscaled composition) for TileLayout and helpers.
 */
#include "tile_internal.h"

namespace tvm {
namespace tirx {

Layout TileLayoutNode::DirectSum(const TileLayout& left_in, const Array<PrimExpr>& left_shape,
                                 const Array<PrimExpr>& right_shape) const {
  // Canonicalize inputs
  auto left = left_in->Canonicalize().as<TileLayout>().value();
  auto right = ffi::GetRef<TileLayout>(this)->Canonicalize().as<TileLayout>().value();

  TVM_FFI_ICHECK_EQ(left_shape.size(), right_shape.size())
      << "Left and right shape size must match for direct sum";

  // Group both layouts by their respective shapes
  auto [grouped_left, left_seps] = Group(left, left_shape);
  auto [grouped_right, right_seps] = Group(right, right_shape);

  left = grouped_left;
  right = grouped_right;

  // Interleave per-rank blocks: [A-block || B-block] for each rank position
  std::vector<Iter> sum_shard;
  for (size_t i = 0; i < left_shape.size(); ++i) {
    sum_shard.insert(sum_shard.end(), left->shard.begin() + left_seps[i],
                     left->shard.begin() + left_seps[i + 1]);
    sum_shard.insert(sum_shard.end(), right->shard.begin() + right_seps[i],
                     right->shard.begin() + right_seps[i + 1]);
  }

  // Replicas concatenate: R^A || R^B
  std::vector<Iter> sum_rep{left->replica.begin(), left->replica.end()};
  sum_rep.insert(sum_rep.end(), right->replica.begin(), right->replica.end());

  // Offsets add: O^A + O^B per-axis
  arith::Analyzer analyzer;
  ffi::Map<Axis, PrimExpr> sum_off;
  for (const auto& [axis, off] : left->offset) sum_off.Set(axis, off);
  for (const auto& [axis, off] : right->offset) {
    auto it = sum_off.find(axis);
    if (it != sum_off.end()) {
      sum_off.Set(axis, analyzer.Simplify((*it).second + off));
    } else {
      sum_off.Set(axis, off);
    }
  }

  return TileLayout(sum_shard, sum_rep, sum_off)->Canonicalize();
}

static bool IterEqualRelaxUnit(const Iter& a, const Iter& b, arith::Analyzer* analyzer) {
  if (!(*analyzer).CanProveEqual(a->extent, b->extent)) return false;
  if (!is_one(a->extent)) {
    if (!(*analyzer).CanProveEqual(a->stride, b->stride)) return false;
    if (!a->axis.same_as(b->axis)) return false;
  }
  return true;
}

// Helper to subtract offsets: left = sum - right
static ffi::Map<Axis, PrimExpr> SubtractOffsets(const ffi::Map<Axis, PrimExpr>& sum,
                                                const ffi::Map<Axis, PrimExpr>& rhs) {
  arith::Analyzer analyzer;
  ffi::Map<Axis, PrimExpr> res;
  for (const auto& [axis, off] : sum) res.Set(axis, off);
  for (const auto& [axis, off] : rhs) {
    auto it = res.find(axis);
    if (it != res.end()) {
      res.Set(axis, analyzer.Simplify((*it).second - off));
    } else {
      res.Set(axis, analyzer.Simplify(-off));
    }
  }
  return res;
}

ffi::Optional<TileLayout> TileLayoutNode::IsDirectSumRight(
    const Layout& sum_layout_in, const ffi::Array<PrimExpr>& interleaved_shape,
    const ffi::Array<PrimExpr>& right_shape) const {
  auto maybe_sum = sum_layout_in.as<TileLayout>();
  if (!maybe_sum) return std::nullopt;

  arith::Analyzer analyzer;
  TileLayout sum_layout = maybe_sum.value()->Canonicalize().as<TileLayout>().value();
  TileLayout right = ffi::GetRef<TileLayout>(this)->Canonicalize().as<TileLayout>().value();

  TVM_FFI_ICHECK_EQ(interleaved_shape.size(), right_shape.size() * 2)
      << "Interleaved shape must have twice the rank of right_shape";

  auto [grouped_sum, sum_seps] = Group(sum_layout, interleaved_shape);
  auto [grouped_right, right_seps] = Group(right, right_shape);

  // Collect left shard (A) from grouped_sum by removing matched right block per rank.
  std::vector<Iter> left_shard;
  for (size_t i = 0; i < right_shape.size(); ++i) {
    int sum_left_cnt = sum_seps[2 * i + 1] - sum_seps[2 * i];
    int sum_right_cnt = sum_seps[2 * i + 2] - sum_seps[2 * i + 1];
    int right_cnt = right_seps[i + 1] - right_seps[i];
    if (right_cnt > sum_right_cnt) return std::nullopt;

    // Left part goes directly into left_shard
    for (int j = 0; j < sum_left_cnt; ++j) {
      left_shard.push_back(grouped_sum->shard[sum_seps[2 * i] + j]);
    }
    // Verify right part matches this layout's grouped_right
    for (int j = 0; j < right_cnt; ++j) {
      Iter s_iter = grouped_sum->shard[sum_seps[2 * i + 2] - right_cnt + j];
      Iter r_iter = grouped_right->shard[right_seps[i] + j];
      if (!IterEqualRelaxUnit(s_iter, r_iter, &analyzer)) return std::nullopt;
    }
    // If sum_right_cnt > right_cnt, residual dims cannot be attributed; reject for now.
    if (sum_right_cnt != right_cnt) return std::nullopt;
  }

  // Replicas: left = sum - right
  std::vector<Iter> left_rep;
  for (const auto& it : sum_layout->replica) {
    bool is_right = std::any_of(right->replica.begin(), right->replica.end(),
                                [&](const Iter& r) { return StructuralEqual()(it, r); });
    if (!is_right) left_rep.push_back(it);
  }

  // Offsets: left = sum - right
  auto left_off = SubtractOffsets(sum_layout->offset, right->offset);
  return TileLayout(left_shard, left_rep, left_off);
}

ffi::Optional<Layout> TileLayoutNode::IsDirectSumLeft(
    const Layout& sum_layout_in, const ffi::Array<PrimExpr>& interleaved_shape,
    const ffi::Array<PrimExpr>& left_shape) const {
  auto maybe_sum = sum_layout_in.as<TileLayout>();
  if (!maybe_sum) return std::nullopt;

  arith::Analyzer analyzer;
  TileLayout sum_layout = maybe_sum.value()->Canonicalize().as<TileLayout>().value();
  TileLayout left = ffi::GetRef<TileLayout>(this)->Canonicalize().as<TileLayout>().value();

  TVM_FFI_ICHECK_EQ(interleaved_shape.size(), left_shape.size() * 2)
      << "Interleaved shape must have twice the rank of left_shape";

  auto [grouped_sum, sum_seps] = Group(sum_layout, interleaved_shape);
  auto [grouped_left, left_seps] = Group(left, left_shape);

  // Collect right shard (B) from grouped_sum by removing matched left block per rank.
  std::vector<Iter> right_shard;
  for (size_t i = 0; i < left_shape.size(); ++i) {
    int sum_left_cnt = sum_seps[2 * i + 1] - sum_seps[2 * i];
    int sum_right_cnt = sum_seps[2 * i + 2] - sum_seps[2 * i + 1];
    int left_cnt = left_seps[i + 1] - left_seps[i];
    if (left_cnt > sum_left_cnt) return std::nullopt;

    // Verify left part matches this layout's grouped_left
    for (int j = 0; j < left_cnt; ++j) {
      Iter s_iter = grouped_sum->shard[sum_seps[2 * i] + j];
      Iter l_iter = grouped_left->shard[left_seps[i] + j];
      if (!IterEqualRelaxUnit(s_iter, l_iter, &analyzer)) return std::nullopt;
    }
    // If sum_left_cnt > left_cnt, residual dims cannot be attributed; reject for now.
    if (sum_left_cnt != left_cnt) return std::nullopt;

    // Right part goes directly into right_shard
    for (int j = 0; j < sum_right_cnt; ++j) {
      right_shard.push_back(grouped_sum->shard[sum_seps[2 * i + 1] + j]);
    }
  }

  // Replicas: right = sum - left
  std::vector<Iter> right_rep;
  for (const auto& it : sum_layout->replica) {
    bool is_left = std::any_of(left->replica.begin(), left->replica.end(),
                               [&](const Iter& l) { return StructuralEqual()(it, l); });
    if (!is_left) right_rep.push_back(it);
  }

  // Offsets: right = sum - left
  auto right_off = SubtractOffsets(sum_layout->offset, left->offset);
  return TileLayout(right_shard, right_rep, right_off);
}

Layout ComposeLayoutNode::DirectSum(const TileLayout& left, const Array<PrimExpr>& left_shape,
                                    const Array<PrimExpr>& right_shape) const {
  // Direct-sum applies to the tile layout then compose with swizzle.
  auto right_sum = tile_layout->DirectSum(left, left_shape, right_shape).as<TileLayout>().value();
  return ComposeLayout(swizzle, right_sum);
}

ffi::Optional<TileLayout> ComposeLayoutNode::IsDirectSumRight(
    const Layout& sum_layout, const ffi::Array<PrimExpr>& interleaved_shape,
    const ffi::Array<PrimExpr>& right_shape) const {
  if (auto comp = sum_layout.as<ComposeLayout>()) {
    if (StructuralEqual()(comp.value()->swizzle, this->swizzle)) {
      return this->tile_layout->IsDirectSumRight(comp.value()->tile_layout, interleaved_shape,
                                                 right_shape);
    }
  }
  return std::nullopt;
}

ffi::Optional<Layout> ComposeLayoutNode::IsDirectSumLeft(
    const Layout& sum_layout, const ffi::Array<PrimExpr>& interleaved_shape,
    const ffi::Array<PrimExpr>& left_shape) const {
  if (auto comp = sum_layout.as<ComposeLayout>()) {
    if (StructuralEqual()(comp.value()->swizzle, this->swizzle)) {
      return this->tile_layout->IsDirectSumLeft(comp.value()->tile_layout, interleaved_shape,
                                                left_shape);
    }
  }
  return std::nullopt;
}

Layout SwizzleLayoutNode::DirectSum(const TileLayout& left, const Array<PrimExpr>& left_shape,
                                    const Array<PrimExpr>& right_shape) const {
  // Compose(Swizzle, Identity(right_shape)) then direct-sum with left.
  auto comp = ComposeLayout(ffi::GetRef<SwizzleLayout>(this), IdentityTileLayout(right_shape));
  return comp->DirectSum(left, left_shape, right_shape);
}

ffi::Optional<TileLayout> SwizzleLayoutNode::IsDirectSumRight(
    const Layout& sum_layout, const ffi::Array<PrimExpr>& interleaved_shape,
    const ffi::Array<PrimExpr>& right_shape) const {
  if (auto comp = sum_layout.as<ComposeLayout>()) {
    if (StructuralEqual()(comp.value()->swizzle, ffi::GetRef<SwizzleLayout>(this))) {
      return comp.value()->tile_layout->IsDirectSumRight(sum_layout, interleaved_shape,
                                                         right_shape);
    }
  }
  return std::nullopt;
}

ffi::Optional<Layout> SwizzleLayoutNode::IsDirectSumLeft(
    const Layout& sum_layout, const ffi::Array<PrimExpr>& interleaved_shape,
    const ffi::Array<PrimExpr>& left_shape) const {
  if (auto comp = sum_layout.as<ComposeLayout>()) {
    if (StructuralEqual()(comp.value()->swizzle, ffi::GetRef<SwizzleLayout>(this))) {
      return comp.value()->tile_layout->IsDirectSumLeft(sum_layout, interleaved_shape, left_shape);
    }
  }
  return std::nullopt;
}

}  // namespace tirx
}  // namespace tvm
