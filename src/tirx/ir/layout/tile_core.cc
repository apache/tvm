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
 * Core TileLayout and Iter methods, basic queries, and reflection registration.
 */
#include "utils.h"

namespace tvm {
namespace tirx {

TVM_FFI_STATIC_INIT_BLOCK() {
  AxisNode::RegisterReflection();
  IterNode::RegisterReflection();
  TileLayoutNode::RegisterReflection();
  SwizzleLayoutNode::RegisterReflection();
  ComposeLayoutNode::RegisterReflection();
}

/**************** Iter ****************/
Iter::Iter(PrimExpr extent, PrimExpr stride, Axis axis) {
  auto n = ffi::make_object<IterNode>();
  n->extent = extent;
  n->stride = stride;
  n->axis = axis;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.Iter", [](PrimExpr extent, PrimExpr stride, Axis axis) {
    return Iter(extent, stride, axis);
  });
}

/**************** TileLayout ****************/
TileLayout::TileLayout(ffi::Array<Iter> shard, ffi::Array<Iter> replica,
                       ffi::Map<Axis, PrimExpr> offset) {
  auto n = ffi::make_object<TileLayoutNode>();
  n->shard = shard;
  n->replica = replica;
  n->offset = offset;
  data_ = std::move(n);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.TileLayout", [](ffi::Array<Iter> shard, ffi::Array<Iter> replica,
                                              ffi::Map<Axis, PrimExpr> offset) {
    return TileLayout(shard, replica, offset);
  });
}

bool TileLayoutNode::CompatibleWithShape(const Array<PrimExpr>& shape) const { return true; }

bool VerifyCompactness(const std::vector<Iter>& iters) {
  arith::Analyzer analyzer;
  PrimExpr stride_to_find = 1;
  for (size_t i = 0; i < iters.size(); ++i) {
    auto iter = std::find_if(iters.begin(), iters.end(), [&](const Iter& iter) {
      return analyzer.CanProveEqual(iter->stride, stride_to_find);
    });
    if (iter == iters.end()) return false;
    stride_to_find *= (*iter)->extent;
  }
  return true;
}

bool TileLayoutNode::VerifyWellFormed() const {
  // // 1. For thread axes, verify its compactness
  // std::unordered_map<String, std::vector<Iter>> thread_axes;
  // auto collect_thread_axis = [&thread_axes](const Iter& iter) {
  //   if (iter->axis->IsThreadAxis()) {
  //     thread_axes[iter->axis->name].push_back(iter);
  //   }
  // };
  // for (const auto& iter : shard) {
  //   collect_thread_axis(iter);
  // }
  // for (const auto& iter : replica) {
  //   collect_thread_axis(iter);
  // }
  // for (const auto& [axis, off] : offset) {
  //   collect_thread_axis(Iter(1, off, axis));
  // }
  // for (const auto& [axis, iters] : thread_axes) {
  //   if (!VerifyCompactness(iters)) {
  //     return false;
  //   }
  // }
  // 1. Check if the scope is connected
  if (!GetScope().defined() && HasThreadAxis()) {
    return false;
  }
  return true;
}

PrimExpr TileLayoutNode::GetSize(ffi::Optional<ffi::String> axis_name) const {
  auto filter = [&](const Iter& iter, PrimExpr acc) {
    if (!axis_name.has_value() || iter->axis->name == axis_name.value()) {
      return acc * iter->extent;
    }
    return acc;
  };
  PrimExpr res = 1;
  for (const auto& iter : shard) {
    res = filter(iter, res);
  }
  return res;
}

PrimExpr TileLayoutNode::GetSpan(ffi::Optional<ffi::String> axis_name) const {
  arith::Analyzer analyzer;
  PrimExpr result = 1;
  auto filter = [&](const Axis& axis) { return AxisMatchesFilter(axis, axis_name); };

  for (const auto& iter : shard) {
    if (filter(iter->axis)) result += (iter->extent - 1) * iter->stride;
  }
  for (const auto& iter : replica) {
    if (filter(iter->axis)) result += (iter->extent - 1) * iter->stride;
  }
  for (const auto& [axis, off] : offset) {
    if (filter(axis)) result += off;
  }
  return analyzer.Simplify(result);
}

ffi::Map<ffi::String, PrimExpr> TileLayoutNode::Apply(PrimExpr coord) const {
  return Apply(SplitCoord(coord, GetShardShape()));
}

ffi::Map<ffi::String, PrimExpr> TileLayoutNode::Apply(Array<PrimExpr> coord) const {
  arith::Analyzer analyzer;
  TVM_FFI_ICHECK_EQ(coord.size(), shard.size())
      << "Coordinate size must match the number of shard axes";
  std::unordered_map<ffi::String, PrimExpr> result;
  for (size_t i = 0; i < shard.size(); ++i) {
    auto it = result.find(shard[i]->axis->name);
    if (it == result.end()) {
      result[shard[i]->axis->name] = analyzer.Simplify(coord[i] * shard[i]->stride);
    } else {
      result[shard[i]->axis->name] = analyzer.Simplify(it->second + coord[i] * shard[i]->stride);
    }
  }
  // Add offset to the result
  for (const auto& [axis, off] : offset) {
    auto it = result.find(axis->name);
    if (it == result.end()) {
      result[axis->name] = analyzer.Simplify(off);
    } else {
      result[axis->name] = analyzer.Simplify(it->second + off);
    }
  }
  return result;
}

ffi::Array<PrimExpr> TileLayoutNode::GetShardShape() const {
  return shard.Map([](const Iter& iter) { return iter->extent; });
}

bool TileLayoutNode::IsTrivial() const {
  if (shard.size() > 1) return false;
  if (shard.size() == 1) {
    if (!shard[0]->axis->IsMemoryAxis() || !is_one(shard[0]->stride)) return false;
  }
  return replica.size() == 0 && offset.size() == 0;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.TileLayoutIsTrivial", [](const TileLayout& layout) {
    return layout->Canonicalize().as<TileLayout>().value()->IsTrivial();
  });
}

bool TileLayoutNode::IsTrainium() const {
  return !std::any_of(shard.begin(), shard.end(), [](const Iter& iter) {
    return iter->axis->IsMemoryAxis() && !iter->axis.same_as(Axis::Get("F")) &&
           !iter->axis.same_as(Axis::Get("P")) && !iter->axis.same_as(Axis::Get("Bank"));
  });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tirx.TileLayoutIsTrainium",
                        [](const TileLayout& layout) { return layout->IsTrainium(); });
}

bool TileLayoutNode::HasMemoryAxis() const {
  return std::any_of(shard.begin(), shard.end(),
                     [](const Iter& iter) { return iter->axis->IsMemoryAxis(); });
}

bool TileLayoutNode::HasThreadAxis() const {
  return std::any_of(shard.begin(), shard.end(),
                     [](const Iter& iter) { return iter->axis->IsThreadAxis(); });
}

ffi::Optional<ffi::Tuple<ExecScope, ExecScope>> TileLayoutNode::GetScope() const {
  if (!HasThreadAxis()) return std::nullopt;

  std::unordered_map<ffi::String, ffi::String> scope_map;
  ffi::Optional<ffi::String> inner_most;

  auto check_axis = [&](const Axis& axis) {
    if (!axis->IsThreadAxis()) return;

    auto subtile_primitivet = axis->GetSubscope();
    auto tile_primitivet = axis->GetScope();
    TVM_FFI_ICHECK(subtile_primitivet.defined() && tile_primitivet.defined())
        << "Thread axis " << axis->name << " has no subscope or scope";

    ffi::String subscope = subtile_primitivet.value()->name();
    ffi::String scope = tile_primitivet.value()->name();

    if (!inner_most.has_value() || ScopeNameHigher(inner_most.value(), subscope))
      inner_most = subscope;

    auto it = scope_map.find(subscope);
    if (it == scope_map.end())
      scope_map[subscope] = scope;
    else
      TVM_FFI_ICHECK_EQ(it->second, scope)
          << "Ill-formed tile layout: conflicting scopes for " << subscope;
  };

  for (const auto& iter : shard) check_axis(iter->axis);
  for (const auto& iter : replica) check_axis(iter->axis);
  for (const auto& [axis, off] : offset) check_axis(axis);

  ffi::String outer_most = inner_most.value();
  size_t count = 0;
  for (auto it = scope_map.find(outer_most); it != scope_map.end();
       it = scope_map.find(outer_most)) {
    count++;
    outer_most = it->second;
  }

  TVM_FFI_ICHECK_EQ(count, scope_map.size()) << "Ill-formed tile layout: disconnected scope chain";
  return Tuple<ExecScope, ExecScope>{ExecScope(inner_most.value()), ExecScope(outer_most)};
}

TileLayout TileLayoutNode::DefaultLayout(ffi::Array<PrimExpr> shape) {
  Array<Iter> shard;
  auto strides = GetDefaultStrides(shape);
  for (size_t i = 0; i < shape.size(); ++i) {
    shard.push_back(Iter(shape[i], strides[i], Axis::Get("m")));
  }
  return TileLayout(shard, ffi::Array<Iter>(), ffi::Map<Axis, PrimExpr>());
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tirx.TileLayoutGetScope",
      [](const TileLayout& layout) -> ffi::Optional<ffi::Tuple<ExecScope, ExecScope>> {
        return layout->GetScope();
      });
}

}  // namespace tirx
}  // namespace tvm
