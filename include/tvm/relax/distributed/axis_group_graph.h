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

#ifndef TVM_RELAX_DISTRIBUTED_AXIS_GROUP_GRAPH_H_
#define TVM_RELAX_DISTRIBUTED_AXIS_GROUP_GRAPH_H_

#include <tvm/relax/distributed/struct_info.h>
#include <tvm/relax/expr.h>
#include <tvm/tir/function.h>

#include <algorithm>
#include <limits>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
namespace tvm {
namespace relax {
namespace distributed {

/*! \brief tensor axis*/
struct Axis {
  const ExprNode* tensor;
  int dim = 0;

  Axis(const ExprNode* tensor, int dim) : tensor(tensor), dim(dim) {
    ICHECK(tensor->IsInstance<ConstantNode>() || tensor->IsInstance<VarNode>());
  }

  bool operator==(const Axis& other) const { return tensor == other.tensor && dim == other.dim; }
};

class AxisHash {
 public:
  size_t operator()(const Axis& axis) const {
    size_t const h1(std::hash<const ExprNode*>()(axis.tensor));
    size_t const h2(std::hash<int>()(axis.dim));
    return h1 ^ (h2 << 1);
  }
};

using AxisGroup = std::unordered_set<Axis, AxisHash>;

class AxisGroupHash {
 public:
  size_t operator()(const AxisGroup& axis_group) const {
    size_t seed = 0;
    for (auto axis : axis_group) {
      seed ^= AxisHash()(axis) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

using ShardingSpec = std::pair<DeviceMesh, Placement>;

// device mesh and the device mesh axis that the tensor axis maps to
using AxisShardingSpec = std::pair<DeviceMesh, int>;
class AxisShardingSpecEqual {
 public:
  bool operator()(const AxisShardingSpec& lhs, const AxisShardingSpec& rhs) const {
    return StructuralEqual()(lhs.first, rhs.first) && lhs.second == rhs.second;
  }
};

class AxisShardingSpecHash {
 public:
  size_t operator()(const AxisShardingSpec& sharding_spec) const {
    size_t seed = 0;
    seed ^= StructuralHash()(sharding_spec.first);
    seed ^= std::hash<int>()(sharding_spec.second) << 1;
    return seed;
  }
};

/*!
 * \brief A graph whose nodes are tensor axes, and the edge means some information can be propagated
 * through the two axes. Although it only does sharding propagation, this data structure can be
 * extended to perform all kinds of propagation that happens on tensor axes.
 */
class AxisGroupGraph {
 public:
  enum class EdgeType { kAscend, kDescend, kSimbling };

 private:
  static EdgeType ReverseEdgeType(EdgeType type) {
    switch (type) {
      case EdgeType::kAscend:
        return EdgeType::kDescend;
      case EdgeType::kDescend:
        return EdgeType::kAscend;
      case EdgeType::kSimbling:
        return EdgeType::kSimbling;
    }
    LOG(FATAL) << "Unreachable code";
    throw;
  }

  static int GetEdgePriority(EdgeType type) {
    switch (type) {
      case EdgeType::kAscend:
        return 0;
      case EdgeType::kDescend:
        return 2;
      case EdgeType::kSimbling:
        return 1;
    }
    LOG(FATAL) << "Unreachable code";
    throw;
  }

  struct AxisGraphEdge {
    Axis src;
    Axis dst;

    // the producer-consumer relationship between src tensor and dst tensor
    // kAscend means consumer->producer
    // kDescend means producer->consumer
    // kSimbling means other cases
    EdgeType type;

    bool operator==(const AxisGraphEdge& other) const {
      return src == other.src && dst == other.dst && type == other.type;
    }
  };

  struct Path {
    int direction = 0;

    Path AddEdge(EdgeType type) { return {direction |= (1 << GetEdgePriority(type))}; }

    int GetPriority() const {
      switch (direction) {
        case 1:  // ascend only
          return 0;
        case 4:  // descend only
          return 2;
        case 0:      // empty path (source node)
          return 3;  // source node must have max priority
        default:     // mixed path
          return 1;
      }
    }
  };

 public:
  AxisGroupGraph() = default;

  /*!
   * \brief add edge between two axes
   * \param axis1 The src axis
   * \param axis2 The dst axis
   * \param type  The producer-consumer relationship between src tensor and dst tensor
   *              kAscend means consumer->producer
   *              kDescend means producer->consumer
   *              kSimbling means other cases
   */
  void JoinAxis(Axis axis1, Axis axis2, EdgeType type) {
    AddEdge(axis1, axis2, type);
    AddEdge(axis2, axis1, ReverseEdgeType(type));
  }

  /*!
   * \brief add a source shardingspec to propagate
   * \param axis The source axis
   * \param spec The axis's sharding spec
   */
  void AddSrcShardingPoint(Axis axis, AxisShardingSpec spec) {
    src_axis_sharding_spec_[axis] = spec;
  }

  /*!
   * \brief propagate sharding specs from source axes
   */
  void PropagateShardingSpec() {
    axis_sharding_specs_priority_.clear();
    for (const auto& pr : src_axis_sharding_spec_) {
      std::unordered_set<Axis, AxisHash> visited;
      PropagateShardingSpec(pr.first, pr.second, Path(), &visited);
    }
    ChooseAxisShardingSpec();
  }

  /*!
   * \brief add a cut point that stops the propagation of a certain sharding spec
   *
   * \param axis The cut point
   * \param spec The spec to stop propagation
   */
  void AddPropagationCutPoint(Axis axis, AxisShardingSpec spec) {
    cutpoint_axis_sharding_spec_[axis] = spec;
  }

  /*!
   * \brief Get the Sharding Spec of an axis after propagation
   *
   * \param axis the specified axis
   * \return if a sharding spec is found, return (axis_sharding_spec, true)
   *         otherwise, return (null axis_sharding_spec, false)
   */
  std::tuple<AxisShardingSpec, bool> GetAxisShardingSpec(Axis axis) {
    if (axis_sharding_specs_priority_.count(axis)) {
      return {axis_sharding_specs_priority_[axis].begin()->first, true};
    } else {
      return {{DeviceMesh(), -1}, false};
    }
  }

 private:
  void AddEdge(Axis src, Axis dst, EdgeType type) {
    if (!graph_.count(src)) {
      graph_[src] = {};
    }
    graph_[src].push_back({src, dst, type});
  }

  void PropagateShardingSpec(Axis axis, AxisShardingSpec spec, Path path,
                             std::unordered_set<Axis, AxisHash>* visited) {
    if (cutpoint_axis_sharding_spec_.count(axis) ||
        (src_axis_sharding_spec_.count(axis) &&
         !AxisShardingSpecEqual()(src_axis_sharding_spec_[axis], spec)) ||
        visited->count(axis)) {
      return;
    }
    visited->insert(axis);
    if (!axis_sharding_specs_priority_.count(axis)) {
      axis_sharding_specs_priority_[axis] = {};
    }
    axis_sharding_specs_priority_[axis][spec] = path.GetPriority();
    for (auto edge : graph_[axis]) {
      PropagateShardingSpec(edge.dst, spec, path.AddEdge(edge.type), visited);
    }
  }

  void ChooseAxisShardingSpec() {
    for (auto& pr : axis_sharding_specs_priority_) {
      auto& axis = pr.first;
      auto& specs = pr.second;
      int max_priority = std::numeric_limits<int>::min();
      for (auto& pr2 : specs) {
        max_priority = std::max(max_priority, pr2.second);
      }
      for (auto it = specs.begin(); it != specs.end();) {
        if (it->second != max_priority) {
          it = specs.erase(it);
        } else {
          it++;
        }
      }
      ICHECK(specs.size() == 1) << "multiple possible sharding for axis: ("
                                << GetRef<Expr>(axis.tensor) << ", " << axis.dim << ")";
    }
  }

  // union set
  std::unordered_map<Axis, std::vector<AxisGraphEdge>, AxisHash> graph_;
  std::unordered_map<Axis, AxisShardingSpec, AxisHash> src_axis_sharding_spec_;
  std::unordered_map<Axis, AxisShardingSpec, AxisHash> cutpoint_axis_sharding_spec_;
  std::unordered_map<
      Axis, std::unordered_map<AxisShardingSpec, int, AxisShardingSpecHash, AxisShardingSpecEqual>,
      AxisHash>
      axis_sharding_specs_priority_;
};

using FBuildAxisGraph = std::function<void(const Var& output_var, const Call& call,
                                           distributed::AxisGroupGraph* axis_group_graph)>;

void BuildAxisGraphUnary(const Var& output_var, const Call& call,
                         distributed::AxisGroupGraph* axis_group_graph);
void BuildAxisGraphBinary(const Var& output_var, const Call& call,
                          distributed::AxisGroupGraph* axis_group_graph);
void BuildAxisGraphReduce(const Var& output_var, const Call& call,
                          distributed::AxisGroupGraph* axis_group_graph);
void BuildAxisGraphMatmul(const Var& output_var, const Call& call,
                          distributed::AxisGroupGraph* axis_group_graph);
void BuildAxisGraphPermuteDims(const Var& output_var, const Call& call,
                               distributed::AxisGroupGraph* axis_group_graph);
void BuildAxisGraphReshape(const Var& output_var, const Call& call,
                           distributed::AxisGroupGraph* axis_group_graph);
// assume output must be a tensor/dtensor (not tuple)
void BuildAxisGraphCallTIR(const Var& output_var, const Call& call, const tir::PrimFunc& func,
                           distributed::AxisGroupGraph* axis_group_graph);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DISTRIBUTED_AXIS_GROUP_GRAPH_H_
