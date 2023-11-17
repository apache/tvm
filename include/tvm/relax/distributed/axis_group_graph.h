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

#include <tvm/arith/iter_affine_map.h>
#include <tvm/relax/distributed/struct_info.h>
#include <tvm/relax/expr.h>
#include <tvm/tir/function.h>
#include <tvm/tir/stmt_functor.h>

#include <algorithm>
#include <limits>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tir {
// (var, axis)
using TIRVarAxis = std::pair<Var, int>;
// (buffer, axis)
using BufferAxis = std::pair<Buffer, int>;
class BufferAxisHash {
 public:
  size_t operator()(const BufferAxis& buffer_axis) const {
    size_t const h1(ObjectPtrHash()(buffer_axis.first));
    size_t const h2(std::hash<int>()(buffer_axis.second));
    return h1 ^ (h2 << 1);
  }
};
/*!
 * \brief Suppose we want to shard a buffer along a specific dimension, we need to know how
 * to rewrite the access index of the buffer. To make it simple, we only support the case that
 * the access can be rewritten by changing the extent of an iter var.
 * \param index The access index
 * \param var_range The range of each iter var
 * \param analyzer The analyzer
 * \return The iter var whose extent to be changed
 */
Var GetShardingVarFromIndex(PrimExpr index, Map<Var, Range> var_range, arith::Analyzer* analyzer);

/*!
 * \brief Construct an axis group graph from a PrimFunc. Two buffer axis are connected if they
 * are accessed by the same index.
 */
class BufferAxisGraphExtractor : public StmtExprVisitor {
 public:
  static std::vector<std::vector<TIRVarAxis>> GetTIRVarAxisGraph(const PrimFunc& prim_func) {
    BufferAxisGraphExtractor extractor;
    extractor(prim_func->body);
    Map<Buffer, Var> inverse_buffer_map;
    for (const auto& pr : prim_func->buffer_map) {
      inverse_buffer_map.Set(pr.second, pr.first);
    }
    std::vector<std::vector<TIRVarAxis>> tir_var_axis_group_list;
    std::unordered_set<BufferAxis, BufferAxisHash> visited;
    for (const auto& pr : prim_func->buffer_map) {
      Var param = pr.first;
      Buffer buffer = pr.second;
      for (int i = 0; i < static_cast<int>(buffer->shape.size()); i++) {
        if (extractor.buffer_axis_graph_.count({buffer, i})) {
          std::vector<BufferAxis> buffer_axis_group;
          extractor.DFSGraph({buffer, i}, &visited, &buffer_axis_group);
          if (buffer_axis_group.size() <= 1) {
            continue;
          }
          std::vector<TIRVarAxis> tir_var_axis_group;
          for (const auto& buffer_axis : buffer_axis_group) {
            if (!inverse_buffer_map.count(buffer_axis.first)) {
              continue;
            }
            tir_var_axis_group.push_back(
                {inverse_buffer_map[buffer_axis.first], buffer_axis.second});
          }
          tir_var_axis_group_list.push_back(tir_var_axis_group);
        }
      }
    }
    return tir_var_axis_group_list;
  }

  void DFSGraph(BufferAxis cur, std::unordered_set<BufferAxis, BufferAxisHash>* visited,
                std::vector<BufferAxis>* buffer_axis_group) {
    if (visited->count(cur)) {
      return;
    }
    visited->insert(cur);
    buffer_axis_group->push_back(cur);
    for (const auto& next : buffer_axis_graph_[cur]) {
      DFSGraph(next, visited, buffer_axis_group);
    }
  }

 private:
  void VisitStmt_(const BufferStoreNode* op) final {
    StmtExprVisitor::VisitStmt_(op);
    buffer_access_indices_.push_back({op->buffer, op->indices});
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    StmtExprVisitor::VisitExpr_(op);
    buffer_access_indices_.push_back({op->buffer, op->indices});
  }

  bool Match(PrimExpr a, PrimExpr buffer_shape_a, PrimExpr b, PrimExpr buffer_shape_b,
             arith::Analyzer* analyzer) {
    if (b.as<VarNode>()) {
      std::swap(a, b);
      std::swap(buffer_shape_a, buffer_shape_b);
    }
    if (!a.as<VarNode>()) {
      return false;
    }
    Var var = Downcast<Var>(a);
    analyzer->Bind(iter_var_range_);
    b = analyzer->Simplify(b);
    // index var `a` must access whole range of a specific buffer dimension
    arith::IntSet intset_b = arith::EvalSet(b, arith::AsIntSet(iter_var_range_));
    if (!analyzer->CanProveEqual(buffer_shape_a, iter_var_range_[var]->extent) ||
        !intset_b.MatchRange(Range::FromMinExtent(0, buffer_shape_b))) {
      return false;
    }
    Var matched_var = GetShardingVarFromIndex(b, iter_var_range_, analyzer);
    if (!matched_var.same_as(var)) {
      return false;
    }
    return true;
  }

  void VisitStmt_(const BlockNode* op) final {
    if (op->name_hint == "root") {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    buffer_access_indices_.clear();
    StmtExprVisitor::VisitStmt_(op);
    iter_var_range_.clear();
    for (const auto& iter_var : op->iter_vars) {
      iter_var_range_.Set(iter_var->var, iter_var->dom);
    }
    arith::Analyzer analyzer;
    for (const auto& access_pr : buffer_access_indices_) {
      Buffer buffer = access_pr.first;
      Array<PrimExpr> indices = access_pr.second;
      for (int i = 0; i < static_cast<int>(indices.size()); i++) {
        for (const auto& another_access_pr : buffer_access_indices_) {
          if (another_access_pr.first.same_as(buffer)) {
            continue;
          }
          Buffer another_buffer = another_access_pr.first;
          Array<PrimExpr> another_indices = another_access_pr.second;
          for (int j = 0; j < static_cast<int>(another_indices.size()); j++) {
            if (Match(indices[i], buffer->shape[i], another_indices[j], another_buffer->shape[j],
                      &analyzer)) {
              JoinBufferAxis({buffer, i}, {another_buffer, j});
            }
          }
        }
      }
    }
  }

  void JoinBufferAxis(BufferAxis axis1, BufferAxis axis2) {
    if (!buffer_axis_graph_.count(axis1)) {
      buffer_axis_graph_[axis1] = {};
    }
    if (!buffer_axis_graph_.count(axis2)) {
      buffer_axis_graph_[axis2] = {};
    }
    buffer_axis_graph_[axis1].push_back(axis2);
    buffer_axis_graph_[axis2].push_back(axis1);
  }

  std::vector<std::pair<Buffer, Array<PrimExpr>>> buffer_access_indices_;
  std::unordered_map<BufferAxis, std::vector<BufferAxis>, BufferAxisHash> buffer_axis_graph_;
  Map<Var, Range> iter_var_range_;
  std::string func_name;
};
}  // namespace tir
}  // namespace tvm

namespace tvm {
namespace relax {
namespace distributed {

/*! \brief tensor axis*/
struct Axis {
  const ExprNode* tensor;
  int dim = 0;
  int tuple_index = 0;

  Axis(const ExprNode* tensor, int dim, int tuple_index = 0)
      : tensor(tensor), dim(dim), tuple_index(tuple_index) {
    ICHECK(tensor->IsInstance<ConstantNode>() || tensor->IsInstance<VarNode>());
  }

  bool operator==(const Axis& other) const {
    return tensor == other.tensor && dim == other.dim && tuple_index == other.tuple_index;
  }
};

class AxisHash {
 public:
  size_t operator()(const Axis& axis) const {
    size_t const h1(std::hash<const ExprNode*>()(axis.tensor));
    size_t const h2(std::hash<int>()(axis.dim));
    size_t const h3(std::hash<int>()(axis.tuple_index));
    return h1 ^ (h2 << 1) ^ (h3 << 2);
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
void BuildAxisGraphCallTIR(const Var& output_var, const Call& call, const tir::PrimFunc& func,
                           distributed::AxisGroupGraph* axis_group_graph);

}  // namespace distributed
}  // namespace relax
}  // namespace tvm

#endif  // TVM_RELAX_DISTRIBUTED_AXIS_GROUP_GRAPH_H_
