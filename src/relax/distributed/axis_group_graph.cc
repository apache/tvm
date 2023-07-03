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
#include <tvm/relax/analysis.h>
#include <tvm/relax/attrs/distributed.h>
#include <tvm/relax/attrs/linear_algebra.h>
#include <tvm/relax/attrs/manipulate.h>
#include <tvm/relax/attrs/nn.h>
#include <tvm/relax/attrs/statistical.h>
#include <tvm/relax/distributed/axis_group_graph.h>
#include <tvm/relax/expr.h>
#include <tvm/tir/stmt_functor.h>

#include <numeric>

namespace tvm {

namespace tir {

class BufferAxisGraphExtractor : public StmtExprVisitor {
 public:
  using TIRVarAxis = std::pair<Var, int>;
  using BufferAxis = std::pair<Buffer, int>;
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

 private:
  class BufferAxisHash {
   public:
    size_t operator()(const BufferAxis& buffer_axis) const {
      size_t const h1(ObjectPtrHash()(buffer_axis.first));
      size_t const h2(std::hash<int>()(buffer_axis.second));
      return h1 ^ (h2 << 1);
    }
  };

  void VisitStmt_(const BufferStoreNode* op) final {
    StmtExprVisitor::VisitStmt_(op);
    buffer_access_indices_.push_back({op->buffer, op->indices});
  }

  void VisitExpr_(const BufferLoadNode* op) final {
    StmtExprVisitor::VisitExpr_(op);
    buffer_access_indices_.push_back({op->buffer, op->indices});
  }

  void VisitStmt_(const BlockNode* op) final {
    if (op->name_hint == "root") {
      StmtExprVisitor::VisitStmt_(op);
      return;
    }
    buffer_access_indices_.clear();
    StmtExprVisitor::VisitStmt_(op);
    std::unordered_set<BufferAxis, BufferAxisHash> mapped_axis_set;
    arith::Analyzer analyzer;
    for (const auto& access_pr : buffer_access_indices_) {
      Buffer buffer = access_pr.first;
      Array<PrimExpr> indices = access_pr.second;
      for (int i = 0; i < static_cast<int>(indices.size()); i++) {
        if (mapped_axis_set.count({buffer, i})) {
          continue;
        }
        mapped_axis_set.insert({buffer, i});
        for (const auto& another_access_pr : buffer_access_indices_) {
          if (another_access_pr.first.same_as(buffer)) {
            continue;
          }
          Buffer another_buffer = another_access_pr.first;
          Array<PrimExpr> another_indices = another_access_pr.second;
          for (int j = 0; j < static_cast<int>(another_indices.size()); j++) {
            if (mapped_axis_set.count({another_buffer, j})) {
              continue;
            }
            if (analyzer.CanProveEqual(indices[i], another_indices[j])) {
              mapped_axis_set.insert({another_buffer, j});
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

  std::vector<std::pair<Buffer, Array<PrimExpr>>> buffer_access_indices_;
  std::unordered_map<BufferAxis, std::vector<BufferAxis>, BufferAxisHash> buffer_axis_graph_;
};

}  // namespace tir

namespace relax {
namespace distributed {

const TensorStructInfoNode* GetTensorStructInfo(Var var) {
  const auto* tensor_sinfo = GetStructInfoAs<TensorStructInfoNode>(var);
  if (tensor_sinfo) {
    return tensor_sinfo;
  }
  const auto* dtensor_sinfo = GetStructInfoAs<DTensorStructInfoNode>(var);
  if (dtensor_sinfo) {
    return dtensor_sinfo->tensor_sinfo.get();
  }
  LOG(FATAL) << var->name_hint() << " must be either Tensor or DTensor";
  throw;
}

void UnaryOpHelper(Array<Var> var_list, distributed::AxisGroupGraph* axis_group_graph) {
  int n_dim = GetTensorStructInfo(var_list[0])->ndim;
  for (const auto& var : var_list) {
    ICHECK(GetTensorStructInfo(var)->ndim == n_dim);
  }
  for (int i = 0; i < n_dim; i++) {
    ICHECK(var_list.size() <= 2);
    for (int j = 0; j < static_cast<int>(var_list.size()) - 1; j++) {
      axis_group_graph->JoinAxis({var_list[j].get(), i}, {var_list[j + 1].get(), i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
    }
  }
}

void BuildAxisGraphUnary(const Var& output_var, const Call& call,
                         distributed::AxisGroupGraph* axis_group_graph) {
  Array<Var> var_list;  // vars in param and output
  if (call->args[0]->IsInstance<VarNode>()) {
    var_list.push_back(Downcast<Var>(call->args[0]));
  }
  var_list.push_back(output_var);
  UnaryOpHelper(var_list, axis_group_graph);
}

void BuildAxisGraphBinary(const Var& output_var, const Call& call,
                          distributed::AxisGroupGraph* axis_group_graph) {
  Array<Var> var_list;  // vars in param and output
  if (call->args[0]->IsInstance<VarNode>()) {
    var_list.push_back(Downcast<Var>(call->args[0]));
  }
  if (call->args[1]->IsInstance<VarNode>()) {
    var_list.push_back(Downcast<Var>(call->args[1]));
  }
  var_list.push_back(output_var);
  if (var_list.size() <= 2) {
    UnaryOpHelper(var_list, axis_group_graph);
    return;
  }
  const auto* x1_sinfo = GetTensorStructInfo(var_list[0]);
  const auto* x2_sinfo = GetTensorStructInfo(var_list[1]);
  int x1_ndim = x1_sinfo->ndim;
  int x2_ndim = x2_sinfo->ndim;
  const auto* x1_shape = x1_sinfo->shape.as<ShapeExprNode>();
  const auto* x2_shape = x2_sinfo->shape.as<ShapeExprNode>();
  ICHECK(x1_shape && x2_shape);
  arith::Analyzer analyzer;
  for (int i = 1; i <= std::min(x1_ndim, x2_ndim); ++i) {
    const PrimExpr& dim0 = x1_shape->values[x1_ndim - i];
    const PrimExpr& dim1 = x2_shape->values[x2_ndim - i];
    if (analyzer.CanProveEqual(dim0, dim1)) {
      // join batch dim
      axis_group_graph->JoinAxis({var_list[0].get(), x1_ndim - i},
                                 {var_list[2].get(), std::max(x1_ndim, x2_ndim) - i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
      axis_group_graph->JoinAxis({var_list[1].get(), x2_ndim - i},
                                 {var_list[2].get(), std::max(x1_ndim, x2_ndim) - i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
    } else if (analyzer.CanProveEqual(dim0, 1)) {
      axis_group_graph->JoinAxis({var_list[1].get(), x2_ndim - i},
                                 {var_list[2].get(), std::max(x1_ndim, x2_ndim) - i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
    } else if (analyzer.CanProveEqual(dim1, 1)) {
      axis_group_graph->JoinAxis({var_list[0].get(), x1_ndim - i},
                                 {var_list[2].get(), std::max(x1_ndim, x2_ndim) - i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
    } else {
      LOG(FATAL) << "Invalid broadcast, dim0: " << dim0 << ", dim1: " << dim1;
    }
  }
}

void BuildAxisGraphReduce(const Var& output_var, const Call& call,
                          distributed::AxisGroupGraph* axis_group_graph) {
  ICHECK(call->args[0]->IsInstance<VarNode>());
  Var input_var = Downcast<Var>(call->args[0]);
  Array<Integer> axes;
  bool keepdims;
  if (const auto* attrs = call->attrs.as<StatisticalAttrs>()) {
    if (attrs->axis.defined()) {
      axes = attrs->axis.value();
    }
    keepdims = attrs->keepdims;
  } else if (const auto* attrs = call->attrs.as<SoftmaxAttrs>()) {
    axes = {attrs->axis};
    keepdims = true;
  } else {
    LOG(FATAL) << "Unsupported reduce op: " << call->op;
  }

  int ndim = GetTensorStructInfo(input_var)->ndim;

  std::unordered_set<int> normalized_axes;
  for (const Integer& i : axes) {
    int val = i->value;
    ICHECK(val < ndim && val >= -ndim);
    if (val < 0) {
      val = ndim + val;
    }
    normalized_axes.insert(val);
  }
  if (keepdims) {
    for (int i = 0; i < ndim; i++) {
      if (!normalized_axes.count(i)) {
        axis_group_graph->JoinAxis({input_var.get(), i}, {output_var.get(), i},
                                   distributed::AxisGroupGraph::EdgeType::kDescend);
      }
    }
  } else {
    for (int i = 0, j = 0; i < ndim; i++) {
      if (!normalized_axes.count(i)) {
        axis_group_graph->JoinAxis({input_var.get(), i}, {output_var.get(), j},
                                   distributed::AxisGroupGraph::EdgeType::kDescend);
        j++;
      }
    }
  }
}

void BuildAxisGraphMatmul(const Var& output_var, const Call& call,
                          distributed::AxisGroupGraph* axis_group_graph) {
  Var x1 = Downcast<Var>(call->args[0]);
  Var x2 = Downcast<Var>(call->args[1]);
  Var x3 = output_var;
  const auto* x1_sinfo = GetTensorStructInfo(x1);
  const auto* x2_sinfo = GetTensorStructInfo(x2);
  int x1_ndim = x1_sinfo->ndim;
  int x2_ndim = x2_sinfo->ndim;
  ICHECK(x1_ndim > 0 && x2_ndim > 0);
  int x1_prepended = 0;
  int x2_appended = 0;
  if (x1_ndim == 1) {
    x1_ndim = 2;
    x1_prepended = 1;
  }
  if (x2_ndim == 1) {
    x2_ndim = 2;
    x2_appended = 1;
  }
  const auto* x1_shape = x1_sinfo->shape.as<ShapeExprNode>();
  const auto* x2_shape = x2_sinfo->shape.as<ShapeExprNode>();
  ICHECK(x1_shape && x2_shape);
  Array<PrimExpr> x1_shape_prefix{x1_shape->values.begin(),
                                  x1_shape->values.end() - 2 + x1_prepended};
  Array<PrimExpr> x2_shape_prefix{x2_shape->values.begin(),
                                  x2_shape->values.end() - 2 + x2_appended};

  int x1_prefix_ndim = x1_shape_prefix.size();
  int x2_prefix_ndim = x2_shape_prefix.size();
  arith::Analyzer analyzer;
  for (int i = 1; i <= std::min(x1_prefix_ndim, x2_prefix_ndim); ++i) {
    const PrimExpr& dim0 = x1_shape_prefix[x1_prefix_ndim - i];
    const PrimExpr& dim1 = x2_shape_prefix[x2_prefix_ndim - i];
    // join batch dim
    if (analyzer.CanProveEqual(dim0, dim1)) {
      axis_group_graph->JoinAxis({x1.get(), x1_prefix_ndim - i},
                                 {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim) - i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
      axis_group_graph->JoinAxis({x2.get(), x2_prefix_ndim - i},
                                 {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim) - i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
    } else if (analyzer.CanProveEqual(dim0, 1)) {
      axis_group_graph->JoinAxis({x2.get(), x2_prefix_ndim - i},
                                 {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim) - i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
    } else if (analyzer.CanProveEqual(dim1, 1)) {
      axis_group_graph->JoinAxis({x1.get(), x1_prefix_ndim - i},
                                 {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim) - i},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
    } else {
      LOG(FATAL) << "Cannot broadcast " << dim0 << " and " << dim1;
    }
  }
  // join reduction dim
  axis_group_graph->JoinAxis({x1.get(), x1_sinfo->ndim - 1}, {x2.get(), x2_ndim - 2},
                             distributed::AxisGroupGraph::EdgeType::kSimbling);
  // join lhs_spatial dim and rhs_spatial dim
  if (!x1_prepended) {
    axis_group_graph->JoinAxis({x1.get(), x1_ndim - 2},
                               {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim)},
                               distributed::AxisGroupGraph::EdgeType::kDescend);
    if (!x2_appended) {
      axis_group_graph->JoinAxis({x2.get(), x2_ndim - 1},
                                 {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim) + 1},
                                 distributed::AxisGroupGraph::EdgeType::kDescend);
    }
  } else if (!x2_appended) {
    axis_group_graph->JoinAxis({x2.get(), x2_ndim - 1},
                               {x3.get(), std::max(x1_prefix_ndim, x2_prefix_ndim)},
                               distributed::AxisGroupGraph::EdgeType::kDescend);
  }
}

void BuildAxisGraphPermuteDims(const Var& output_var, const Call& call,
                               distributed::AxisGroupGraph* axis_group_graph) {
  Var input_var = Downcast<Var>(call->args[0]);
  const auto* attrs = call->attrs.as<PermuteDimsAttrs>();
  ICHECK(attrs);
  int ndim = GetTensorStructInfo(input_var)->ndim;
  std::vector<int> normalized_axes;
  if (attrs->axes.defined()) {
    for (const Integer& i : attrs->axes.value()) {
      int val = i->value;
      ICHECK(val < ndim && val >= -ndim);
      if (val < 0) {
        val = ndim + val;
      }
      normalized_axes.push_back(val);
    }
  } else {
    normalized_axes.resize(ndim);
    std::iota(normalized_axes.rbegin(), normalized_axes.rend(), 0);
  }
  for (int i = 0; i < ndim; i++) {
    axis_group_graph->JoinAxis({input_var.get(), normalized_axes[i]}, {output_var.get(), i},
                               distributed::AxisGroupGraph::EdgeType::kDescend);
  }
}
void BuildAxisGraphReshape(const Var& output_var, const Call& call,
                           distributed::AxisGroupGraph* axis_group_graph) {
  Var input_var = Downcast<Var>(call->args[0]);
  const auto* tensor_sinfo = GetTensorStructInfo(input_var);
  const auto* new_shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(call->args[1]);
  const auto* old_shape_sinfo = GetStructInfoAs<ShapeStructInfoNode>(tensor_sinfo->shape.value());
  ICHECK_NOTNULL(old_shape_sinfo);
  Array<PrimExpr> old_shape_values = old_shape_sinfo->values.value();
  Array<PrimExpr> new_shape_values = new_shape_sinfo->values.value();
  int i = old_shape_values.size();
  int j = new_shape_values.size();
  PrimExpr old_shape_product = 1, new_shape_product = 1;
  arith::Analyzer analyzer_;
  while (i > 0 && j > 0) {
    if (analyzer_.CanProve(new_shape_product > old_shape_product)) {
      i--;
      old_shape_product *= old_shape_values[i];
    } else if (analyzer_.CanProve(new_shape_product < old_shape_product)) {
      j--;
      new_shape_product *= new_shape_values[j];
    } else {
      if (i != static_cast<int>(old_shape_values.size())) {
        axis_group_graph->JoinAxis({input_var.get(), i}, {output_var.get(), j},
                                   distributed::AxisGroupGraph::EdgeType::kDescend);
      }
      i--;
      j--;
      old_shape_product *= old_shape_values[i];
      new_shape_product *= new_shape_values[j];
    }
  }
}

void BuildAxisGraphCallTIR(const Var& output_var, const Call& call, const tir::PrimFunc& func,
                           distributed::AxisGroupGraph* axis_group_graph) {
  auto tir_var_axis_group_list = tir::BufferAxisGraphExtractor::GetTIRVarAxisGraph(func);
  Map<tir::Var, Var> tir_var_to_relax_var;
  Array<Expr> var_list = Downcast<Tuple>(call->args[1])->fields;
  var_list.push_back(output_var);
  for (int i = 0; i < static_cast<int>(var_list.size()); i++) {
    if (func->buffer_map.count(func->params[i])) {
      tir_var_to_relax_var.Set(func->params[i], Downcast<Var>(var_list[i]));
    }
  }
  for (const auto& var_axis_group : tir_var_axis_group_list) {
    int output_idx = -1;
    for (int i = 0; i < static_cast<int>(var_axis_group.size()); i++) {
      if (tir_var_to_relax_var[var_axis_group[i].first].same_as(output_var)) {
        output_idx = i;
        break;
      }
    }
    if (output_idx == -1) {
      for (int i = 1; i < static_cast<int>(var_axis_group.size()); i++) {
        axis_group_graph->JoinAxis(
            {tir_var_to_relax_var[var_axis_group[i].first].get(), var_axis_group[i].second},
            {tir_var_to_relax_var[var_axis_group[0].first].get(), var_axis_group[0].second},
            distributed::AxisGroupGraph::EdgeType::kSimbling);
      }
    } else {
      for (int i = 0; i < static_cast<int>(var_axis_group.size()); i++) {
        if (i != output_idx) {
          axis_group_graph->JoinAxis(
              {tir_var_to_relax_var[var_axis_group[i].first].get(), var_axis_group[i].second},
              {tir_var_to_relax_var[var_axis_group[output_idx].first].get(),
               var_axis_group[output_idx].second},
              distributed::AxisGroupGraph::EdgeType::kDescend);
        }
      }
    }
  }
}
}  // namespace distributed
}  // namespace relax
}  // namespace tvm
