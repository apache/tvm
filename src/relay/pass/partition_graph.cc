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

/*! Copyright (c) 2019 by Contributorsr
 * \file src/relay/pass/partition_graph.cc
 *
 * \brief  Partition an input function into multiple Functions according based
 * on the inserted annotation nodes (i.e. begin and end). These nodes are used
 * as boundaries to partition the Relay function into multiple regions that can
 * be offloaded to different accelerators.
 *
 * Each of these paritioned functions, a.k.a subgraphs, will be viewed as
 * external functions, and they will use external tools for codegen.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {
namespace graph_partitioning {

/*!
 * \brief The subgraph properties for partition.
 */
struct Subgraph {
  /*! \brief The subgraph ID. */
  int id;

  /*! \brief The input arguments of this subgraph. */
  std::vector<std::pair<Var, Expr>> args;

  /*! \brief Nodes in this subgraph. */
  std::unordered_set<Expr, ExprHash, ExprEqual> nodes;
};

/*!
 * \brief The checker that verifies if a Relay program is annotated correctly
 * for graph partitioning.
 */
class AnnotationChecker : public ExprVisitor {
 public:
  bool Check() {
    if (!this->found_start && !this->found_end) {
      LOG(WARNING) << "No subgraph annotation found";
    } else if (!this->found_start) {
      LOG(ERROR) << "Subgraph start annotation is missing";
      return false;
    } else if (!this->found_end) {
      LOG(ERROR) << "Subgraph end annotation is missing";
      return false;
    }
    return true;
  }

  void VisitExpr_(const CallNode* call) final {
    auto op_node = call->op.as<OpNode>();
    if (op_node == nullptr || call->attrs.as<SubgraphAttrs>() == nullptr) {
      return;
    } else if (GetRef<Op>(op_node) == Op::Get("annotation.subgraph_begin")) {
      this->found_start = true;
    } else if (GetRef<Op>(op_node) == Op::Get("annotation.subgraph_end")) {
      this->found_end = true;
    }
  }

 private:
  bool found_start = false;
  bool found_end = false;
};

/*! \brief This class partitions the graph labeled with begin and end annoations
 * into function containing multiple subgraphs. Each subgraph is labeled as
 * external.
 *
 * TODO(@zhiics) This following algorithm is not adequate to handle all cases,
 * i.e. multiple `end` nodes.
 */
class Partitioner : public ExprMutator {
 public:
  Subgraph* GetSubgraph(const Expr node) {
    for (auto candidate : this->subgraphs_) {
      if (candidate->nodes.find(node) != candidate->nodes.end()) {
        return candidate;
      }
    }
    return nullptr;
  }

  void MergeSubgraph(Subgraph* subgraph1, Subgraph* subgraph2) {
    // Merge subgraph 2 to subgraph 1 and erase subgraph 2.
    subgraph1->nodes.insert(subgraph2->nodes.begin(), subgraph2->nodes.end());
    for (auto arg : subgraph2->args) {
      subgraph1->args.push_back(arg);
    }
    this->subgraphs_.erase(subgraph2);
  }

  void AddToSubgraph(Subgraph* subgraph, const Expr expr) {
    auto subgraph2 = GetSubgraph(expr);
    if (subgraph2) {
      MergeSubgraph(subgraph, subgraph2);
    } else {
      subgraph->nodes.insert(expr);
    }
  }

  Expr VisitExpr_(const CallNode* call) final {
    auto op_node = call->op.as<OpNode>();

    if (op_node == nullptr || call->attrs.as<SubgraphAttrs>() == nullptr) {
      // Propogate subgraph to arguments
      auto subgraph = GetSubgraph(GetRef<Call>(call));
      if (subgraph) {
        for (auto arg : call->args) {
          AddToSubgraph(subgraph, arg);
        }
      }
      return ExprMutator::VisitExpr_(call);
    } else if (GetRef<Op>(op_node) == Op::Get("annotation.subgraph_begin")) {
      // The annotation node is inserted on edge so it must have only one argument.
      CHECK(call->args.size() == 1);

      // Traverse the rest graph.
      auto input_expr = VisitExpr(call->args[0]);

      // Replace the begin annotation with an external call input variable.
      // TODO: Confirm if it is safe to use checked_type_ instead of checked_type()
      auto subgraph_attrs = call->attrs.as<SubgraphAttrs>();
      auto var = VarNode::make(subgraph_attrs->compiler + "_input" + std::to_string(var_id_++),
                               input_expr->checked_type_);

      // Find the corresponding subgraph and add the argument.
      auto subgraph = GetSubgraph(GetRef<Call>(call));
      if (!subgraph) {
        throw Error(RELAY_ERROR("Cannot find the corresponding subgraph for end annotation:\n"
                                << AsText(GetRef<Call>(call), false)));
      }
      subgraph->args.push_back({var, input_expr});
      return std::move(var);
    } else {
      CHECK(GetRef<Op>(op_node) == Op::Get("annotation.subgraph_end"));
      // The annotation node is inserted on edge so it must have only one argument.
      CHECK(call->args.size() == 1);

      auto subgraph_attrs = call->attrs.as<SubgraphAttrs>();

      // Check if the argument is already belonged to an exist subgraph
      auto subgraph = GetSubgraph(call->args[0]);
      if (!subgraph) {
        auto ret = this->subgraphs_.emplace(new Subgraph());
        subgraph = *ret.first;
        subgraph->nodes.insert(call->args[0]);
        subgraph->id = this->subgraph_id_++;
      }
      subgraph->nodes.insert(GetRef<Call>(call));

      // Traverse towarding to subgraph inputs.
      auto input = VisitExpr(call->args[0]);
      Array<Var> params;
      Array<Expr> args;

      // The subgraph may be merged so we need to update it again.
      subgraph = GetSubgraph(GetRef<Call>(call));
      for (auto pair : subgraph->args) {
        params.push_back(pair.first);
        args.push_back(pair.second);
      }

      auto subgraph_func =
          FunctionNode::make(params, input, call->args[0]->checked_type_, {}, Attrs());

      // FIXME: How to determine the function name?
      // This is a hack for multiple subgraph test where each subgraph only has
      // one call node.
      // We can probably only pass "external" to indicate that this is an
      // external funciton and leave the processing of the function to codegen.
      // Otherwise, it's hard to deal with multiple-node subgraphs.
      Expr arg0 = call->args[0];
      std::string name = "Subgraph";
      if (const auto* arg_call = arg0.as<CallNode>()) {
        if (const auto* op_node = arg_call->op.as<OpNode>()) {
          name = op_node->name;
          name[0] = name[0] - 32;
        }
      }
      subgraph_func =
          FunctionSetAttr(subgraph_func, "func_name", tvm::ir::StringImm::make(name));
      subgraph_func = FunctionSetAttr(subgraph_func, "Primitive", tvm::Integer(1));
      subgraph_func = FunctionSetAttr(subgraph_func, "External",
                                      tvm::ir::StringImm::make(subgraph_attrs->compiler));
      return CallNode::make(subgraph_func, args);
    }
  }

  Expr VisitExpr_(const TupleNode* op) {
    auto subgraph = GetSubgraph(GetRef<Tuple>(op));
    if (!subgraph) {
      return ExprMutator::VisitExpr_(op);
    } else {
      for (auto field : op->fields) {
        AddToSubgraph(subgraph, field);
      }
      Array<Expr> fields;
      for (auto field : op->fields) {
        fields.push_back(VisitExpr(field));
      }
      return TupleNode::make(fields);
    }
  }

 private:
  int var_id_{0};
  int subgraph_id_{0};
  std::unordered_set<Subgraph*> subgraphs_;
};

/*!
 * \brief Combine parallel subgraphs that belong to the same codegen backend.
 *
 * For example, sg1 and sg2 should be combined if they belong to the same
 * codegen tool in the following case.
 *
 *      op1
 *     /   \
 *   sg1   sg2
 *
 *       |
 *      \|/
 *
 *      op1
 *       |
 *    sg1_sg2
 *
 * where the return type of the new subgraph sg1_sg2 is a tuple, and op1 has two
 * inputs that obtained from the tuple.
 */
class ParallelSubgraphCombiner : public ExprMutator {
  using ParallelGroup = std::vector<std::unordered_map<std::string, CallNode*>>;

 public:
  Expr Combine(const Expr& expr) {
    ParallelGroup groups = GroupFinder().FindGroups(expr);
    return expr;
  }

 private:
  class GroupFinder : public ExprVisitor {
   public:
    ParallelGroup FindGroups(const Expr& expr) {
      this->VisitExpr(expr);
      return groups_;
    }

    void VisitExpr_(const CallNode* call) final { ExprVisitor::VisitExpr_(call); }

   private:
    ParallelGroup groups_;
  };
};

Expr PartitionGraph(const Expr& expr) {
  Partitioner part;
  return part.Mutate(expr);
}

}  // namespace graph_partitioning

namespace transform {

Pass PartitionGraph() {
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> part_func =
      [=](Function f, Module m, PassContext pc) {
        return Downcast<Function>(graph_partitioning::PartitionGraph(f));
      };
  auto partitioned = CreateFunctionPass(part_func, 1, "PartitionGraph", {});
  return Sequential({partitioned, InferType()});
}

TVM_REGISTER_API("relay._transform.PartitionGraph").set_body_typed(transform::PartitionGraph);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
