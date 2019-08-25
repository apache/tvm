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
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/transform.h>

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {
namespace graph_partitioning {

/*!
 * \brief The checker that verifies if a Relay program is annotated correctly
 * for graph partitioning.
 */
class AnnotationChecker : public ExprVisitor {
 public:
  bool Check(const Expr& expr) {
    return true;
  }
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
  Expr VisitExpr_(const CallNode* call) final {
    auto op_node = call->op.as<OpNode>();

    // Use the default visitor to traverse the nodes that are not subgraph
    // nodes.
    if (op_node == nullptr || call->attrs.as<SubgraphAttrs>() == nullptr) {
      return ExprMutator::VisitExpr_(call);
    }

    if (GetRef<Op>(op_node) == Op::Get("annotation.subgraph_begin")) {
      auto input_expr = VisitExpr(call->args[0]);
      auto subgraph_attrs = call->attrs.as<SubgraphAttrs>();
      auto var = VarNode::make(subgraph_attrs->compiler + "_input" + std::to_string(var_count_++),
                               call->args[0]->checked_type());
      subgraph_args_.push_back({var, input_expr});
      return std::move(var);
    } else {
      CHECK(GetRef<Op>(op_node) == Op::Get("annotation.subgraph_end"));

      auto subgraph_attrs = call->attrs.as<SubgraphAttrs>();
      CHECK(subgraph_attrs);
      auto input = VisitExpr(call->args[0]);
      Array<Var> params;
      Array<Expr> args;

      for (auto pair : subgraph_args_) {
        params.push_back(pair.first);
        args.push_back(pair.second);
      }

      auto subgraph_func = FunctionNode::make(params, input, Type(), {}, Attrs());

      subgraph_func =
          FunctionSetAttr(subgraph_func, "func_name", tvm::ir::StringImm::make("Subtract"));
      subgraph_func = FunctionSetAttr(subgraph_func, "Primitive", tvm::Integer(1));
      subgraph_func = FunctionSetAttr(subgraph_func, "External",
                                      tvm::ir::StringImm::make(subgraph_attrs->compiler));
      subgraph_args_.clear();
      var_count_ = 0;
      return CallNode::make(subgraph_func, args);
    }
  }

  /*
   * \brief For cases like the following:
   *
   *       op1
   *        |
   *       end
   *        |
   *       op2
   *      /   \
   *     x     y
   *
   * where x and y could be inputs, e.g. vars and/or constants. Here, we should
   * group all nodes/expressions that are dominated by op2 in the same subgraph.
   */
  Expr VisitExpr_(const VarNode* vn) final {
    Expr var = GetRef<Var>(vn);
    return var;
  }

  Expr VisitExpr_(const ConstantNode* cn) final {
    Expr constant = GetRef<Constant>(cn);
    return constant;
  }

 private:
  int var_count_{0};
  std::vector<std::pair<Var, Expr> > subgraph_args_;
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

    void VisitExpr_(const CallNode* call) final {
      ExprVisitor::VisitExpr_(call);
    }

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
  runtime::TypedPackedFunc<Function(Function, Module, PassContext)> pass_func =
    [=](Function f, Module m, PassContext pc) {
    return Downcast<Function>(graph_partitioning::PartitionGraph(f));
  };
  auto partitioned = CreateFunctionPass(pass_func, 1, "PartitionGraph", {});
  return Sequential({partitioned, InferType()});
}

TVM_REGISTER_API("relay._transform.PartitionGraph")
.set_body_typed(transform::PartitionGraph);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
