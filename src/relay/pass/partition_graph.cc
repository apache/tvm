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
 * \file src/relay/pass/partition_graph.cc
 *
 * \brief Partition an input function into multiple functions according based
 * on the inserted annotation nodes (i.e. compiler_begin and compiler_end).
 * These nodes are used as boundaries to partition the Relay function into
 * multiple regions that can be offloaded to different accelerators/backends.
 *
 * Each of these paritioned functions, a.k.a subgraphs, will be viewed as
 * external functions, and they will use the provided compiler for codegen.
 */

#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr.h>
#include <tvm/ir/error.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace relay {
namespace partitioning {

// Cache compiler_begin and compiler_end annotation ops for equivalence check to
// reduce registry lookup overhead.
static const Op& compiler_begin_op = Op::Get("annotation.compiler_begin");
static const Op& compiler_end_op = Op::Get("annotation.compiler_end");

/*!
 * \brief The subgraph properties for partitioning.
 */
struct Subgraph {
  /*! \brief The subgraph ID. */
  int id;

  /*! \brief The input arguments of this subgraph. */
  std::vector<std::pair<Var, Expr>> args;

  /*! \brief Nodes in this subgraph. */
  std::unordered_set<Expr, ObjectHash, ObjectEqual> nodes;
};

/*!
 * \brief The checker that verifies if a Relay program is annotated correctly
 * for partitioning.
 */
class AnnotationChecker : public ExprVisitor {
 public:
  bool Check() {
    if (!found_start_ && !found_end_) {
      LOG(WARNING) << "No compiler annotation found";
    } else if (!found_start_) {
      LOG(ERROR) << "compiler_begin annotation is missing";
      return false;
    } else if (!found_end_) {
      LOG(ERROR) << "compiler_end annotation is missing";
      return false;
    }
    return true;
  }

  void VisitExpr_(const CallNode* call) final {
    auto op_node = call->op.as<OpNode>();
    if (op_node == nullptr || call->attrs.as<CompilerAttrs>() == nullptr) {
      return;
    } else if (call->op == compiler_begin_op) {
      found_start_ = true;
    } else if (call->op == compiler_end_op) {
      found_end_ = true;
    }
  }

 private:
  bool found_start_{false};
  bool found_end_{false};
};

/*! \brief This class partitions the expr labeled with begin and end annoations
 * into function containing multiple regions. Each region is labeled with
 * a compiler attribute so that it will be handled by any compilers that are not
 * in the TVM stack.
 *
 * TODO(@zhiics) This following algorithm is not adequate to handle all cases,
 * i.e. multiple `compiler_end` nodes.
 */
class Partitioner : public ExprMutator {
 public:
  explicit Partitioner(const IRModule& module) : module_(module) {}

  std::shared_ptr<Subgraph> GetSubgraph(const Expr node) {
    for (auto candidate : this->subgraphs_) {
      if (candidate->nodes.find(node) != candidate->nodes.end()) {
        return candidate;
      }
    }
    return nullptr;
  }

  void MergeSubgraph(std::shared_ptr<Subgraph> subgraph1,
                     std::shared_ptr<Subgraph> subgraph2) {
    if (subgraph1 == subgraph2) {
      return;
    }

    // Merge subgraph 2 to subgraph 1 and erase subgraph 2.
    subgraph1->nodes.insert(subgraph2->nodes.begin(), subgraph2->nodes.end());
    for (auto arg : subgraph2->args) {
      subgraph1->args.push_back(arg);
    }
    this->subgraphs_.erase(subgraph2);
  }

  void AddToSubgraph(std::shared_ptr<Subgraph> subgraph, const Expr expr) {
    auto subgraph2 = GetSubgraph(expr);
    if (subgraph2) {
      MergeSubgraph(subgraph, subgraph2);
    } else {
      subgraph->nodes.insert(expr);
    }
  }

  Expr VisitExpr_(const CallNode* call) final {
    auto op_node = call->op.as<OpNode>();

    if (op_node == nullptr || call->attrs.as<CompilerAttrs>() == nullptr) {
      // Propogate subgraph to arguments
      auto subgraph = GetSubgraph(GetRef<Call>(call));
      if (subgraph) {
        for (auto arg : call->args) {
          AddToSubgraph(subgraph, arg);
        }
      }
      return ExprMutator::VisitExpr_(call);
    } else if (call->op == compiler_begin_op) {
      // The annotation node is inserted on edge so it must have only one argument.
      CHECK_EQ(call->args.size(), 1U);

      // Traverse the rest graph.
      auto input_expr = VisitExpr(call->args[0]);

      // Replace the begin annotation with an external call input variable.
      auto compiler_attrs = call->attrs.as<CompilerAttrs>();
      // The type of the created variable is the same as the compiler_begin
      // node.
      auto var = VarNode::make(compiler_attrs->compiler + "_input" + std::to_string(var_id_++),
                               call->checked_type_);

      // Find the corresponding subgraph and add the argument.
      auto subgraph = GetSubgraph(GetRef<Call>(call));
      if (!subgraph) {
        throw Error(ErrorBuilder()
                    << "Cannot find the corresponding subgraph for start annotation:\n"
                    << AsText(GetRef<Call>(call), false));
      }
      subgraph->args.push_back({var, input_expr});
      return std::move(var);
    } else {
      CHECK_EQ(call->op, compiler_end_op);
      // The annotation node is inserted on edge so it must have only one argument.
      CHECK_EQ(call->args.size(), 1U);

      auto compiler_attrs = call->attrs.as<CompilerAttrs>();

      // Check if the argument already belongs to an existing subgraph
      auto subgraph = GetSubgraph(call->args[0]);
      if (!subgraph) {
        auto ret = this->subgraphs_.emplace(std::make_shared<Subgraph>());
        subgraph = *ret.first;
        subgraph->nodes.insert(call->args[0]);
        subgraph->id = this->subgraph_id_++;
      }
      subgraph->nodes.insert(GetRef<Call>(call));

      // Traverse subgraph inputs.
      auto input = VisitExpr(call->args[0]);
      Array<Var> params;
      Array<Expr> args;

      // The subgraph may be merged so we need to update it again.
      subgraph = GetSubgraph(GetRef<Call>(call));
      CHECK(subgraph);

      for (auto pair : subgraph->args) {
        params.push_back(pair.first);
        args.push_back(pair.second);
      }

      auto subgraph_func =
          FunctionNode::make(params, input, call->checked_type_, {}, Attrs());

      std::string name = compiler_attrs->compiler + "_" + std::to_string(subgraph->id);
      subgraph_func =
          FunctionSetAttr(subgraph_func, attr::kExternalSymbol, tir::StringImmNode::make(name));
      subgraph_func = FunctionSetAttr(subgraph_func, attr::kPrimitive, tvm::Integer(1));
      subgraph_func = FunctionSetAttr(subgraph_func, attr::kCompiler,
                                      tvm::tir::StringImmNode::make(compiler_attrs->compiler));
      subgraph_func = FunctionSetAttr(subgraph_func, attr::kInline, tvm::Integer(1));
      CHECK(!module_->ContainGlobalVar(name))
          << "Global function " << name << " already exists";
      // Create a global function and add it to the IRModule for the subgraph.
      // This way we lift the functions that should be handled by external
      // codegen to the module scope and rely on the pass manager to prevent relay
      // function level passes (i.e. simplify inference and fusion) optimizing it.
      GlobalVar glob_func(name);
      module_->Add(glob_func, subgraph_func);
      // The return type of callnode is the same as the type of the
      // compiler_end node.
      auto ret = CallNode::make(glob_func, args);
      ret->checked_type_ = call->checked_type_;
      return std::move(ret);
    }
  }

  Expr VisitExpr_(const TupleNode* op) final {
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

  Expr VisitExpr_(const TupleGetItemNode* g) final {
    auto subgraph = GetSubgraph(GetRef<TupleGetItem>(g));
    if (!subgraph) {
      return ExprMutator::VisitExpr_(g);
    } else {
      AddToSubgraph(subgraph, g->tuple);
      auto t = VisitExpr(g->tuple);
      return TupleGetItemNode::make(t, g->index);
    }
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    auto subgraph = GetSubgraph(GetRef<Function>(op));
    if (!subgraph) {
      return ExprMutator::VisitExpr_(op);
    } else {
      Array<Var> params;
      for (auto param : op->params) {
        AddToSubgraph(subgraph, param);
      }
      for (auto param : op->params) {
        Var new_param = Downcast<Var>(VisitExpr(param));
        params.push_back(new_param);
      }
      auto body = VisitExpr(op->body);
      return FunctionNode::make(params, body, op->ret_type, op->type_params, op->attrs);
    }
  }

  Expr VisitExpr_(const LetNode* op) final {
    auto subgraph = GetSubgraph(GetRef<Let>(op));
    if (!subgraph) {
      return ExprMutator::VisitExpr_(op);
    } else {
      AddToSubgraph(subgraph, op->var);
      AddToSubgraph(subgraph, op->value);
      AddToSubgraph(subgraph, op->body);
      Var var = Downcast<Var>(VisitExpr(op->var));
      auto value = VisitExpr(op->value);
      auto body = VisitExpr(op->body);

      return LetNode::make(var, value, body);
    }
  }

  Expr VisitExpr_(const IfNode* op) final {
    auto subgraph = GetSubgraph(GetRef<If>(op));
    if (!subgraph) {
      return ExprMutator::VisitExpr_(op);
    } else {
      AddToSubgraph(subgraph, op->cond);
      AddToSubgraph(subgraph, op->true_branch);
      AddToSubgraph(subgraph, op->false_branch);
      auto guard = VisitExpr(op->cond);
      auto true_b = VisitExpr(op->true_branch);
      auto false_b = VisitExpr(op->false_branch);
      return IfNode::make(guard, true_b, false_b);
    }
  }

  Expr VisitExpr_(const RefCreateNode* op) final {
    auto subgraph = GetSubgraph(GetRef<RefCreate>(op));
    if (!subgraph) {
      return ExprMutator::VisitExpr_(op);
    } else {
      AddToSubgraph(subgraph, op->value);
      Expr value = VisitExpr(op->value);
      return RefCreateNode::make(value);
    }
  }

  Expr VisitExpr_(const RefReadNode* op) final {
    auto subgraph = GetSubgraph(GetRef<RefRead>(op));
    if (!subgraph) {
      return ExprMutator::VisitExpr_(op);
    } else {
      AddToSubgraph(subgraph, op->ref);
      Expr ref = VisitExpr(op->ref);
      return RefReadNode::make(ref);
    }
  }

  Expr VisitExpr_(const RefWriteNode* op) final {
    auto subgraph = GetSubgraph(GetRef<RefWrite>(op));
    if (!subgraph) {
      return ExprMutator::VisitExpr_(op);
    } else {
      AddToSubgraph(subgraph, op->ref);
      Expr ref = VisitExpr(op->ref);
      Expr value = VisitExpr(op->value);
      return RefWriteNode::make(ref, value);
    }
  }

  IRModule Partition() {
    auto glob_funcs = module_->functions;
    for (const auto& pair : glob_funcs) {
      if (auto* fn = pair.second.as<FunctionNode>()) {
        auto func = GetRef<Function>(fn);
        func = FunctionNode::make(func->params,
                                  VisitExpr(func->body),
                                  func->ret_type,
                                  func->type_params,
                                  func->attrs);
        module_->Update(pair.first, func);
      }
    }
    return module_;
  }

 private:
  int var_id_{0};
  int subgraph_id_{0};
  std::unordered_set<std::shared_ptr<Subgraph>> subgraphs_;
  IRModule module_;
};

}  // namespace partitioning

namespace transform {

Pass PartitionGraph() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> part_func =
      [=](IRModule m, PassContext pc) {
        return partitioning::Partitioner(m).Partition();
      };
  auto partitioned = CreateModulePass(part_func, 0, "PartitionGraph", {});
  return Sequential({partitioned, InferType()});
}

TVM_REGISTER_GLOBAL("relay._transform.PartitionGraph")
.set_body_typed(transform::PartitionGraph);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
