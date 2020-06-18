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
 * \file src/relay/transforms/partition_graph.cc
 *
 * \brief Partition an input function into multiple functions according based
 * on the inserted annotation nodes (i.e. compiler_begin and compiler_end).
 * These nodes are used as boundaries to partition the Relay function into
 * multiple regions that can be offloaded to different accelerators/backends.
 *
 * Each of these paritioned functions, a.k.a regions, will be viewed as
 * external functions, and they will use the provided compiler for codegen.
 */

#include <tvm/ir/error.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/container.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../analysis/annotated_region_set.h"
#include "../backend/utils.h"

namespace tvm {
namespace relay {
namespace partitioning {

// Cache compiler_begin and compiler_end annotation ops for equivalence check to
// reduce registry lookup overhead.
static const Op& compiler_begin_op = Op::Get("annotation.compiler_begin");
static const Op& compiler_end_op = Op::Get("annotation.compiler_end");

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

/*! \brief This class partitions the expr labeled with begin and end annotations
 * into function containing multiple regions. Each region is labeled with
 * a compiler attribute so that it will be handled by any compilers that are not
 * in the TVM stack.
 *
 * Input : A Relay module that have functions with disjoint annotated regions
 *         using compiler_begin and compiler_end. There could be multiple
 * outputs.
 *
 * Output : A Relay module with global functions for such disjoint annotated
 * regions with calls inserted at the respective location
 *
 * Dependencies : AnnotatedRegionSet Utility class.
 *
 * Methodology :
 *      1) The AnnotatedRegionSet utility class is able to construct a collection
 *      of nodes that are bound by a given annotation -- here we use
 *      compiler_begin and compiler_end
 *      2) Initially, for each function in the module RegionSets are populated.
 *      3) Then, Vistor pass is traversed until a compiler_end node is encountered
 *         that belongs to a "region".
 *      4) When the first compiler_end of a given annotated region is found,
 *         a function is formed and inserted.
 *         a) if the region has multiple outputs, a Tuple node (capturing
 *            all outputs) is returned.
 *      5) Thereafter, if we encounter an another output of the same annotated
 *         region, it is important to note that the function is already formed.
 *         Therefore, it will lookup the function and add a TupleGetItemNode.
 *         a) We will use the location index of "rets" of each Region" of
 *         AnnotatedRegionSet as TupleGetItemNode index.
 *      6) Therefore, functions will be created for all annotated regions.
 *         The name for each global function is created using "Region" id and
 *         the compiler name.
 */

class Partitioner : public ExprMutator {
 public:
  explicit Partitioner(const IRModule& module) : module_(module) {
    for (auto f : module->functions) {
      GlobalVar f_var = f.first;
      BaseFunc f_func = f.second;

      // Creating regionset per function in the module
      auto region_set = AnnotatedRegionSet::Create(f_func, partitioning::compiler_begin_op,
                                                   partitioning::compiler_end_op);
      regions_sets_[region_set] = f_func;
    }
  }

  Expr VisitExpr_(const CallNode* call) final {
    auto op_node = call->op.as<OpNode>();
    if (op_node == nullptr || call->attrs.as<CompilerAttrs>() == nullptr) {
      return ExprMutator::VisitExpr_(call);
    } else if (call->op == compiler_begin_op) {
      // The annotation node is inserted on edge so it must have only one
      // argument.
      CHECK_EQ(call->args.size(), 1U);

      // Traverse the rest graph.
      Expr parent = call->args[0];
      auto input_expr = VisitExpr(parent);

      // Backtrace the parent to find the first ancestor node that is not a begin or end op
      while (const auto* parent_call = parent.as<CallNode>()) {
        if (parent_call->op == compiler_begin_op ||
            parent_call->op == compiler_end_op) {
          parent = parent_call->args[0];
        } else {
          break;
        }
      }

      AnnotatedRegion sg = GetRegion(GetRef<Call>(call));
      int index = GetArgIdx(sg, GetRef<Call>(call));
      CHECK_NE(index, -1);

      if (shared_output_.count(parent) && shared_output_[parent].count(sg)) {
        return shared_output_[parent][sg];
      } else {
        // The type of the created variable is the same as the compiler_begin
        // node.
        std::string target = call->attrs.as<CompilerAttrs>()->compiler;
        std::string varname =
            target + "_" + std::to_string(sg->GetID()) + "_i" + std::to_string(index);
        auto var = Var(varname, GetRef<Call>(call)->checked_type_);

        std::pair<Var, Expr> cand = std::make_pair(var, input_expr);

        if (std::find(region_args[sg].begin(), region_args[sg].end(), cand) ==
            region_args[sg].end()) {
          region_args[sg].push_back(cand);
        }
        shared_output_[parent][sg] = var;
        return std::move(var);
      }
    } else {
      CHECK_EQ(call->op, compiler_end_op);
      // The annotation node is inserted on edge so it must have only one
      // argument.
      CHECK_EQ(call->args.size(), 1U);

      AnnotatedRegion region = GetRegion(GetRef<Call>(call));

      // TODO(@manupa-arm) : need to use the parent function (to which region
      // belongs to) name/key for the funtions that are created
      BaseFunc f = GetFunc(GetRef<Call>(call));

      // Traverse subgraph inputs.
      auto input = VisitExpr(call->args[0]);
      CHECK(region.defined()) << "Region not defined for " << GetRef<Call>(call);
      // functions are created for each annotated regions,
      // when their first output is encountered.
      // If multiple outputs are there, a tuple node is inserted at the end.
      // region_function_calls is map that maintains
      // (each annotated regions) --> created function

      if (region_function_calls.find(region) == region_function_calls.end()) {
        // First time this region is encountered in the traversal.
        // Creating the function.
        CreateFunction(region, call);
      }
      // Retrieve this particular output of function.
      return GetFunctionOutput(region, GetRef<Call>(call));
    }
  }

  Expr VisitExpr_(const TupleNode* op) final {
    auto region = GetRegion(GetRef<Tuple>(op));
    if (!region.defined()) {
      return ExprMutator::VisitExpr_(op);
    } else {
      Array<Expr> fields;
      for (auto field : op->fields) {
        fields.push_back(VisitExpr(field));
      }
      return Tuple(fields);
    }
  }

  Expr VisitExpr_(const TupleGetItemNode* g) final {
    auto region = GetRegion(GetRef<TupleGetItem>(g));
    if (!region.defined()) {
      return ExprMutator::VisitExpr_(g);
    } else {
      auto t = VisitExpr(g->tuple);
      return TupleGetItem(t, g->index);
    }
  }

  Expr VisitExpr_(const FunctionNode* op) final {
    auto region = GetRegion(GetRef<Function>(op));
    if (!region.defined()) {
      return ExprMutator::VisitExpr_(op);
    } else {
      Array<Var> params;
      for (auto param : op->params) {
        Var new_param = Downcast<Var>(VisitExpr(param));
        params.push_back(new_param);
      }
      auto body = VisitExpr(op->body);
      return Function(params, body, op->ret_type, op->type_params, op->attrs);
    }
  }

  Expr VisitExpr_(const LetNode* op) final {
    auto region = GetRegion(GetRef<Let>(op));
    if (!region.defined()) {
      return ExprMutator::VisitExpr_(op);
    } else {
      Var var = Downcast<Var>(VisitExpr(op->var));
      auto value = VisitExpr(op->value);
      auto body = VisitExpr(op->body);
      return Let(var, value, body);
    }
  }

  Expr VisitExpr_(const IfNode* op) final {
    auto region = GetRegion(GetRef<If>(op));
    if (!region.defined()) {
      return ExprMutator::VisitExpr_(op);
    } else {
      auto guard = VisitExpr(op->cond);
      auto true_b = VisitExpr(op->true_branch);
      auto false_b = VisitExpr(op->false_branch);
      return If(guard, true_b, false_b);
    }
  }

  Expr VisitExpr_(const RefCreateNode* op) final {
    auto region = GetRegion(GetRef<RefCreate>(op));
    if (!region.defined()) {
      return ExprMutator::VisitExpr_(op);
    } else {
      Expr value = VisitExpr(op->value);
      return RefCreate(value);
    }
  }

  Expr VisitExpr_(const RefReadNode* op) final {
    auto region = GetRegion(GetRef<RefRead>(op));
    if (!region.defined()) {
      return ExprMutator::VisitExpr_(op);
    } else {
      Expr ref = VisitExpr(op->ref);
      return RefRead(ref);
    }
  }

  Expr VisitExpr_(const RefWriteNode* op) final {
    auto region = GetRegion(GetRef<RefWrite>(op));
    if (!region.defined()) {
      return ExprMutator::VisitExpr_(op);
    } else {
      Expr ref = VisitExpr(op->ref);
      Expr value = VisitExpr(op->value);
      return RefWrite(ref, value);
    }
  }

  IRModule Partition() {
    auto glob_funcs = module_->functions;
    for (const auto& pair : glob_funcs) {
      if (auto* fn = pair.second.as<FunctionNode>()) {
        auto func = GetRef<Function>(fn);
        func = Function(func->params, VisitExpr(func->body), func->ret_type, func->type_params,
                        func->attrs);
        module_->Update(pair.first, func);
      }
    }
    return module_;
  }

 private:
  /*!
   * \brief Get the region an expression belongs to
   * if its in a region.
   */
  AnnotatedRegion GetRegion(const Expr& e) {
    for (auto sg_set_it : regions_sets_) {
      auto sg_set = sg_set_it.first;
      AnnotatedRegion sg = sg_set->GetRegion(e);
      if (sg.defined()) {
        return sg;
      }
    }
    return AnnotatedRegion(nullptr);
  }

  /*!
   * \brief Get the function an expression belongs to
   * if its in a region.
   */
  BaseFunc GetFunc(const Expr& e) {
    for (auto sg_set_it : regions_sets_) {
      auto sg_set = sg_set_it.first;
      auto func = sg_set_it.second;

      AnnotatedRegion sg = sg_set->GetRegion(e);
      if (sg.defined()) {
        return func;
      }
    }
    return BaseFunc(nullptr);
  }

  /*!
   * \brief Get the index of the argument;
   * this is to be used as tuplegetitem idx
   */
  int GetArgIdx(AnnotatedRegion sg, const Expr& arg) {
    int idx = 0;
    for (auto arg_ : sg->GetInputs()) {
      if (arg == arg_) {
        return idx;
      }
      idx++;
    }
    return -1;
  }

  /*!
   * \brief This function is called first time that we encounter a compiler_end
   * node to create the function for the subgraph.
   */
  void CreateFunction(AnnotatedRegion region, const CallNode* call) {
    // Create fields which is a unique list of outputs. Also populate
    // region_return_indices_ map which maps parent of compiler_end node to
    // corresponding index in fields.
    Array<Expr> fields;
    int i = 0;
    for (auto ret : region->GetOutputs()) {
      auto ret_node = Downcast<Call>(ret)->args[0];
      // Don't duplicate outputs.
      if (!region_return_indices_.count(region) ||
          !region_return_indices_[region].count(ret_node)) {
        auto ret_expr = VisitExpr(ret_node);
        fields.push_back(ret_expr);
        region_return_indices_[region][ret_node] = i;
        i++;
      }
    }

    Array<Var> params;
    Array<Expr> param_expr;
    std::unordered_map<std::string, runtime::NDArray> params_bind;

    for (auto pair : region_args[region]) {
      params.push_back(pair.first);
      if (const auto* cn = pair.second.as<ConstantNode>()) {
        params_bind[pair.first->name_hint()] = cn->data;
      } else {
        param_expr.push_back(pair.second);
      }
    }

    Function global_region_func;
    if (fields.size() == 1) {
      // If there are only a single output; no need to add a tuple
      global_region_func =
          Function(params, fields[0], call->args[0]->checked_type_, {}, DictAttrs());
    } else {
      auto tuple = Tuple(fields);
      global_region_func = Function(params, tuple, tuple->checked_type_, {}, DictAttrs());
    }

    std::string target = call->attrs.as<CompilerAttrs>()->compiler;
    std::string name = target + "_" + std::to_string(region->GetID());

    global_region_func = WithAttr(std::move(global_region_func), tvm::attr::kGlobalSymbol,
                                  runtime::String(name));
    global_region_func =
        WithAttr(std::move(global_region_func), attr::kPrimitive, tvm::Integer(1));
    global_region_func = WithAttr(std::move(global_region_func), attr::kCompiler,
                                  tvm::runtime::String(target));
    global_region_func =
        WithAttr(std::move(global_region_func), attr::kInline, tvm::Integer(1));

    // Constant propagation
    if (!params_bind.empty()) {
      global_region_func = backend::BindParamsByName(global_region_func, params_bind);
    }

    std::string fname = name;
    CHECK(!module_->ContainGlobalVar(fname))
        << "Global function " << fname << " already exists";
    // Create a global function and add it to the IRModule for the region.
    // This way we lift the functions that should be handled by external
    // codegen to the module scope and rely on the pass manager to prevent
    // relay function level passes (i.e. simplify inference and fusion)
    // optimizing it.
    GlobalVar glob_func(fname);
    module_->Add(glob_func, global_region_func);

    // The return type of callnode is the same as the type of the
    // compiler_end node.
    auto ret = Call(glob_func, param_expr);
    region_function_calls[region] = ret;
  }

  /*!
   * \brief Get the return(output) of the function for compiler end node "end_arg".
   * This will return either a Call (for a function with a single output) or a
   * TupleGetItem (for a function with multiple outputs).
   */
  Expr GetFunctionOutput(AnnotatedRegion region, const Expr& end_arg) {
    Expr arg = Downcast<Call>(end_arg)->args[0];
    // Function has one output.
    if (region_return_indices_[region].size() == 1) {
      return region_function_calls[region];
    }
    // Function has multiple outputs.
    // Use already made TupleGetItem.
    if (region_return_tuplegetitem_.count(region) &&
        region_return_tuplegetitem_[region].count(arg)) {
      return region_return_tuplegetitem_[region][arg];
    }
    // Create new TupleGetItem.
    CHECK(region_return_indices_.count(region) &&
          region_return_indices_[region].count(arg));
    int index = region_return_indices_[region][arg];

    auto func_call = region_function_calls[region];
    auto tuple_get_item_ = TupleGetItem(func_call, index);
    tuple_get_item_->checked_type_ = arg->checked_type_;
    region_return_tuplegetitem_[region][arg] = tuple_get_item_;
    return std::move(tuple_get_item_);
  }

  /*!
   * \brief This map maintains the already created function calls.
   * This is required in the multi-output scenario, to link rest of the outputs
   * to call
   */
  std::unordered_map<AnnotatedRegion, Call, ObjectHash, ObjectEqual> region_function_calls;

  /*!
   * \brief This map maintains arguments (of region) visits through visitor
   * patterns. Those arguement var and expression will be used to when creating
   * the function.
   */
  std::unordered_map<AnnotatedRegion, std::vector<std::pair<Var, Expr>>, ObjectHash, ObjectEqual>
      region_args;

  /*!
   * \brief This map maintains the index of an output in the subgraph function
   * for a given region. If there are multiple entries for a region, then the
   * function has a tuple of multiple outputs for its return.
   */
  using RegionRetIndexMap = std::unordered_map<Expr, int, ObjectHash, ObjectEqual>;
  std::unordered_map<AnnotatedRegion, RegionRetIndexMap, ObjectHash, ObjectEqual>
      region_return_indices_;

  /*!
   * \brief This map holds already created TupleGetItem nodes for accessing
   * outputs of a function.
   */
  using RegionRetTupleGetItemMap = std::unordered_map<Expr, TupleGetItem, ObjectHash, ObjectEqual>;
  std::unordered_map<AnnotatedRegion, RegionRetTupleGetItemMap, ObjectHash, ObjectEqual>
      region_return_tuplegetitem_;

  /*!
   * \brief Each region set is associated with a function in the module.
   * This map maintains the mapping between regionsets and the function it
   * belongs to
   */
  std::unordered_map<AnnotatedRegionSet, BaseFunc, ObjectHash, ObjectEqual> regions_sets_;

  /*!\brief Cache the output that is shared by different nodes. */
  using RegionOutputMap = std::unordered_map<AnnotatedRegion, Var, ObjectHash, ObjectEqual>;
  std::unordered_map<Expr, RegionOutputMap, ObjectHash, ObjectEqual> shared_output_;

  /*!\brief The IRModule used for partitioning. */
  IRModule module_;
};

class DefaultRemover : public ExprMutator {
 public:
  explicit DefaultRemover(const IRModule& module) : module_(module) {}

  IRModule Remove() {
    auto glob_funcs = module_->functions;
    for (const auto& pair : glob_funcs) {
      if (auto* fn = pair.second.as<FunctionNode>()) {
        auto func = GetRef<Function>(fn);
        func = Function(func->params, VisitExpr(func->body), func->ret_type, func->type_params,
                        func->attrs);
        module_->Update(pair.first, func);
      }
    }
    return module_;
  }

  Expr VisitExpr_(const CallNode* call) final {
    auto attrs = call->attrs.as<CompilerAttrs>();
    if (attrs != nullptr && attrs->compiler == "default") {
      return VisitExpr(call->args[0]);
    }
    return ExprMutator::VisitExpr_(call);
  }

 private:
  IRModule module_;
};

}  // namespace partitioning

namespace transform {

Pass PartitionGraph() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> part_func =
      [=](IRModule m, PassContext pc) {
        // TODO(@comaniac, @zhiics): We should also handle the annotation with "default" attribute
        // by treating them as un-annotated, but we don't have it yet. This workaround pass removes
        // all "default" annotations and should be deleted in the future.
        auto new_m = partitioning::DefaultRemover(m).Remove();
        return partitioning::Partitioner(new_m).Partition();
  };
  auto partitioned = CreateModulePass(part_func, 0, "PartitionGraph", {});
  return Sequential({partitioned, InferType()});
}

TVM_REGISTER_GLOBAL("relay._transform.PartitionGraph").set_body_typed(transform::PartitionGraph);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
