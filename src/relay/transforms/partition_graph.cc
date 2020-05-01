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

/*! \brief This struct maintains the required metadata for a region to generate a corresponding
 * global function and function call. Global function will be passed to the target specific codegen
 * and function call will be used in the transform Relay graph to invoke the function in runtime.
 */
struct RegionFuncMetadata {
  /*! \brief The call node of the generated global function for this region. */
  Call func_call;

  /*! \brief A list of argument pairs. Each pair includes (var, expr). var is used
   * as a function node argument; input expression is used as a function call parameter.
   */
  std::vector<std::pair<Var, Expr>> args;

  /*! \brief Map from each region output expr (compiler end) node to
   * the corresponding function output expr.
   */
  std::unordered_map<Expr, Expr, ObjectHash, ObjectEqual> region_func_out;

  /*! \brief Map from each region input expression (compiler begin) to
   * the corresponding function input variable. This cache is used to make sure
   * a region function will not have duplicated inputs even if it refers to
   * the same expr multiple times.
   */
  std::unordered_map<Expr, Var, ObjectHash, ObjectEqual> region_func_in;
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

class Partitioner : public MixedModeMutator {
 public:
  explicit Partitioner(const IRModule& module) : module_(module) {
    for (auto f : module->functions) {
      GlobalVar f_var = f.first;
      BaseFunc f_func = f.second;

      // Creating regionset per function in the module.
      auto region_set = AnnotatedRegionSet::Create(f_func, partitioning::compiler_begin_op,
                                                   partitioning::compiler_end_op);
      regions_sets_[region_set] = f_func;
    }
  }

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    auto op_node = call->op.as<OpNode>();
    if (op_node == nullptr || call->attrs.as<CompilerAttrs>() == nullptr) {
      return post;
    } else if (call->op == compiler_begin_op) {
      // The annotation node is inserted on edge so it must have only one argument.
      CHECK_EQ(call->args.size(), 1U);

      // Traverse the rest graph.
      Expr parent = call->args[0];
      auto input_expr = Downcast<Call>(post)->args[0];

      // Backtrace the parent to find the first ancestor node that is not a begin or end op
      while (const auto* parent_call = parent.as<CallNode>()) {
        if (parent_call->op == compiler_begin_op || parent_call->op == compiler_end_op) {
          parent = parent_call->args[0];
        } else {
          break;
        }
      }

      AnnotatedRegion sg = GetRegion(GetRef<Call>(call));
      int index = GetArgIdx(sg, GetRef<Call>(call));
      CHECK_NE(index, -1);

      if (region_func_meta_[sg].region_func_in.count(parent)) {
        return region_func_meta_[sg].region_func_in[parent];
      } else {
        // The type of the created variable is the same as the compiler_begin
        // node.
        std::string target = call->attrs.as<CompilerAttrs>()->compiler;
        std::string varname =
            target + "_" + std::to_string(sg->GetID()) + "_i" + std::to_string(index);
        auto var = Var(varname, GetRef<Call>(call)->checked_type_);

        std::pair<Var, Expr> cand = std::make_pair(var, input_expr);

        if (std::find(region_func_meta_[sg].args.begin(), region_func_meta_[sg].args.end(), cand) ==
            region_func_meta_[sg].args.end()) {
          region_func_meta_[sg].args.push_back(cand);
        }
        region_func_meta_[sg].region_func_in[parent] = var;
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
      auto input = Downcast<Call>(post)->args[0];
      CHECK(region.defined()) << "Region not defined for " << GetRef<Call>(call);
      // functions are created for each annotated regions,
      // when their first output is encountered.
      // If multiple outputs are there, a tuple node is inserted at the end.

      if (!region_func_meta_[region].func_call.defined()) {
        // First time this region is encountered in the traversal. Creating the function.
        CreateFunction(region, call);
      }

      // Retrieve this particular output of function.
      Expr region_out_expr = Downcast<Call>(GetRef<Call>(call))->args[0];
      CHECK(region_func_meta_[region].region_func_out.count(region_out_expr));
      return region_func_meta_[region].region_func_out[region_out_expr];
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
   * \brief Create a function and its function call for the given region. If the function has
   * multiple outputs, a Tuple will be formed to aggregate all outputs, and TupleGetItem nodes
   * will be created to serve output consumers.
   */
  void CreateFunction(AnnotatedRegion region, const CallNode* end_node) {
    // Create fields which is a unique list of outputs.
    Array<Expr> fields;
    std::unordered_map<Expr, int, ObjectHash, ObjectEqual> out_expr_to_idx;
    int out_idx = 0;
    for (auto region_end_node : region->GetOutputs()) {
      auto ret_node = Downcast<Call>(region_end_node)->args[0];
      // Don't duplicate outputs.
      if (!out_expr_to_idx.count(ret_node)) {
        auto ret_expr = MixedModeMutator::VisitExpr(ret_node);
        fields.push_back(ret_expr);
        out_expr_to_idx[ret_node] = out_idx++;
      }
    }

    Array<Var> params;
    Array<Expr> param_expr;
    Map<Var, Expr> params_bind;

    auto IsConstant = [](const Expr& expr) {
      if (expr->IsInstance<ConstantNode>()) return true;
      if (!expr->IsInstance<TupleNode>()) return false;
      const auto* tn = expr.as<TupleNode>();
      return std::all_of(tn->fields.begin(), tn->fields.end(),
                         [](const Expr& e) { return e->IsInstance<ConstantNode>(); });
    };

    for (auto pair : region_func_meta_[region].args) {
      params.push_back(pair.first);
      if (IsConstant(pair.second)) {
        params_bind.Set(pair.first, pair.second);
      } else {
        param_expr.push_back(pair.second);
      }
    }

    Function global_region_func;
    if (fields.size() == 1) {
      // If there are only a single output; no need to add a tuple
      global_region_func =
          Function(params, fields[0], end_node->args[0]->checked_type_, {}, DictAttrs());
    } else {
      auto tuple = Tuple(fields);
      global_region_func = Function(params, tuple, tuple->checked_type_, {}, DictAttrs());
    }

    std::string target = end_node->attrs.as<CompilerAttrs>()->compiler;
    std::string name = target + "_" + std::to_string(region->GetID());

    global_region_func =
        WithAttr(std::move(global_region_func), tvm::attr::kGlobalSymbol, runtime::String(name));
    global_region_func = WithAttr(std::move(global_region_func), attr::kPrimitive, tvm::Integer(1));
    global_region_func =
        WithAttr(std::move(global_region_func), attr::kCompiler, tvm::runtime::String(target));
    global_region_func = WithAttr(std::move(global_region_func), attr::kInline, tvm::Integer(1));

    // Constant propagation
    if (!params_bind.empty()) {
      global_region_func = Downcast<Function>(relay::Bind(global_region_func, params_bind));
    }

    std::string fname = name;
    CHECK(!module_->ContainGlobalVar(fname)) << "Global function " << fname << " already exists";
    // Create a global function and add it to the IRModule for the region.
    // This way we lift the functions that should be handled by external
    // codegen to the module scope and rely on the pass manager to prevent
    // relay function level passes (i.e. simplify inference and fusion)
    // optimizing it.
    GlobalVar glob_func(fname);
    module_->Add(glob_func, global_region_func);

    // Create a call node for the function.
    auto call = Call(glob_func, param_expr);
    region_func_meta_[region].func_call = call;

    // Create output expr(s) for the function call.
    if (out_expr_to_idx.size() == 1) {
      // Single output direcly uses the call node as the output expr.
      region_func_meta_[region].region_func_out[out_expr_to_idx.begin()->first] = call;
    } else {
      // Multiple outptus need to create TupleGetItem nodes as output exprs.
      for (auto pair : out_expr_to_idx) {
        Expr region_out_expr = pair.first;  // The arg of a compiler end node of this region.
        int idx = pair.second;              // Corresponding function output tuple index.
        auto tuple_get_item = TupleGetItem(call, idx);
        tuple_get_item->checked_type_ = region_out_expr->checked_type_;
        region_func_meta_[region].region_func_out[region_out_expr] = tuple_get_item;
      }
    }
  }

  /*! \brief Map from each region to its metadata of the generated function. */
  std::unordered_map<AnnotatedRegion, RegionFuncMetadata, ObjectHash, ObjectEqual>
      region_func_meta_;

  /*! \brief Each region set is associated with a function in the module.
   * This map maintains the mapping between regionsets and the function it
   * belongs to
   */
  std::unordered_map<AnnotatedRegionSet, BaseFunc, ObjectHash, ObjectEqual> regions_sets_;

  /*!\brief The IRModule used for partitioning. */
  IRModule module_;
};

IRModule RemoveDefaultAnnotations(IRModule module) {
  class DefaultRemover : public ExprRewriter {
   public:
    DefaultRemover() = default;

    Expr Rewrite_(const CallNode* call, const Expr& post) final {
      auto attrs = call->attrs.as<CompilerAttrs>();
      if (attrs != nullptr && attrs->compiler == "default") {
        return Downcast<Call>(post)->args[0];
      }
      return post;
    }
  };

  auto glob_funcs = module->functions;
  // module is mutable, hence, we make a copy of it.
  module.CopyOnWrite();
  for (const auto& pair : glob_funcs) {
    if (auto* fn = pair.second.as<FunctionNode>()) {
      auto func = GetRef<Function>(fn);
      DefaultRemover remover;
      auto removed = PostOrderRewrite(func->body, &remover);
      func = Function(func->params, removed, func->ret_type, func->type_params, func->attrs);
      module->Update(pair.first, func);
    }
  }
  return module;
}

}  // namespace partitioning

namespace transform {

Pass PartitionGraph() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> part_func = [=](IRModule m,
                                                                            PassContext pc) {
    // TODO(@comaniac, @zhiics): We should also handle the annotation with "default" attribute
    // by treating them as un-annotated, but we don't have it yet. This workaround pass removes
    // all "default" annotations and should be deleted in the future.
    auto new_m = partitioning::RemoveDefaultAnnotations(m);
    return partitioning::Partitioner(new_m).Partition();
  };
  auto partitioned = CreateModulePass(part_func, 0, "PartitionGraph", {});
  return Sequential({partitioned, InferType()});
}

TVM_REGISTER_GLOBAL("relay._transform.PartitionGraph").set_body_typed(transform::PartitionGraph);

}  // namespace transform

}  // namespace relay
}  // namespace tvm
