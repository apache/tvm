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

/*!
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
#include <tvm/ir/module.h>
#include <tvm/relay/analysis.h>
#include <tvm/relay/attrs/annotation.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/name_transforms.h>

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../analysis/annotated_region_set.h"
#include "../backend/name_transforms.h"
#include "../backend/utils.h"
#include "pass_utils.h"

namespace tvm {
namespace relay {

namespace partitioning {

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
  std::unordered_map<Expr, Expr, ObjectPtrHash, ObjectPtrEqual> region_func_out;

  /*! \brief Map from each region input expression (compiler begin) to
   * the corresponding function input variable. This cache is used to make sure
   * a region function will not have duplicated inputs even if it refers to
   * the same expr multiple times.
   */
  std::unordered_map<Expr, Var, ObjectPtrHash, ObjectPtrEqual> region_func_in;
};

/*! \brief This class partitions the expr labeled with begin and end annotations
 * into function containing multiple regions. Each region is labeled with
 * a compiler attribute so that it will be handled by any compilers that are not
 * in the TVM stack.
 *
 * Input : A Relay module that has functions with disjoint annotated regions
 *         using compiler_begin and compiler_end. There could be multiple
 *         outputs.
 *
 * Output : A Relay module with global functions for such disjoint annotated
 *          regions with calls inserted at the respective location
 *
 * Dependencies : AnnotatedRegionSet Utility class.
 *
 * Methodology :
 *      1) The AnnotatedRegionSet utility class is able to construct a collection
 *         of nodes that are bound by a given annotation -- here we use
 *         compiler_begin and compiler_end
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
  Partitioner(const IRModule& module, bool bind_constants)
      : module_(module), bind_constants_(bind_constants) {
    std::set<std::string> func_names;
    for (auto f : module->functions) {
      GlobalVar f_var = f.first;
      BaseFunc f_func = f.second;
      std::string f_name = f_var.as<GlobalVarNode>()->name_hint;
      while (func_names.find(f_name) != func_names.end()) {
        f_name += "_a";
      }
      func_names.insert(f_name);

      // Creating regionset per function in the module.
      auto region_set =
          AnnotatedRegionSet::Create(f_func, CompilerBeginOp(), CompilerEndOp(), f_name);
      regions_sets_[region_set] = f_func;
    }
  }

  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    auto op_node = call->op.as<OpNode>();
    if (op_node == nullptr || call->attrs.as<CompilerAttrs>() == nullptr) {
      return post;
    } else if (call->op == CompilerBeginOp()) {
      // The annotation node is inserted on edge so it must have only one argument.
      ICHECK_EQ(call->args.size(), 1U);

      // Traverse the rest graph.
      Expr parent = call->args[0];
      auto input_expr = Downcast<Call>(post)->args[0];

      // Backtrace the parent to find the first ancestor node that is not a begin or end op
      while (const auto* parent_call = parent.as<CallNode>()) {
        if (parent_call->op == CompilerBeginOp() || parent_call->op == CompilerEndOp()) {
          parent = parent_call->args[0];
        } else {
          break;
        }
      }

      AnnotatedRegion sg = GetRegion(GetRef<Call>(call));
      int index = GetArgIdx(sg, GetRef<Call>(call));
      ICHECK_NE(index, -1);

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
      ICHECK_EQ(call->op, CompilerEndOp());
      // The annotation node is inserted on edge so it must have only one
      // argument.
      ICHECK_EQ(call->args.size(), 1U);

      AnnotatedRegion region = GetRegion(GetRef<Call>(call));

      // TODO(@manupa-arm) : need to use the parent function (to which region
      // belongs to) name/key for the functions that are created
      BaseFunc f = GetFunc(GetRef<Call>(call));

      // Traverse subgraph inputs.
      auto input = Downcast<Call>(post)->args[0];
      ICHECK(region.defined()) << "Region not defined for " << GetRef<Call>(call);
      // functions are created for each annotated regions,
      // when their first output is encountered.
      // If multiple outputs are there, a tuple node is inserted at the end.

      if (!region_func_meta_[region].func_call.defined()) {
        // First time this region is encountered in the traversal. Creating the function.
        CreateFunction(region, call);
      }

      // Retrieve this particular output of function.
      Expr region_out_expr = Downcast<Call>(GetRef<Call>(call))->args[0];
      ICHECK(region_func_meta_[region].region_func_out.count(region_out_expr));
      return region_func_meta_[region].region_func_out[region_out_expr];
    }
  }

  IRModule Partition() {
    auto glob_funcs = module_->functions;
    for (const auto& pair : glob_funcs) {
      if (auto* fn = pair.second.as<FunctionNode>()) {
        Function func = GetRef<Function>(fn);
        func = WithFields(func, func->params, VisitExpr(func->body));
        module_->Update(pair.first, func);
        module_ = transform::InferType()(module_);
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
   * \brief Check if an expr is a constant or a tuple that only contain constants.
   */
  bool IsConstant(const Expr& expr) const {
    if (expr->IsInstance<ConstantNode>()) return true;
    if (!expr->IsInstance<TupleNode>()) return false;
    const auto* tn = expr.as<TupleNode>();
    return std::all_of(tn->fields.begin(), tn->fields.end(),
                       [](const Expr& e) { return e->IsInstance<ConstantNode>(); });
  }

  /*!
   * \brief Create a call to the function that represents a region.
   * \note The customized optimization pipeline will be invoked as well to
   *       optimize each function that is handled by external codegen.
   */
  Call CreateRegionCall(AnnotatedRegion region, const Array<Expr>& fields,
                        const CallNode* end_node) {
    Array<Var> params;
    Array<Expr> param_expr;
    Map<Var, Expr> params_bind;
    for (auto pair : region_func_meta_[region].args) {
      params.push_back(pair.first);
      if (bind_constants_ && IsConstant(pair.second)) {
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
    std::string name = target + "_" + region->GetName() + "_" + std::to_string(region->GetID());

    // Constant propagation
    if (!params_bind.empty()) {
      global_region_func = Downcast<Function>(relay::Bind(global_region_func, params_bind));
    }
    std::string ext_opt = "relay.ext." + target + ".optimize";
    auto pf = tvm::runtime::Registry::Get(ext_opt);
    if (pf != nullptr) {
      auto mod = IRModule::FromExpr(global_region_func);
      mod = transform::InferType()(mod);
      mod = (*pf)(mod);
      global_region_func = Downcast<Function>(mod->Lookup("main"));
    }

    global_region_func =
        WithAttr(std::move(global_region_func), tvm::attr::kGlobalSymbol, runtime::String(name));
    global_region_func = WithAttr(std::move(global_region_func), attr::kPrimitive, tvm::Integer(1));
    global_region_func =
        WithAttr(std::move(global_region_func), attr::kCompiler, tvm::runtime::String(target));
    global_region_func = WithAttr(std::move(global_region_func), attr::kInline, tvm::Integer(1));

    GlobalVarSupply global_var_supply = GlobalVarSupply(module_);
    GlobalVar glob_func = global_var_supply->FreshGlobal(name, false);
    ICHECK(!module_->ContainGlobalVar(glob_func->name_hint))
        << "Global function " << glob_func->name_hint << " already exists";
    // Create a global function and add it to the IRModule for the region.
    // This way we lift the functions that should be handled by external
    // codegen to the module scope and rely on the pass manager to prevent
    // relay function level passes (i.e. simplify inference and fusion)
    // optimizing it.
    module_->Add(glob_func, global_region_func);
    module_ = relay::transform::InferType()(module_);

    // Create a call node for the function.
    auto call = Call(glob_func, param_expr);
    region_func_meta_[region].func_call = call;

    return call;
  }

  /*!
   * \brief Create a function and its function call for the given region. If the function has
   * multiple outputs, a Tuple will be formed to aggregate all outputs, and TupleGetItem nodes
   * will be created to serve output consumers.
   */
  void CreateFunction(AnnotatedRegion region, const CallNode* end_node) {
    // Create fields which is a unique list of outputs.
    Array<Expr> fields;
    std::unordered_map<Expr, int, ObjectPtrHash, ObjectPtrEqual> out_expr_to_idx;
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

    Call call = CreateRegionCall(region, fields, end_node);

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
  std::unordered_map<AnnotatedRegion, RegionFuncMetadata, ObjectPtrHash, ObjectPtrEqual>
      region_func_meta_;

  /*! \brief Each region set is associated with a function in the module.
   * This map maintains the mapping between regionsets and the function it
   * belongs to
   */
  std::unordered_map<AnnotatedRegionSet, BaseFunc, ObjectPtrHash, ObjectPtrEqual> regions_sets_;

  /*!\brief The IRModule used for partitioning. */
  IRModule module_;

  /*!\brief Whether or not to bind constants in partitioned subgraphs. */
  bool bind_constants_{false};
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
      func = WithFields(func, func->params, removed);
      module->Update(pair.first, func);
      module = relay::transform::InferType()(module);
    }
  }
  return module;
}

/*! \brief There can be regions with multiple outputs where each output
 *  could be a tuple output. Such tuple outputs needs to be flattened
 *  otherwise the function would create tuples of tuples. Moreover, tuple
 *  of tuples are valid relay, however they are not currently supported by
 *  graph executor or relay VM.
 */

// New annotations would be required to be added for each flattened output
static const PackedFunc* make_end_op =
    runtime::Registry::Get("relay.op.annotation._make.compiler_end");

IRModule FlattenTupleOutputs(IRModule module) {
  class TupleOutFlattener : public ExprRewriter {
   public:
    TupleOutFlattener() = default;

    Expr Rewrite_(const CallNode* call, const Expr& post) final {
      if (call->op == CompilerEndOp()) {
        std::string target = call->attrs.as<CompilerAttrs>()->compiler;
        // Arguments of annotation ops should be 1
        ICHECK_EQ(call->args.size(), 1U);
        auto annotated_op = Downcast<Call>(post)->args[0];
        if (const auto* tuple_node = annotated_op.as<TupleNode>()) {
          Array<Expr> new_fields;
          new_fields.reserve(tuple_node->fields.size());

          // Here each input of the tuple will be annotated with compiler_ends
          for (auto& tn_arg : tuple_node->fields) {
            new_fields.push_back((*make_end_op)(tn_arg, target));
          }

          // Return a tuple of compiler_ends in the place of the tuple that was
          // annotated with a compiler_end.
          return WithFields(GetRef<Tuple>(tuple_node), new_fields);
        }
      }
      return post;
    }
  };

  auto glob_funcs = module->functions;
  // module is mutable, hence, we make a copy of it.
  module.CopyOnWrite();
  for (const auto& pair : glob_funcs) {
    if (auto* fn = pair.second.as<FunctionNode>()) {
      Function func = GetRef<Function>(fn);
      TupleOutFlattener to_flattener;
      auto removed = PostOrderRewrite(func->body, &to_flattener);
      func = WithFields(func, func->params, removed);
      module->Update(pair.first, func);
      module = relay::transform::InferType()(module);
    }
  }
  return module;
}

class NameMangleExtFuncs : public MixedModeMutator {
 public:
  explicit NameMangleExtFuncs(const IRModule& module, std::function<String(String)> mangle_fn)
      : module_(module), mangle_fn_(mangle_fn) {}

  IRModule Run() {
    auto glob_funcs = module_->functions;

    // Collect function names to be mangled and create
    // global mangled variables
    for (const auto& pair : glob_funcs) {
      if (auto* fn = pair.second.as<FunctionNode>()) {
        auto func = GetRef<Function>(fn);
        if (func->GetAttr<String>(attr::kCompiler).defined()) {
          auto fn_name_mangled = tvm::runtime::SanitizeName(mangle_fn_(pair.first->name_hint));
          GlobalVar gvar = GlobalVar(fn_name_mangled);
          mangled_gvars_[pair.first->name_hint] = gvar;
        }
      }
    }

    // Walk the tree and mangle the functions. Then replace compiler functions
    // with mangled functions in the module
    IRModule new_module = module_->ShallowCopy();
    new_module->functions = {};

    for (const auto& pair : glob_funcs) {
      if (auto* fn = pair.second.as<FunctionNode>()) {
        auto func = GetRef<Function>(fn);

        if (func->GetAttr<String>(attr::kCompiler).defined()) {
          auto new_dict = func->attrs->dict;
          new_dict.Set(tvm::attr::kGlobalSymbol,
                       String(tvm::runtime::SanitizeName(mangle_fn_(pair.first->name_hint))));
          func = WithFields(func, func->params, VisitExpr(func->body), func->ret_type,
                            func->type_params, DictAttrs(new_dict));

          new_module->Add(mangled_gvars_[pair.first->name_hint], func);
        } else {
          func = WithFields(func, func->params, VisitExpr(func->body));
          new_module->Add(pair.first, func);
        }
      }
    }

    return new_module;
  }

 private:
  Expr Rewrite_(const CallNode* call, const Expr& post) final {
    Expr new_expr = post;
    const CallNode* new_call = new_expr.as<CallNode>();
    auto op_node = new_call->op.as<GlobalVarNode>();
    if (op_node == nullptr || mangled_gvars_.find(op_node->name_hint) == mangled_gvars_.end()) {
      return new_expr;
    } else {
      return Call(mangled_gvars_[op_node->name_hint], new_call->args, new_call->attrs,
                  new_call->type_args, new_call->span);
    }
  }

  /*!\brief The IRModule used for partitioning. */
  IRModule module_;
  /*!\brief The function used to mangle operators name */
  std::function<String(String)> mangle_fn_;
  /*!\brief Tabled used to store (unmangled_var_name, mangled_gvar) pairs*/
  std::unordered_map<std::string, GlobalVar> mangled_gvars_;
};

}  // namespace partitioning

namespace transform {

Pass PartitionGraph(String mod_name, bool bind_constants) {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> flatten_tuples = [=](IRModule m,
                                                                                 PassContext pc) {
    // There could be compiler_end annotations on tuples
    // If the corresponding region is having multiple compiler_ends,
    // this would lead to creation of tuples of tuples.
    // Thus, we flatten the tuples by transfering the compiler_end to
    // the tuple inputs.
    return partitioning::FlattenTupleOutputs(m);
  };

  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> remove_defaults = [=](IRModule m,
                                                                                  PassContext pc) {
    // TODO(@comaniac, @zhiics): We should also handle the annotation with "default" attribute
    // by treating them as un-annotated, but we don't have it yet. This workaround pass removes
    // all "default" annotations and should be deleted in the future.
    return partitioning::RemoveDefaultAnnotations(m);
  };

  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> part_func = [=](IRModule m,
                                                                            PassContext pc) {
    return partitioning::Partitioner(m, bind_constants).Partition();
  };

  auto name_mangling_fn = [mod_name](String name) {
    return runtime::get_name_mangled(mod_name, name);
  };

  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> name_mangling_func =
      [=](IRModule m, PassContext pc) {
        return partitioning::NameMangleExtFuncs(m, name_mangling_fn).Run();
      };

  auto flatten_tuples_pass = CreateModulePass(flatten_tuples, 0, "FlattenNestedTuples", {});
  auto remove_default_pass = CreateModulePass(remove_defaults, 0, "RemoveDefaultAnnotations", {});
  auto partition_pass = CreateModulePass(part_func, 0, "PartitionGraph", {});
  auto name_mangling_pass = CreateModulePass(name_mangling_func, 0, "NameMangleExtFuncs", {});
  return Sequential(
      {flatten_tuples_pass, remove_default_pass, partition_pass, name_mangling_pass, InferType()});
}

TVM_REGISTER_GLOBAL("relay._transform.PartitionGraph")
    .set_body_typed([](String mod_name, bool bind_constants) {
      return transform::PartitionGraph(mod_name, bind_constants);
    });

}  // namespace transform

}  // namespace relay
}  // namespace tvm
