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
 * \file src/relay/backend/annotate_used_memory.cc
 * \brief Analyzes the used memory at the callsite of primitive functions.
 */

#include <tvm/ir/module.h>
#include <tvm/relay/attrs/memory.h>
#include <tvm/relay/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "../transforms/device_aware_visitors.h"
#include "../transforms/pass_utils.h"
#include "./liveness_analysis.h"
#include "./utils.h"

namespace tvm {
namespace relay {
namespace backend {

/*!
 * \brief Annotates the minimum required memory of each primitive function callsite by analyzing
 * the liveness of the input/output tensors at each function callsite and calculating the total
 * amount of memory these tensors require. This is added as a "used_memory" annotation to the
 * function in question as a list of the number of bytes for each callsite. In addition, the
 * containing function is annotated with an "io_used_memory" annotation which refers to the total
 * memory required for the IO tensors.
 *
 * Note: This pass does not support dynamic shapes, it is the users responsibility to check this
 * pass isn't applied where dynamic shapes may be input.
 *
 * A simple example:
 *
 * Before:
 * \verbatim
 * def @main(%input: Tensor[(1, 2, 2, 4), int8]) -> Tensor[(1, 2, 2, 4), int8] {
 *   let %x_0 = fn (%x: Tensor[(1, 2, 2, 4), int8], Primitive=1) -> Tensor[(1, 2, 2, 4), int8] {
 *     nn.max_pool2d(%x, pool_size=[1, 1], padding=[0, 0, 0, 0])
 *   };
 *   let %x_1 = %x_0(%input);
 *   %x_1
 * }
 * \endverbatim
 *
 * After:
 * \verbatim
 * def @main(%input: Tensor[(1, 2, 2, 4), int8], io_used_memory=32) -> Tensor[(1, 2, 2, 4), int8] {
 *   let %x_0: fn (%x: Tensor[(1, 2, 2, 4), int8], Primitive=1, used_memory=[32]) -> Tensor[(1, 2,
 * 2, 4), int8] {
 *      nn.max_pool2d(%x, pool_size=[1, 1], padding=[0, 0, 0, 0])
 *   };
 *   let %x_1: Tensor[(1, 2, 2, 4), int8] = %x_0(%input);
 *   %x_1
 * }
 * \endverbatim
 *
 * Note that in the simple example above io_used_memory and used_memory are the same since there
 * is only one primitive function.
 */
class AnnotateUsedMemoryMutator : public transform::DeviceAwareExprMutator {
 public:
  AnnotateUsedMemoryMutator(const IRModule& module, const transform::ControlFlowGraph& cfg,
                            const transform::LivenessAnalysis& lva)
      : DeviceAwareExprMutator(module), control_flow_graph_(cfg), liveness_(lva) {}

  /*!
   * \brief Mutates the input function. In addition, an "io_used_memory" annotation is
   * added to the input function which refers to the total size required for the IO
   * tensors.
   */
  Function operator()(const Function& func) {
    uint64_t io_used_memory = 0;

    // Inputs
    for (const Var& param : func->params) {
      Type type = param->checked_type();
      ICHECK(type.defined()) << "InferType pass should be run before AnnotateUsedMemory.";
      ICHECK(!IsDynamic(type)) << "AnnotateUsedMemory does not support dynamic shapes.";
      io_used_memory += CalculateRelayExprSizeBytes(type);
    }

    // Outputs
    Type type = func->body->checked_type();
    ICHECK(type.defined()) << "InferType pass should be run before AnnotateUsedMemory.";
    ICHECK(!IsDynamic(type)) << "AnnotateUsedMemory does not support dynamic shapes.";
    io_used_memory += CalculateRelayExprSizeBytes(type);

    Expr new_func_body = VisitExpr(func->body);
    Function new_func = WithFields(func, func->params, new_func_body);
    return WithAttr(std::move(new_func), "io_used_memory",
                    tvm::IntImm(tvm::DataType::UInt(64), io_used_memory));
  }

  /*!
   * \brief Establish which let bindings have primitive function values.
   */
  std::pair<Var, Expr> PreVisitLetBinding_(const Var& var, const Expr& value) override {
    if (const auto* func_node = value.as<FunctionNode>()) {
      ICHECK(func_node->attrs.HasNonzeroAttr(attr::kPrimitive))
          << "Expect top-level functions to be primitive.";
      let_bound_prim_func_.insert(var);
    }
    return DeviceAwareExprMutator::PreVisitLetBinding_(var, value);
  }

  /*!
   * \brief Visit let nodes and perform one of two actions depending on their value:
   *
   * 1. CallNode - Calculate "used_memory" annotation value at the callsite of
   *               primitive functions.
   *
   * 2. FunctionNode - Annotate functions with "used_memory" annotation based on the
   *                   previous analysis at the callsite.
   *
   */
  Expr PostVisitLet_(const LetNode* pre_let_node, const LetNode* post_let_node) override {
    Var let_var = post_let_node->var;
    Expr let_value = IgnoreOnDevice(post_let_node->value);

    if (let_value->IsInstance<CallNode>()) {
      Call callsite = Downcast<Call>(let_value);
      if (CheckPrimitiveFunctionCall(callsite)) {
        Var call_op = Downcast<Var>(callsite->op);

        // Find all the vars that are live at the callsite. This is done by merging the
        // in and out varset's and then removing the var that references the primitive
        // function itself since we don't want this included in the calculation.
        const transform::ControlFlowGraph::NodePtr cfg_node =
            control_flow_graph_.let_map.at(GetRef<Let>(pre_let_node));
        transform::VarSet live_tensors = liveness_.live_in.at(cfg_node);
        const transform::VarSet& live_out = liveness_.live_out.at(cfg_node);
        live_tensors.insert(live_out.begin(), live_out.end());
        live_tensors.erase(call_op);

        // Calculate size of live tensors and store to allow annotation when the function
        // gets visited.
        uint64_t used_memory = 0;
        for (const auto& var : live_tensors) {
          Type type = var->checked_type();
          ICHECK(type.defined()) << "InferType pass should be run before AnnotateUsedMemory.";
          ICHECK(!IsDynamic(type)) << "AnnotateUsedMemory does not support dynamic shapes.";
          used_memory += CalculateRelayExprSizeBytes(type);
        }
        IntImm annotation(DataType::UInt(64), used_memory);
        used_memory_annotations_[call_op].push_back(annotation);
      }
    } else if (let_value->IsInstance<FunctionNode>()) {
      Function func = Downcast<Function>(let_value);
      ICHECK(used_memory_annotations_.find(let_var) != used_memory_annotations_.end())
          << "Could not find used_memory value for primitive function bound at "
          << let_var->name_hint();
      Array<IntImm> used_memory = used_memory_annotations_[let_var];
      used_memory_annotations_.erase(let_var);

      Function new_func = WithAttr(std::move(func), "used_memory",
                                   Array<IntImm>(used_memory.rbegin(), used_memory.rend()));
      return Let(let_var, new_func, post_let_node->body, post_let_node->span);
    }

    return DeviceAwareExprMutator::PostVisitLet_(pre_let_node, post_let_node);
  }

 private:
  /*!
   * \brief Check if a call is a primitive function callsite.
   */
  bool CheckPrimitiveFunctionCall(const Call& callsite) {
    if (auto var = callsite->op.as<Var>()) {
      if (let_bound_prim_func_.find(var.value()) != let_bound_prim_func_.end()) {
        return true;
      }
    }
    return false;
  }

  /*! \brief Control flow graph representation of the main function. */
  transform::ControlFlowGraph control_flow_graph_;
  /*! \brief Liveness analysis of the main function. */
  transform::LivenessAnalysis liveness_;
  /*! \brief Var's that reference primitive functions. */
  std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual> let_bound_prim_func_;
  /*! \brief Stores the calculated uint64 used_memory values so they can be annotated on the
   * relevant function. */
  std::unordered_map<Var, Array<IntImm>, ObjectPtrHash, ObjectPtrEqual> used_memory_annotations_;
};

}  // namespace backend

namespace transform {

Pass AnnotateUsedMemory() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule mod,
                                                                            PassContext ctx) {
    GlobalVar gv = mod->GetGlobalVar("main");
    Function main_func = Downcast<Function>(mod->Lookup("main"));

    // Perform liveness analysis to determine what tensors are 'live' at each functions callsite.
    support::Arena arena;
    ControlFlowGraph cfg = ControlFlowGraph::Create(&arena, main_func);
    UseDefAnalysis use_def = UseDefAnalysis::Analyze(cfg);
    LivenessAnalysis lva = LivenessAnalysis::Analyze(cfg, use_def);

    auto new_main_func = backend::AnnotateUsedMemoryMutator(mod, cfg, lva)(main_func);
    if (!new_main_func.same_as(main_func)) {
      mod->Update(gv, new_main_func);
    }
    return mod;
  };
  return CreateModulePass(pass_func, 0, "AnnotateUsedMemory", {"ToANormalForm", "InferType"});
}

TVM_REGISTER_GLOBAL("relay._transform.AnnotateUsedMemory").set_body_typed(AnnotateUsedMemory);

}  // namespace transform
}  // namespace relay
}  // namespace tvm
