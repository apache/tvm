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
 * \file src/relay/backend/aot/annotate_used_memory.cc
 * \brief Analyzes the memory pressure at the callsite of primitive functions.
 */

#include <tvm/ir/module.h>
#include <tvm/relay/transform.h>

#include "../../transforms/device_aware_visitors.h"
#include "../manifest_lifetimes.h"

namespace tvm {
namespace relay {
namespace backend {
namespace aot {

/*!
 * \brief Annotates the memory usage of each primitive function by analysing the liveness
 * of the input/output tensors at the function callsite and calculating the total amount of
 * memory these tensors require.
 */
class AnnotateUsedMemoryMutator : public transform::DeviceAwareExprMutator {
 public:
  AnnotateUsedMemoryMutator(const IRModule& module, const transform::ControlFlowGraph& cfg,
                            const transform::LivenessAnalysis& lva)
      : DeviceAwareExprMutator(module), control_flow_graph_(cfg), liveness_(lva) {}

  /*!
   * \brief Get the memory required for a primitive Relay function by calculating the total
   * bytes of the live tensors at the callsite of the function.
   *
   * \param live_tensors The tensors that are live when the function is called.
   * \return int The total number of bytes a function requires.
   */
  int GetMemoryUsage(const transform::VarSet& live_tensors) {
    Array<Type> types_stack = {};
    int memory_usage = 0;

    for (const Var& var : live_tensors) {
      Type var_type = var->checked_type();
      ICHECK(var_type.defined()) << "InferTypes pass should be run before AnnotateUsedMemory pass.";
      types_stack.push_back(var_type);
    }

    while (!types_stack.empty()) {
      Type current_type = types_stack.back();
      types_stack.pop_back();

      if (const auto* tt_node = current_type.as<TupleTypeNode>()) {
        for (const Type& type : tt_node->fields) {
          types_stack.push_back(type);
        }
        continue;
      } else if (const auto* ft_node = current_type.as<FuncTypeNode>()) {
        types_stack.push_back(ft_node->ret_type);
        continue;
      }

      const auto* tt_node = current_type.as<TensorTypeNode>();
      ICHECK(tt_node) << "Expected TensorTypeNode but was " << current_type->GetTypeKey();
      int total_tensor_bytes = GetTensorBytes(tt_node);
      memory_usage += total_tensor_bytes;
    }
    return memory_usage;
  }

  /*!
   * \brief Get the number of bytes a tensor requires.
   *
   * \param tensor_type_node The checked type of the tensor.
   * \return int The number of bytes required.
   */
  int GetTensorBytes(const TensorTypeNode* tensor_type_node) {
    PrimExpr size = tensor_type_node->Size();
    const auto* size_int_imm = size.as<IntImmNode>();
    ICHECK(size_int_imm) << "Expected tensor size to be an IntImmNode but was "
                         << size->GetTypeKey();

    int total_size = size_int_imm->value;
    int dtype_bytes = tensor_type_node->dtype.bytes();
    return total_size * dtype_bytes;
  }

  Expr PostVisitLet_(const LetNode* pre_let_node, const LetNode* post_let_node) override {
    if (const auto* func_node = pre_let_node->value.as<FunctionNode>()) {
      const auto let_bound_values = control_flow_graph_.let_map;
      const transform::ControlFlowGraph::NodePtr cfg_node =
          let_bound_values.at(GetRef<Let>(pre_let_node));
      const transform::VarSet& liveness_out = liveness_.live_out.at(cfg_node);
      int memory_pressure = GetMemoryUsage(liveness_out);
      Function new_func = WithAttr(std::move(GetRef<Function>(func_node)), "used_memory",
                                   tvm::Integer(memory_pressure));
      return Let(post_let_node->var, new_func, post_let_node->body, post_let_node->span);
    }
    return DeviceAwareExprMutator::PostVisitLet_(pre_let_node, post_let_node);
  }

 private:
  /*! \brief Control flow graph representation of the main function. */
  transform::ControlFlowGraph control_flow_graph_;
  /*! \brief Liveness analysis of the main function. */
  transform::LivenessAnalysis liveness_;
};

}  // namespace aot
}  // namespace backend

namespace transform {
Pass AnnotateUsedMemory() {
  runtime::TypedPackedFunc<IRModule(IRModule, PassContext)> pass_func = [=](IRModule mod,
                                                                            PassContext ctx) {
    GlobalVar gv = mod->GetGlobalVar("main");
    Function main_func = Downcast<Function>(mod->Lookup("main"));

    // Perform liveness analysis to determine what tensors are 'live' at each functions
    // callsite.
    support::Arena arena;
    ControlFlowGraph cfg = ControlFlowGraph::Create(&arena, main_func);
    UseDefAnalysis use_def = UseDefAnalysis::Analyze(cfg);
    LivenessAnalysis lva = LivenessAnalysis::Analyze(cfg, use_def);

    auto new_main_body =
        backend::aot::AnnotateUsedMemoryMutator(mod, cfg, lva).VisitExpr(main_func->body);
    if (!new_main_body.same_as(main_func->body)) {
      Function new_main_func = WithFields(main_func, main_func->params, new_main_body);
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
