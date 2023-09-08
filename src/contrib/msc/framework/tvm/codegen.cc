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
 * \file src/contrib/msc/framework/tvm/codegen.cc
 */
#include "codegen.h"

namespace tvm {
namespace contrib {
namespace msc {

void RelaxCodeGen::CodeGenHeader() {
  PyCodeGen<RelaxCodeGenConfig>::CodeGenHeader();
  stack_.line("from tvm import relax");
}

void RelaxCodeGen::CodeGenGraph() {
  stack_.func_def(graph()->name, "tvm.IRModule");
  Array<String> idx_inputs;
  for (const auto& i : graph()->GetInputs()) {
    const auto& pair = graph()->FindProducerAndIdx(i);
    const auto& idx_input = IdxOutput(pair.first, pair.second);
    stack_.func_arg(idx_input, "relax.Var");
    idx_inputs.push_back(idx_input);
  }
  stack_.func_start().assign_list("inputs", idx_inputs);
  // define weights
  stack_.comment("Define the weights and constant");
  for (const auto& n : graph()->node_names) {
    const auto& node = graph()->FindNode(n);
    for (const auto& pair : node->weights) {
      const auto& idx_weight = IdxWeight(node, pair.first);
      stack_.call_start("relax.Var")
          .call_str_arg(pair.second->name)
          .call_inplace_start("relax.TensorStructInfo")
          .call_list_arg(pair.second->shape, "", true)
          .call_str_arg(pair.second->DTypeName())
          .call_inplace_end()
          .call_end(idx_weight)
          .call_start("inputs.append")
          .call_arg(idx_weight)
          .call_end();
    }
    if (node->optype == "constant") {
      CodeGenNode(node);
      stack_.call_start("inputs.append").call_arg(IdxNode(node)).call_end();
    }
  }
  stack_.comment("Define the module");
  stack_.assign("block_builder", "relax.BlockBuilder()")
      .scope_start("block_builder.function(name=\"" + graph()->name + "\", params=inputs.copy())");
  for (const auto& n : graph()->node_names) {
    const auto& node = graph()->FindNode(n);
    if (node->optype == "input" || node->optype == "constant") {
      continue;
    }
    int scope_level = CompareScope(node);
    if (scope_level == 1) {
      stack_.scope_start("block_builder.dataflow()");
    } else if (scope_level == -1) {
      stack_.scope_end();
    }
    CodeGenNode(node);
  }
  if (scopes().size() > 1) {
    // end left scopes
    for (size_t i = 0; i < scopes().size() - 1; i++) {
      stack_.scope_end();
    }
  } else if (scopes().size() == 0) {
    // start dataflow scope for non-scope graph
    stack_.scope_start("block_builder.dataflow()");
  }
  // mark outputs
  stack_.comment("Emit the outputs");
  Array<String> idx_exits;
  for (const auto& e : graph()->GetExits()) {
    const auto& idx_exit = IdxNode(e, false);
    stack_.call_start("block_builder.emit_output").call_arg(idx_exit).call_end(idx_exit);
    idx_exits.push_back(idx_exit);
  }
  stack_.scope_end().call_start("block_builder.emit_func_output");
  if (idx_exits.size() == 1) {
    stack_.call_arg(idx_exits[0]);
  } else {
    stack_.call_list_arg(idx_exits);
  }
  stack_.call_end().scope_end().assign("mod", "block_builder.get()").func_end("mod");
}

void RelaxCodeGen::CodeGenInference() {
  for (const auto& i : graph()->GetInputs()) {
    const auto& producer = graph()->FindProducer(i);
    stack_.call_start("relax.Var")
        .call_str_arg(i->alias)
        .call_inplace_start("relax.TensorStructInfo")
        .call_list_arg(i->shape)
        .call_str_arg(i->DTypeName())
        .call_inplace_end()
        .call_end(IdxNode(producer));
  }
  stack_.comment("Build Module").call_start(graph()->name);
  for (const auto& i : graph()->GetInputs()) {
    const auto& producer = graph()->FindProducer(i);
    stack_.call_arg(IdxNode(producer));
  }
  stack_.call_end("mod");
  String target, device;
  if (config()->test_device == "cpu") {
    target = "llvm";
    device = "tvm.cpu()";
  } else if (config()->test_device == "gpu") {
    target = "cuda";
    device = "tvm.cuda()";
  }
  stack_.comment("Load weights")
      .scope_start("open(\"" + graph()->name + "_params.bin\", \"rb\")", "f")
      .call_start("tvm.runtime.load_param_dict")
      .call_arg("f.read()")
      .call_end("params")
      .scope_end()
      .call_start("tvm.relax.transform.BindParams")
      .call_str_arg("main")
      .call_arg("params")
      .call_end("bind_params")
      .call_start("bind_params")
      .call_arg("mod")
      .call_end("mod")
      .call_start("tvm.target.Target")
      .call_str_arg(target)
      .call_end("target")
      .call_start("relax.build")
      .call_arg("mod")
      .call_arg("target")
      .call_end("ex")
      .call_start("relax.VirtualMachine")
      .call_arg("ex")
      .call_arg(device)
      .call_end("vm")
      .call_start("vm[\"main\"]");
  for (const auto& i : graph()->GetInputs()) {
    stack_.call_arg("inputs[\"" + i->alias + "\"]");
  }
  stack_.call_end("outputs");
}

const Array<Doc> RelaxCodeGen::GetOpCodes(const MSCJoint& node) {
  const auto& ops_map = GetRelaxOpCodes();
  auto it = ops_map->find(node->optype);
  ICHECK(it != ops_map->end()) << "Unsupported relax op(" << node->optype << "): " << node;
  it->second->Config(node, config());
  try {
    return it->second->GetDocs();
  } catch (runtime::InternalError& err) {
    LOG(WARNING) << "Failed to get docs for " << node << " : " << err.message();
    throw err;
  }
}

TVM_REGISTER_GLOBAL("msc.framework.tvm.GetRelaxSources")
    .set_body_typed([](const MSCGraph& graph, const String& codegen_config,
                       const String print_config) -> Map<String, String> {
      RelaxCodeGen codegen = RelaxCodeGen(graph, codegen_config);
      return codegen.GetSources(print_config);
    });

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
