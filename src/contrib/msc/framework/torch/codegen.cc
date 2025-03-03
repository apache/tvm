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
 * \file src/contrib/msc/framework/torch/codegen.cc
 */
#include "codegen.h"

namespace tvm {
namespace contrib {
namespace msc {

void TorchCodeGen::CodeGenHeader() {
  PyCodeGen<TorchCodeGenConfig, TorchCodeGenHelper>::CodeGenHeader();
  stack_.line("import torch");
  stack_.line("from torch import nn");
  stack_.line("from torch.nn import functional");
}

void TorchCodeGen::CodeGenGraph() {
  stack_.class_def(graph()->name + "(torch.nn.Module)");
  stack_.class_start();

  // Write init
  is_init_ = true;
  stack_.func_def("__init__", "torch.nn.Module");
  if (config()->use_tools) {
    stack_.func_decorator("msc_tools.wrap_step(\"build\",\"" + config()->tools_tag + "\")");
  }
  stack_.func_arg("self", "torch.nn.Module");
  if (config()->use_plugin) {
    stack_.func_arg("plugin", "Any");
  }
  stack_.func_start()
      .func_call("super")
      .call_arg(graph()->name)
      .call_arg("self")
      .method_call("__init__");
  for (const auto& n : graph()->node_names) {
    const auto& node = graph()->FindNode(n);
    if (node->optype == "input") {
      continue;
    }
    CodeGenNode(node, false);
  }
  stack_.func_end();

  // Write forward
  is_init_ = false;
  stack_.func_def("forward", "List[torch.Tensor]");
  if (config()->use_tools) {
    stack_.func_decorator("msc_tools.wrap_step(\"forward\",\"" + config()->tools_tag + "\")");
  }
  stack_.func_arg("self", "torch.nn.Module");
  for (const auto& i : graph()->GetInputs()) {
    const auto& pair = graph()->FindProducerAndIdx(i);
    stack_.func_arg(IdxOutputBase(pair.first, pair.second), "torch.Tensor");
  }
  stack_.func_start();
  if (config()->use_tools) {
    stack_.comment("Define all weights");
    for (const auto& n : graph()->node_names) {
      const auto& node = graph()->FindNode(n);
      for (const auto& pair : node->weights) {
        stack_.assign(IdxWeightBase(node, pair.first, false), "self." + pair.second->alias);
      }
    }
    stack_.comment("End of define all weights").line();
  }
  for (const auto& n : graph()->node_names) {
    const auto& node = graph()->FindNode(n);
    if (node->optype == "input") {
      continue;
    }
    CodeGenNode(node, config()->use_tools);
  }
  Array<String> idx_outputs;
  for (const auto& o : graph()->GetOutputs()) {
    const auto& pair = graph()->FindProducerAndIdx(o);
    idx_outputs.push_back(IdxOutputBase(pair.first, pair.second, true));
  }
  if (idx_outputs.size() == 1) {
    stack_.assign("outputs", idx_outputs[0]);
  } else {
    stack_.assign("outputs", DocUtils::ToList(idx_outputs));
  }
  stack_.func_end("outputs");
  stack_.class_end();
}

void TorchCodeGen::CodeGenInference() {
  if (config()->use_plugin) {
    stack_.comment("Import Plugin")
        .line("from msc_plugin.torch import PluginManager")
        .line()
        .func_call("PluginManager", "plugin");
  }
  stack_.comment("Build Model").func_call(graph()->name, "model");
  if (config()->use_plugin) {
    stack_.call_arg("plugin");
  }
  stack_.comment("Load weights")
      .func_call("torch.load", "weights")
      .call_arg(DocUtils::ToStr(graph()->name + ".pth"))
      .func_call("load_state_dict", "", "model")
      .call_arg("weights");
  if (config()->test_device == "gpu") {
    stack_.func_call("to", "", "model").func_call("torch.device").call_arg("cuda").pop_nest();
  }
  for (const auto& i : graph()->GetInputs()) {
    const auto& producer = graph()->FindProducer(i);
    stack_.func_call("torch.from_numpy", IdxNodeBase(producer))
        .call_arg(DocUtils::ToIndex("inputs", DocUtils::ToStr(i->alias)));
  }
  stack_.func_call("model", "outputs");
  for (const auto& i : graph()->GetInputs()) {
    const auto& producer = graph()->FindProducer(i);
    stack_.call_arg(IdxNodeBase(producer));
    if (config()->test_device == "gpu") {
      stack_.method_call("to").func_call("torch.device").call_arg("cuda");
    }
  }
}

const Array<Doc> TorchCodeGen::GetOpCodes(const MSCJoint& node) {
  const auto& ops_map = GetTorchOpCodes();
  auto it = ops_map->find(GetOpType(node));
  ICHECK(it != ops_map->end()) << "Unsupported torch op(" << node->optype << "): " << node;
  it->second->Config(node, config(), is_init_, prims());
  try {
    return it->second->GetDocs();
  } catch (runtime::InternalError& err) {
    LOG(WARNING) << "Failed to get docs for " << node << " : " << err.message();
    throw err;
  }
}

TVM_REGISTER_GLOBAL("msc.framework.torch.GetTorchSources")
    .set_body_typed([](const MSCGraph& graph, const String& codegen_config,
                       const String& print_config) -> Map<String, String> {
      TorchCodeGen codegen = TorchCodeGen(graph, codegen_config);
      codegen.Init();
      return codegen.GetSources(print_config);
    });

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
