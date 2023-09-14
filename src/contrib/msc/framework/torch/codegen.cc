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
  PyCodeGen<TorchCodeGenConfig>::CodeGenHeader();
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
  stack_.func_arg("self", "torch.nn.Module");
  stack_.func_start();
  stack_.call_start("super")
      .call_arg(graph()->name)
      .call_arg("self")
      .call_end()
      .inplace_start("__init__")
      .inplace_end();
  for (const auto& n : graph()->node_names) {
    const auto& node = graph()->FindNode(n);
    if (node->optype == "input") {
      continue;
    }
    CodeGenNode(node);
  }
  stack_.func_end();

  // Write forward
  is_init_ = false;
  stack_.func_def("forward", "List[torch.Tensor]");
  stack_.func_arg("self", "torch.nn.Module");
  for (const auto& i : graph()->GetInputs()) {
    const auto& pair = graph()->FindProducerAndIdx(i);
    stack_.func_arg(IdxOutput(pair.first, pair.second), "torch.Tensor");
  }
  stack_.func_start();
  for (const auto& n : graph()->node_names) {
    const auto& node = graph()->FindNode(n);
    if (node->optype == "input") {
      continue;
    }
    CodeGenNode(node);
  }
  Array<String> idx_outputs;
  for (const auto& o : graph()->GetOutputs()) {
    const auto& pair = graph()->FindProducerAndIdx(o);
    idx_outputs.push_back(IdxOutput(pair.first, pair.second));
  }
  if (idx_outputs.size() == 1) {
    stack_.assign("outputs", idx_outputs[0]);
  } else {
    stack_.assign_list("outputs", idx_outputs);
  }
  stack_.func_end("outputs");
  stack_.class_end();
}

void TorchCodeGen::CodeGenInference() {
  stack_.comment("Build Model")
      .call_start(graph()->name)
      .call_end("model")
      .comment("Load weights")
      .call_start("torch.load")
      .call_str_arg(graph()->name + ".pth")
      .call_end("weights")
      .call_start("model.load_state_dict")
      .call_arg("weights")
      .call_end();
  if (config()->test_device == "gpu") {
    stack_.call_start("model.to").call_start("torch.device").call_arg("cuda").call_end().call_end();
  }
  for (const auto& i : graph()->GetInputs()) {
    const auto& producer = graph()->FindProducer(i);
    stack_.call_start("torch.from_numpy")
        .call_arg("inputs[\"" + i->alias + "\"]")
        .call_end(IdxNode(producer));
  }
  stack_.call_start("model");
  for (const auto& i : graph()->GetInputs()) {
    const auto& producer = graph()->FindProducer(i);
    stack_.call_arg(IdxNode(producer));
    if (config()->test_device == "gpu") {
      stack_.inplace_start("to")
          .call_start("torch.device")
          .call_arg("cuda")
          .call_end()
          .inplace_end();
    }
  }
  stack_.call_end("outputs");
}

const Array<Doc> TorchCodeGen::GetOpCodes(const MSCJoint& node) {
  const auto& ops_map = GetTorchOpCodes();
  auto it = ops_map->find(node->optype);
  ICHECK(it != ops_map->end()) << "Unsupported torch op(" << node->optype << "): " << node;
  it->second->Config(node, config(), is_init_);
  try {
    return it->second->GetDocs();
  } catch (runtime::InternalError& err) {
    LOG(WARNING) << "Failed to get docs for " << node << " : " << err.message();
    throw err;
  }
}

TVM_REGISTER_GLOBAL("msc.framework.torch.GetTorchSources")
    .set_body_typed([](const MSCGraph& graph, const String& codegen_config,
                       const String print_config) -> Map<String, String> {
      TorchCodeGen codegen = TorchCodeGen(graph, codegen_config);
      return codegen.GetSources(print_config);
    });

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
