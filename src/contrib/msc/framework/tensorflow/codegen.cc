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
 * \file src/contrib/msc/framework/tensorflow/codegen.cc
 */
#include "codegen.h"

namespace tvm {
namespace contrib {
namespace msc {

void TensorflowCodeGen::CodeGenHeader() {
  PyCodeGen<TensorflowCodeGenConfig, TFV1CodeGenHelper>::CodeGenHeader();
  stack_.line("from tensorflow.python import ops")
      .line("from tvm.contrib.msc.framework.tensorflow import tf_v1");
}

void TensorflowCodeGen::CodeGenHelper() {
  PyCodeGen<TensorflowCodeGenConfig, TFV1CodeGenHelper>::CodeGenHelper();
  stack_.func_def("get_variable", TensorType())
      .func_arg("name", "str")
      .func_arg("shape", "List[int]")
      .func_arg("dtype", "str")
      .func_arg("weights", "Dict[str, tvm.nd.array]")
      .func_start()
      .cond_if("name in weights")
      .func_call("tf_v1.get_variable", "var")
      .call_arg("name")
      .inplace_start("asnumpy", DocUtils::ToDoc("initializer"),
                     DocUtils::ToIndex("weights", "name"))
      .inplace_end()
      .cond_else()
      .func_call("tf_v1.get_variable", "var")
      .call_arg("name")
      .call_arg("shape")
      .call_arg("dtype")
      .cond_end()
      .func_end("var");
}

void TensorflowCodeGen::CodeGenGraph() {
  stack_.func_def(graph()->name, "List[tf_v1.Tensor]");
  for (const auto& i : graph()->GetInputs()) {
    const auto& pair = graph()->FindProducerAndIdx(i);
    stack_.func_arg(IdxOutputBase(pair.first, pair.second), "tf_v1.Tensor");
  }
  stack_.func_arg("weights", "Dict[str, tvm.nd.array]").func_start();
  // define weights
  stack_.comment("Define the weights");
  for (const auto& n : graph()->node_names) {
    const auto& node = graph()->FindNode(n);
    if (node->optype == "nn.batch_norm") {
      continue;
    }
    for (const auto& pair : node->weights) {
      stack_.func_call("get_variable", IdxWeightBase(node, pair.first, false))
          .call_arg(DocUtils::ToStr(pair.second->name))
          .call_arg(DocUtils::ToList(pair.second->shape, true))
          .call_arg(DocUtils::ToStr(pair.second->DTypeName()))
          .call_arg("weights");
    }
  }
  // define ops
  stack_.comment("Define the ops");
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
    idx_outputs.push_back(IdxOutputBase(pair.first, pair.second));
  }
  if (idx_outputs.size() == 1) {
    stack_.assign("outputs", idx_outputs[0]);
  } else {
    stack_.assign("outputs", DocUtils::ToList(idx_outputs));
  }
  stack_.func_end("outputs");
}

void TensorflowCodeGen::CodeGenInference() {
  stack_.comment("Load weights")
      .scope_start("open(\"" + graph()->name + "_params.bin\", \"rb\")", "f")
      .func_call("tvm.runtime.load_param_dict", "params")
      .func_call("read", "", "f")
      .pop_nest()
      .scope_end()
      .comment("Build Graph")
      .scope_start("tf_v1.Graph().as_default()");
  for (const auto& i : graph()->GetInputs()) {
    const auto& producer = graph()->FindProducer(i);
    stack_.func_call("tf_v1.placeholder", IdxNodeBase(producer))
        .call_arg(DocUtils::ToStr(i->DTypeName()))
        .call_arg(DocUtils::ToList(i->shape))
        .call_arg(DocUtils::ToStr(i->alias));
  }
  stack_.func_call(graph()->name, "outs");
  for (const auto& i : graph()->GetInputs()) {
    const auto& producer = graph()->FindProducer(i);
    stack_.call_arg(IdxNodeBase(producer));
  }
  stack_.call_arg("params").assign("feed_dict", "{}");
  for (const auto& i : graph()->GetInputs()) {
    const auto& producer = graph()->FindProducer(i);
    stack_.assign(DocUtils::ToIndex("feed_dict", IdxNodeBase(producer)),
                  DocUtils::ToIndex("inputs", DocUtils::ToStr(i->alias)));
  }
  stack_.scope_start("tf_v1.Session()", "sess")
      .func_call("run", "", "sess")
      .func_call("ops.variables.global_variables_initializer")
      .pop_nest()
      .func_call("run", "outputs", "sess")
      .call_arg("outs")
      .call_arg("feed_dict", "feed_dict")
      .scope_end()
      .scope_end();
}

const Array<Doc> TensorflowCodeGen::GetOpCodes(const MSCJoint& node) {
  const auto& ops_map = GetTFV1OpCodes();
  auto it = ops_map->find(node->optype);
  ICHECK(it != ops_map->end()) << "Unsupported tensorflow op(" << node->optype << "): " << node;
  it->second->Config(node, config(), prims());
  try {
    return it->second->GetDocs();
  } catch (runtime::InternalError& err) {
    LOG(WARNING) << "Failed to get docs for " << node << " : " << err.message();
    throw err;
  }
}

TVM_REGISTER_GLOBAL("msc.framework.tensorflow.GetTensorflowSources")
    .set_body_typed([](const MSCGraph& graph, const String& codegen_config,
                       const String& print_config) -> Map<String, String> {
      TensorflowCodeGen codegen = TensorflowCodeGen(graph, codegen_config);
      codegen.Init();
      return codegen.GetSources(print_config);
    });

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
