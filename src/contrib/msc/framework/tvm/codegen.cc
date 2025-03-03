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
  PyCodeGen<RelaxCodeGenConfig, RelaxCodeGenHelper>::CodeGenHeader();
  stack_.line("from tvm import relax");
}

void RelaxCodeGen::CodeGenGraph() {
  stack_.func_def(graph()->name, "tvm.IRModule");
  Array<String> idx_inputs;
  for (const auto& i : graph()->GetInputs()) {
    const auto& pair = graph()->FindProducerAndIdx(i);
    const auto& idx_input = IdxOutputBase(pair.first, pair.second);
    stack_.func_arg(idx_input, "relax.Var");
    idx_inputs.push_back(idx_input);
  }
  if (config()->use_plugin) {
    stack_.func_arg("plugin", "Any");
  }
  stack_.func_start().assign("inputs", DocUtils::ToList(idx_inputs, true));
  // define weights
  stack_.comment("Define the weights");
  for (const auto& n : graph()->node_names) {
    const auto& node = graph()->FindNode(n);
    for (const auto& pair : node->weights) {
      const auto& idx_weight = IdxWeightBase(node, pair.first, false);
      stack_.func_call("relax.Var", idx_weight)
          .call_arg(DocUtils::ToStr(pair.second->name))
          .func_call("relax.TensorStructInfo")
          .call_arg(DocUtils::ToList(pair.second->shape, true), "")
          .call_arg(DocUtils::ToStr(pair.second->DTypeName()))
          .pop_nest()
          .func_call("append", "", "inputs")
          .call_arg(idx_weight);
    }
  }
  stack_.comment("Define the module");
  stack_.func_call("relax.BlockBuilder", "block_builder")
      .scope_start("block_builder.function(name=\"" + graph()->name + "\", params=inputs.copy())");
  if (config()->use_tools) {
    stack_.func_call("msc_tools.execute_step")
        .call_arg(DocUtils::ToStr("before_build"))
        .call_arg("block_builder");
  }
  for (const auto& n : graph()->node_names) {
    const auto& node = graph()->FindNode(n);
    if (node->optype == "input") {
      continue;
    }
    int scope_level = CompareScope(node);
    if (scope_level == -1) {
      stack_.scope_end();
    }
    CodeGenNode(node, config()->use_tools);
  }
  if (scopes().size() > 1) {
    // end left scopes
    for (size_t i = 0; i < scopes().size() - 1; i++) {
      stack_.scope_end();
    }
  }
  // mark outputs
  stack_.comment("Emit the outputs");
  Array<String> idx_exits;

  for (const auto& e : graph()->GetExits()) {
    const auto& idx_exit = IdxNodeBase(e) + (config()->use_tools ? "_exit" : "");
    if (config()->use_tools) {
      if (e->outputs.size() > 1) {
        Array<String> tuple_outputs;
        for (size_t o_idx = 0; o_idx < e->outputs.size(); o_idx++) {
          const auto& t_output = IdxOutputBase(e, o_idx, true);
          tuple_outputs.push_back(t_output);
        }
        stack_.func_call("relax.Tuple", idx_exit).call_arg(DocUtils::ToList(tuple_outputs));
        stack_.func_call("emit", idx_exit, "block_builder").call_arg(idx_exit);
        stack_.call_arg(DocUtils::ToStr(e->name + "_exit"), "name_hint");
      }
    }
    idx_exits.push_back(idx_exit);
  }

  if (config()->use_tools) {
    stack_.func_call("msc_tools.execute_step", "output").call_arg(DocUtils::ToStr("after_build"));
    if (idx_exits.size() == 1) {
      stack_.call_arg(idx_exits[0]);
    } else {
      stack_.call_arg(DocUtils::ToList(idx_exits));
    }
  }
  stack_.func_call("emit_func_output", "", "block_builder");
  if (config()->use_tools) {
    stack_.call_arg("output");
  } else if (idx_exits.size() == 1) {
    stack_.call_arg(idx_exits[0]);
  } else {
    stack_.call_arg(DocUtils::ToList(idx_exits));
  }
  stack_.scope_end().func_call("finalize", "mod", "block_builder").func_end("mod");
}

void RelaxCodeGen::CodeGenInference() {
  if (config()->use_plugin) {
    stack_.comment("Import Plugin")
        .line("from msc_plugin.tvm import PluginManager")
        .line()
        .func_call("PluginManager", "plugin");
  }
  for (const auto& i : graph()->GetInputs()) {
    const auto& producer = graph()->FindProducer(i);
    stack_.func_call("relax.Var", IdxNodeBase(producer))
        .call_arg(DocUtils::ToStr(i->alias))
        .func_call("relax.TensorStructInfo")
        .call_arg(DocUtils::ToList(i->shape))
        .call_arg(DocUtils::ToStr(i->DTypeName()))
        .pop_nest();
  }
  stack_.comment("Build Module").func_call(graph()->name, "mod");
  if (config()->use_plugin) {
    stack_.call_arg("plugin");
  }
  for (const auto& i : graph()->GetInputs()) {
    const auto& producer = graph()->FindProducer(i);
    stack_.call_arg(IdxNodeBase(producer));
  }
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
      .func_call("tvm.runtime.load_param_dict", "params")
      .inplace_start("read", "", "f")
      .inplace_end()
      .scope_end()
      .func_call("tvm.relax.transform.BindParams", "bind_params")
      .call_arg(DocUtils::ToStr("main"))
      .call_arg("params")
      .func_call("bind_params", "mod")
      .call_arg("mod")
      .func_call("tvm.target.Target", "target")
      .call_arg(DocUtils::ToStr(target))
      .func_call("tvm.relax.transform.LegalizeOps()", "mod")
      .call_arg("mod")
      .scope_start("tvm.transform.PassContext(opt_level=3)")
      .func_call("relax.build", "ex")
      .call_arg("mod")
      .call_arg("target")
      .func_call("relax.VirtualMachine", "vm")
      .call_arg("ex")
      .call_arg(device)
      .scope_end()
      .assign("f_main", DocUtils::ToIndex("vm", DocUtils::ToStr("main")))
      .func_call("f_main", "outputs");
  for (const auto& i : graph()->GetInputs()) {
    stack_.call_arg(DocUtils::ToIndex("inputs", DocUtils::ToStr(i->alias)));
  }
}

const String RelaxCodeGen::DescribePrim(const MSCPrim& prim) {
  if (prim->optype == "shape") {
    const auto& producer = graph()->FindNode(prim->GetTypeAttr<std::string>("producer"));
    int out_idx = prim->GetTypeAttr<int>("out_idx");
    const auto& dim = prim->GetTypeAttr<std::string>("dim");
    return IdxOutputBase(producer, out_idx) + ".struct_info.shape[" + dim + "]";
  }
  return PyCodeGen<RelaxCodeGenConfig, RelaxCodeGenHelper>::DescribePrim(prim);
}

const Array<Doc> RelaxCodeGen::GetOpCodes(const MSCJoint& node) {
  const auto& ops_map = GetRelaxOpCodes();
  auto it = ops_map->find(GetOpType(node));
  ICHECK(it != ops_map->end()) << "Unsupported relax op(" << node->optype << "): " << node;
  it->second->Config(node, config(), prims());
  try {
    return it->second->GetDocs();
  } catch (runtime::InternalError& err) {
    LOG(WARNING) << "Failed to get docs for " << node << " : " << err.message();
    throw err;
  }
}

TVM_REGISTER_GLOBAL("msc.framework.tvm.GetRelaxSources")
    .set_body_typed([](const MSCGraph& graph, const String& codegen_config,
                       const String& print_config) -> Map<String, String> {
      RelaxCodeGen codegen = RelaxCodeGen(graph, codegen_config);
      codegen.Init();
      return codegen.GetSources(print_config);
    });

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
