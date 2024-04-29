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
 * \file src/contrib/msc/plugin/torch_codegen.cc
 */
#include "torch_codegen.h"

namespace tvm {
namespace contrib {
namespace msc {

void TorchPluginCodeGen::CodeGenAttrDeclare(const Plugin& plugin) {
  BasePluginCodeGen<TorchPluginCodeGenConfig>::CodeGenAttrDeclare(plugin);
  const auto& attr_name = MetaAttrCls(plugin);
  // serialize method for attr
  stack_.comment("serialize method")
      .func_def(attr_name + "_serialize", "std::vector<std::string>")
      .func_arg("meta_attr", "const " + attr_name + "&");
  // deserialize method for attr
  stack_.comment("deserialize method")
      .func_def(attr_name + "_deserialize")
      .func_arg("attrs", "const std::vector<std::string>&")
      .func_arg("meta_attr", attr_name + "&");
}

void TorchPluginCodeGen::CodeGenAttrDefine(const Plugin& plugin) {
  const auto& attr_name = MetaAttrCls(plugin);
  // serialize method for attr
  stack_.func_def(attr_name + "_serialize", "std::vector<std::string>")
      .func_arg("meta_attr", "const " + attr_name + "&")
      .func_start()
      .declare("std::vector<std::string>", "attrs");
  for (const auto& a : plugin->attrs) {
    stack_.func_call("push_back", "", "attrs")
        .inplace_start("SerializeUtils::ToString")
        .call_arg(DocUtils::ToAttrAccess("meta_attr", a->name))
        .inplace_end();
  }
  stack_.func_end("attrs");
  // deserialize method for attr
  stack_.func_def(attr_name + "_deserialize")
      .func_arg("attrs", "const std::vector<std::string>&")
      .func_arg("meta_attr", attr_name + "&")
      .func_start();
  for (size_t i = 0; i < plugin->attrs.size(); i++) {
    stack_.func_call("SerializeUtils::FromString")
        .call_arg(DocUtils::ToIndex("attrs", i))
        .call_arg(DocUtils::ToAttrAccess("meta_attr", plugin->attrs[i]->name));
  }
  stack_.func_end();
}

void TorchPluginCodeGen::CodeGenOpDeclare(const Plugin& plugin) {
  stack_.struct_start(plugin->name + " : torch::CustomClassHolder");
  // constructor
  stack_.constructor_def(plugin->name).constructor_arg("attrs", "const std::vector<std::string>&");
  // serialize method
  stack_.comment("serialize method").func_def("serialize", "const std::vector<std::string>");
  // compute method
  stack_.comment("main compute")
      .func_def("compute", "std::vector<torch::Tensor>")
      .func_arg("input_tensors", "const std::vector<torch::Tensor>&");
  // members
  stack_.comment("members")
      .declare(MetaAttrCls(plugin), "meta_attr_")
      .declare("std::vector<MetaLayout>", "layouts_")
      .declare("std::string", "name_");
  stack_.struct_end();
  // entry method
  stack_.comment("Entry method for plugin " + plugin->name)
      .func_def(EntryName(plugin), "std::vector<torch::Tensor>")
      .func_arg("instance", "const c10::intrusive_ptr<" + plugin->name + ">&");
  for (const auto& input : plugin->inputs) {
    stack_.func_arg(input->name, "const torch::Tensor&");
  }
  for (const auto& a : plugin->attrs) {
    stack_.func_arg(a->name, "const " + ToTorchType(a->type) + "&");
  }
  stack_.func_arg("name", "const std::string&");
}

void TorchPluginCodeGen::CodeGenOpDefine(const Plugin& plugin) {
  const auto& attr_name = MetaAttrCls(plugin);
  // define constructor
  stack_.constructor_def(plugin->name + "::" + plugin->name)
      .constructor_arg("attrs", "const std::vector<std::string>&")
      .constructor_start()
      .comment("get attributes")
      .func_call(attr_name + "_deserialize")
      .call_arg("attrs")
      .call_arg("meta_attr_")
      .comment("get extra info")
      .assign("name_", DocUtils::ToIndex("attrs", plugin->attrs.size()))
      .for_start("i", 1 + plugin->attrs.size(), 1 + plugin->attrs.size() + plugin->inputs.size())
      .func_call("push_back", "", "layouts_")
      .inplace_start("MetaLayout")
      .call_arg(DocUtils::ToIndex("attrs", "i"))
      .inplace_end()
      .for_end()
      .constructor_end();
  // define serialize
  stack_.func_def(plugin->name + "::serialize", "const std::vector<std::string>")
      .func_start()
      .assign("attrs", attr_name + "_serialize(meta_attr_)", "std::vector<std::string>")
      .func_call("push_back", "", "attrs")
      .call_arg("name_")
      .for_start("i", 0, plugin->inputs.size())
      .func_call("push_back", "", "attrs")
      .call_arg(DocUtils::ToAttrAccess(DocUtils::ToIndex("layouts_", "i"), "name()"))
      .for_end()
      .func_end("attrs");
  // compute method
  stack_.func_def(plugin->name + "::compute", "std::vector<torch::Tensor>")
      .func_arg("input_tensors", "const std::vector<torch::Tensor>&")
      .func_start()
      .declare("std::vector<torch::Tensor>", "output_tensors");
  if (plugin->externs.count("infer_buffer")) {
    stack_.declare("std::vector<torch::Tensor>", "buffer_tensors");
  }
  stack_.line()
      .comment("extract meta inputs")
      .declare("std::vector<MetaTensor>", "input_metas")
      .for_start("i", 0, plugin->inputs.size())
      .func_call("push_back", "", "input_metas")
      .inplace_start("TorchUtils::ToMetaTensor")
      .call_arg(DocUtils::ToIndex("input_tensors", "i"))
      .call_arg(DocUtils::ToIndex("layouts_", "i"))
      .inplace_end()
      .for_end();
  // malloc outputs and buffers
  ICHECK(plugin->externs.count("infer_output")) << "Can not find extern shape";
  CodeGenMalloc(plugin, plugin->outputs, "output");
  if (plugin->externs.count("infer_buffer")) {
    CodeGenMalloc(plugin, plugin->buffers, "buffer");
  }
  // do the compute
  String device_cond = "";
  for (size_t i = 0; i < plugin->inputs.size(); i++) {
    if (plugin->inputs[i]->device == "cuda" || plugin->inputs[i]->device == "default") {
      device_cond = device_cond + "input_tensors[" + std::to_string(i) + "].is_cuda()";
    } else {
      device_cond = device_cond + "!input_tensors[" + std::to_string(i) + "].is_cuda()";
    }
    device_cond = device_cond + (i == plugin->inputs.size() - 1 ? "" : " && ");
  }
  stack_.line().comment("do the compute").cond_if(device_cond);
  CodeGenCompute(plugin, "cuda");
  stack_.cond_else();
  CodeGenCompute(plugin, "cpu");
  stack_.cond_end();
  stack_.func_end("output_tensors");

  // register op
  const auto& entry_name = EntryName(plugin);
  stack_.func_def(entry_name, "std::vector<torch::Tensor>")
      .func_arg("instance", "const c10::intrusive_ptr<" + plugin->name + ">&");
  for (const auto& input : plugin->inputs) {
    stack_.func_arg(input->name, "const torch::Tensor&");
  }
  for (const auto& a : plugin->attrs) {
    stack_.func_arg(a->name, "const " + ToTorchType(a->type) + "&");
  }
  stack_.func_arg("name", "const std::string&");
  stack_.func_start().declare("std::vector<torch::Tensor>", "inputs", 0, false);
  for (const auto& input : plugin->inputs) {
    stack_.declare_arg(input->name);
  }
  const auto& outputs_doc = DocUtils::ToDeclare("std::vector<torch::Tensor>", "outputs");
  stack_.func_call("compute", outputs_doc, DocUtils::ToPtr("instance")).call_arg("inputs");
  stack_.func_end("outputs");
  stack_.comment("Bind plugin " + plugin->name + " to python")
      .func_def("TORCH_LIBRARY", DocSymbol::Empty())
      .func_arg(plugin->name, DocSymbol::Empty())
      .func_arg("m", DocSymbol::Empty())
      .func_start()
      .lambda_def("serialize")
      .lambda_arg("op", "const c10::intrusive_ptr<" + plugin->name + ">&")
      .lambda_start()
      .lambda_end(DocUtils::ToAttrAccess(DocUtils::ToPtr("op"), "serialize()"))
      .lambda_def("deserialize")
      .lambda_arg("state", "std::vector<std::string>")
      .lambda_start()
      .lambda_end("c10::make_intrusive<" + plugin->name + ">(std::move(state))")
      .func_call("class_<" + plugin->name + ">", "", "m")
      .call_arg(DocUtils::ToStr(plugin->name))
      .method_call("def", true)
      .call_arg("torch::init<const std::vector<std::string>>()")
      .method_call("def", true)
      .call_arg(DocUtils::ToStr("compute"))
      .call_arg("&" + plugin->name + "::compute")
      .method_call("def_pickle", true)
      .call_arg("serialize")
      .call_arg("deserialize")
      .func_call("def", "", "m")
      .call_arg(DocUtils::ToStr(entry_name))
      .call_arg(entry_name)
      .func_end();
}

void TorchPluginCodeGen::CodeGenCmake(const std::set<String>& devices) {
  Map<String, String> flags;
  flags.Set("PLUGIN_SUPPORT_TORCH", "");
  CodeGenPreCmake(devices, flags);
  stack_.line()
      .line("set(CMAKE_CXX_STANDARD 14)")
      .line("list(APPEND CMAKE_PREFIX_PATH \"" + config()->torch_prefix + "\")")
      .line("find_package(Torch REQUIRED)");
  Array<String> includes, libs;
  libs.push_back("${TORCH_LIBRARIES}");
  CodeGenPostCmake(devices, includes, libs);
}

void TorchPluginCodeGen::CodeGenManagerDepends() {
  BasePluginCodeGen<TorchPluginCodeGenConfig>::CodeGenManagerDepends();
  stack_.line("import torch")
      .line()
      .func_def("to_string", "str")
      .func_arg("value", "Any")
      .func_start()
      .switch_start("isinstance(value, (list, tuple))")
      .assign("str_value", "\",\".join([str(len(value))] + [to_string(v) for v in value])")
      .switch_case("isinstance(value, bool)")
      .assign("str_value", "\"1\" if value else \"0\"")
      .switch_case()
      .assign("str_value", "str(value)")
      .switch_end()
      .func_end("str_value");
}

void TorchPluginCodeGen::CodeGenManagerMethods() {
  BasePluginCodeGen<TorchPluginCodeGenConfig>::CodeGenManagerMethods();
  // libs_loaded method
  stack_.func_def("libs_loaded")
      .func_arg("self", "object")
      .func_start()
      .assign("loaded_libs", "set()")
      .assign("loaded", DocUtils::ToDoc(false))
      .for_start("lib", "torch.classes.loaded_libraries")
      .func_call("add", "", "loaded_libs")
      .inplace_start("os.path.basename")
      .call_arg("lib")
      .inplace_end()
      .for_end()
      .for_start("lib", "os.listdir(self._lib_folder)")
      .cond_if("lib in loaded_libs")
      .assign("loaded", DocUtils::ToDoc(true))
      .line("break")
      .cond_end()
      .for_end()
      .func_end("loaded");
  // setup method
  stack_.func_def("setup")
      .func_arg("self", "object")
      .func_start()
      .for_start("lib", "os.listdir(self._lib_folder)")
      .assign("lib_file", "os.path.join(self._lib_folder, lib)")
      .cond_if("\"" + config()->project_name + "\" in lib")
      .func_call("load_library", "", "torch.classes")
      .call_arg("lib_file")
      .cond_else()
      .func_call("CDLL", "", "ctypes")
      .call_arg("lib_file")
      .cond_end()
      .for_end()
      .func_end();
}

void TorchPluginCodeGen::CodeGenOpBuilder(const Plugin& plugin) {
  const auto& entry_name = EntryName(plugin);
  stack_.func_def(plugin->name).func_arg("self", "object");
  for (const auto& attr : plugin->attrs) {
    stack_.func_arg(attr->name, attr->type, attr->default_value);
  }
  stack_.func_arg("name", "str", "\"" + plugin->name + "\"")
      .func_arg("layouts", "List[str]", "None")
      .func_start()
      .class_def(plugin->name + "(torch.nn.Module)")
      .class_start();
  // init method
  stack_.func_def("__init__").func_arg("self", "torch.nn.Module");
  for (const auto& attr : plugin->attrs) {
    stack_.func_arg(attr->name, attr->type, attr->default_value);
  }
  stack_.func_arg("name", "str", "\"" + plugin->name + "\"")
      .func_arg("layouts", "List[str]", "None")
      .func_start()
      .func_call("__init__", "", "super()");
  for (const auto& attr : plugin->attrs) {
    stack_.assign(DocUtils::ToAttrAccess("self", attr->name), attr->name);
  }
  stack_.assign(DocUtils::ToAttrAccess("self", "name"), "name")
      .cond_if("layouts is None")
      .assign(DocUtils::ToAttrAccess("self", "layouts"),
              "[\"\"] * " + std::to_string(plugin->inputs.size()))
      .cond_else()
      .assign(DocUtils::ToAttrAccess("self", "layouts"), "layouts")
      .cond_end()
      .line()
      .assign("attr_strs", "[]");
  for (const auto& attr : plugin->attrs) {
    stack_.func_call("append", "", "attr_strs")
        .inplace_start("to_string")
        .call_arg(attr->name)
        .inplace_end();
  }
  stack_.func_call("append", "", "attr_strs")
      .call_arg("name")
      .func_call("extend", "", "attr_strs")
      .call_arg(DocUtils::ToAttrAccess("self", "layouts"))
      .line()
      .func_call(plugin->name + "." + plugin->name, "self._inner_class", "torch.classes")
      .call_arg("attr_strs")
      .func_end();
  // forward method
  stack_.func_def("forward", "List[torch.Tensor]").func_arg("self", "torch.nn.Module");
  for (const auto& t : plugin->inputs) {
    stack_.func_arg(t->name, "torch.Tensor");
  }
  stack_.func_start()
      .func_call(plugin->name + "." + entry_name, "outputs", "torch.ops")
      .call_arg("self._inner_class");
  for (const auto& t : plugin->inputs) {
    stack_.call_arg(t->name);
  }
  for (const auto& a : plugin->attrs) {
    stack_.call_arg(DocUtils::ToAttrAccess("self", a->name));
  }
  stack_.call_arg(DocUtils::ToAttrAccess("self", "name"));
  if (plugin->outputs.size() == 1) {
    stack_.func_end(DocUtils::ToIndex("outputs", 0));
  } else {
    stack_.func_end("outputs");
  }
  // end of inner class
  stack_.class_end();
  stack_.func_call(plugin->name, "op");
  for (const auto& attr : plugin->attrs) {
    stack_.call_arg(attr->name);
  }
  stack_.call_arg("name").call_arg("layouts").func_end("op").comment(GetPyComment(plugin), true);
}

void TorchPluginCodeGen::CodeGenConvertDepends() {
  BasePluginCodeGen<TorchPluginCodeGenConfig>::CodeGenConvertDepends();
  stack_.line("from torch import fx")
      .line("from tvm.relax.frontend.torch.fx_translator import TorchFXImporter")
      .line();
}

const String TorchPluginCodeGen::CodeGenOpConvert(const Plugin& plugin) {
  stack_.func_def(ConverterName(plugin), "relax.Var")
      .func_arg("node", "fx.node.Node")
      .func_arg("ctx", "TorchFXImporter")
      .func_start()
      .func_call("retrieve_args", "args", "ctx")
      .call_arg("node");
  Array<String> args;
  for (size_t i = 0; i < plugin->inputs.size(); i++) {
    const auto& tensor = plugin->inputs[i];
    stack_.assign(tensor->name, DocUtils::ToIndex("args", i + 1));
    args.push_back(tensor->name);
  }
  for (size_t i = 0; i < plugin->attrs.size(); i++) {
    const auto& attr = plugin->attrs[i];
    stack_.func_call("plugin_utils.to_expr", attr->name)
        .call_arg(DocUtils::ToIndex("args", i + plugin->inputs.size() + 1));
    args.push_back(attr->name);
  }
  stack_.assign("name",
                DocUtils::ToIndex("args", 1 + plugin->inputs.size() + plugin->attrs.size()));
  stack_.func_call("relax.Tuple", "args")
      .call_arg(DocUtils::ToList(args))
      .func_call("InferStructInfo" + plugin->name, "out_sinfo", "_plugin_api");
  for (const auto& t : plugin->inputs) {
    stack_.call_arg(t->name);
  }
  for (const auto& a : plugin->attrs) {
    stack_.call_arg(a->name);
  }
  stack_.func_call("call_dps_packed", "op")
      .call_arg(DocUtils::ToStr(plugin->name))
      .call_arg("args", "args")
      .call_arg("list(out_sinfo)", "out_sinfo")
      .func_call("msc_utils.set_expr_name", "op")
      .call_arg("op")
      .call_arg("name")
      .func_call("emit", "var", "ctx.block_builder")
      .call_arg("op")
      .call_arg("name");
  if (plugin->outputs.size() == 1) {
    stack_.func_end(DocUtils::ToList(Array<String>{"var"}));
  } else {
    Array<String> outputs;
    for (size_t i = 0; i < plugin->outputs.size(); i++) {
      const auto& tensor = plugin->outputs[i];
      stack_.func_call("relax.TupleGetItem", tensor->name).call_arg("var").call_arg(i);
      outputs.push_back(tensor->name);
    }
    stack_.func_end(DocUtils::ToList(outputs));
  }
  return EntryName(plugin);
}

void TorchPluginCodeGen::CodeGenMalloc(const Plugin& plugin, const Array<PluginTensor>& tensors,
                                       const String& collect) {
  Array<String> call_args{"input_metas", "meta_attr_", "true"};
  stack_.line().comment("malloc " + collect).declare("std::vector<MetaTensor>", collect + "_metas");
  CodeGenSafeCall(plugin->externs["infer_" + collect], call_args, collect + "_metas");
  for (size_t i = 0; i < tensors.size(); i++) {
    stack_.func_call("push_back", "", collect + "_tensors")
        .inplace_start("TorchUtils::MallocTorchTensor")
        .call_arg(DocUtils::ToIndex(collect + "_metas", i));
    int device_idx = plugin->FindDeviceRefIdx(tensors[i]);
    if (device_idx >= 0) {
      const auto& input_doc = DocUtils::ToIndex("input_tensors", device_idx);
      stack_.inplace_start("device", NullOpt, input_doc).inplace_end();
    } else {
      stack_.inplace_start("TorchUtils::ToTorchDevice")
          .call_arg(DocUtils::ToStr(tensors[i]->device))
          .inplace_end();
    }
    stack_.inplace_end();
  }
}

void TorchPluginCodeGen::CodeGenCompute(const Plugin& plugin, const String& device) {
  auto prepare_tensor = [this](const PluginTensor& tensor, const Map<String, String>& dtypes,
                               size_t idx, const String& collect) {
    const String& t_name = "d_" + tensor->name;
    const String& t_dtype = dtypes.count(tensor->name) ? dtypes[tensor->name] : tensor->dtype;
    const String& tensor_type = "DataTensor<" + t_dtype + ">";
    const String& anno = collect == "input" ? "const " + tensor_type + "&" : tensor_type;
    stack_.func_call("TorchUtils::To" + tensor_type, DocUtils::ToDeclare(anno, t_name))
        .call_arg(DocUtils::ToIndex(collect + "_tensors", idx))
        .call_arg(DocUtils::ToIndex(collect + "_metas", idx))
        .call_arg(collect == "input");
    return t_name;
  };

  if (plugin->externs.count(device + "_compute")) {
    for (const auto& dtypes : GetDtypeMatrix(plugin)) {
      const auto& tensor_dtypes = GetTensorDtypes(plugin, dtypes);
      Array<String> compute_args;
      String dtype_cond = "";
      for (size_t i = 0; i < plugin->inputs.size(); i++) {
        dtype_cond = dtype_cond + "input_metas[" + std::to_string(i) +
                     "].data_type() == DataUtils::ToMetaType(\"" + dtypes.at(i) + "\")";
        dtype_cond = dtype_cond + (i == plugin->inputs.size() - 1 ? "" : " && ");
      }
      // prepare compute datas
      stack_.cond_if(dtype_cond).comment("prepare compute datas");
      for (size_t i = 0; i < plugin->inputs.size(); i++) {
        const String& t_name = prepare_tensor(plugin->inputs[i], tensor_dtypes, i, "input");
        compute_args.push_back(t_name);
      }
      for (size_t i = 0; i < plugin->outputs.size(); i++) {
        const String& t_name = prepare_tensor(plugin->outputs[i], tensor_dtypes, i, "output");
        compute_args.push_back(t_name);
      }
      for (size_t i = 0; i < plugin->buffers.size(); i++) {
        const String& t_name = prepare_tensor(plugin->buffers[i], tensor_dtypes, i, "buffer");
        compute_args.push_back(t_name);
      }
      compute_args.push_back("meta_attr_");
      if (device == "cuda") {
        stack_.func_call("at::cuda::getCurrentCUDAStream",
                         DocUtils::ToDeclare("cudaStream_t", "stream"));
        compute_args.push_back("stream");
      }
      CodeGenSafeCall(plugin->externs[device + "_compute"], compute_args);
      stack_.cond_end();
    }
  } else {
    stack_.comment("Skip compute on " + device);
  }
}

TVM_REGISTER_GLOBAL("msc.plugin.GetTorchPluginSources")
    .set_body_typed([](const String& codegen_config, const String& print_config,
                       const String& codegen_type) -> Map<String, String> {
      TorchPluginCodeGen codegen = TorchPluginCodeGen(codegen_config);
      if (codegen_type == "build") {
        return codegen.GetBuildSources(print_config);
      }
      if (codegen_type == "manager") {
        return codegen.GetManagerSources(print_config);
      }
      return Map<String, String>();
    });

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
