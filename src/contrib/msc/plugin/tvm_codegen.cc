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
 * \file src/contrib/msc/plugin/tvm_codegen.cc
 */
#include "tvm_codegen.h"

namespace tvm {
namespace contrib {
namespace msc {

void TVMPluginCodeGen::CodeGenAttrDeclare(const Plugin& plugin) {
  BasePluginCodeGen<TVMPluginCodeGenConfig>::CodeGenAttrDeclare(plugin);
  const auto& attr_name = MetaAttrCls(plugin);
  // exprs to meta_attr
  stack_.comment("convert exprs to meta attrs method")
      .func_def(attr_name + "_from_exprs", "const " + attr_name);
  for (const auto& a : plugin->attrs) {
    const String& anno = IsListType(a->type) ? "Tuple" : "PrimValue";
    stack_.func_arg(a->name, "const " + anno + "&");
  }
  // args to meta_attr
  stack_.comment("convert args to meta attrs method")
      .func_def(attr_name + "_from_args", "const " + attr_name)
      .func_arg("args", "TVMArgs")
      .func_arg("pos", "size_t&");
}

void TVMPluginCodeGen::CodeGenAttrDefine(const Plugin& plugin) {
  const auto& attr_name = MetaAttrCls(plugin);
  // exprs to meta_attr
  stack_.func_def(attr_name + "_from_exprs", "const " + attr_name);
  for (const auto& a : plugin->attrs) {
    const String& anno = IsListType(a->type) ? "Tuple" : "PrimValue";
    stack_.func_arg(a->name, "const " + anno + "&");
  }
  stack_.func_start().declare(attr_name, "meta_attr");
  for (const auto& a : plugin->attrs) {
    const String& convert = IsListType(a->type) ? "AttrFromPrims" : "AttrFromPrim";
    stack_.func_call("TVMUtils::" + convert)
        .call_arg(a->name)
        .call_arg(DocUtils::ToAttrAccess("meta_attr", a->name));
  }
  stack_.func_end("meta_attr");
  // args to meta_attr
  stack_.comment("convert args to meta attrs method")
      .func_def(attr_name + "_from_args", "const " + attr_name)
      .func_arg("args", "TVMArgs")
      .func_arg("pos", "size_t&")
      .func_start()
      .declare(attr_name, "meta_attr");
  for (const auto& a : plugin->attrs) {
    if (IsListType(a->type)) {
      // TODO(meng.tong): support list atribute
      LOG_FATAL << "ListType argument is not supported for tvm runtime";
      stack_.func_call("TVMUtils::AttrFromArg", a->name + "_size")
          .call_arg(DocUtils::ToIndex("args", "pos"))
          .func_call("TVMUtils::AttrFromArgs")
          .call_arg("args")
          .call_arg("pos")
          .call_arg(a->name + "_size")
          .call_arg(DocUtils::ToAttrAccess("meta_attr", a->name))
          .assign("pos", "pos + 1 + " + a->name + "_size");
    } else {
      stack_.func_call("TVMUtils::AttrFromArg")
          .call_arg(DocUtils::ToIndex("args", "pos"))
          .call_arg(DocUtils::ToAttrAccess("meta_attr", a->name))
          .assign("pos", "pos + 1");
    }
  }
  stack_.func_end("meta_attr");
}

void TVMPluginCodeGen::CodeGenOpDeclare(const Plugin& plugin) {
  // infer struct info
  stack_.func_def("InferStructInfo" + plugin->name, "Array<TensorStructInfo>");
  for (const auto& t : plugin->inputs) {
    stack_.func_arg(t->name, "const Expr&");
  }
  for (const auto& a : plugin->attrs) {
    const String& anno = IsListType(a->type) ? "Tuple" : "PrimValue";
    stack_.func_arg(a->name, "const " + anno + "&");
  }
  // infer layout
  stack_.func_def("InferLayout" + plugin->name, "InferLayoutOutput")
      .func_arg("inputs", "const Array<Expr>&")
      .func_arg("var_layout_map", "const VarLayoutMap&");
}

void TVMPluginCodeGen::CodeGenOpDefine(const Plugin& plugin) {
  const auto& attr_name = MetaAttrCls(plugin);
  // infer struct info
  Array<String> infer_args{"input_metas", "meta_attr", "false"};
  stack_.func_def("InferStructInfo" + plugin->name, "Array<TensorStructInfo>");
  for (const auto& t : plugin->inputs) {
    stack_.func_arg(t->name, "const Expr&");
  }
  for (const auto& a : plugin->attrs) {
    const String& anno = IsListType(a->type) ? "Tuple" : "PrimValue";
    stack_.func_arg(a->name, "const " + anno + "&");
  }
  stack_.func_start()
      .comment("extract meta attrs")
      .func_call(attr_name + "_from_exprs", DocUtils::ToDeclare("const auto&", "meta_attr"));
  for (const auto& a : plugin->attrs) {
    stack_.call_arg(a->name);
  }
  stack_.comment("extract meta inputs").declare("std::vector<MetaTensor>", "input_metas");
  for (const auto& t : plugin->inputs) {
    stack_.func_call("push_back", "", "input_metas")
        .inplace_start("TVMUtils::ToMetaTensor")
        .call_arg(t->name)
        .inplace_end();
  }
  stack_.declare("std::vector<MetaTensor>", "output_metas");
  CodeGenSafeCall(plugin->externs["infer_output"], infer_args, "output_metas");
  stack_.declare("Array<TensorStructInfo>", "output_sinfo");
  for (size_t i = 0; i < plugin->outputs.size(); i++) {
    stack_.func_call("push_back", "", "output_sinfo")
        .inplace_start("TVMUtils::ToTensorStructInfo")
        .call_arg(DocUtils::ToIndex("output_metas", i));
    int device_idx = plugin->FindDeviceRefIdx(plugin->outputs[i]);
    if (device_idx >= 0) {
      stack_.call_arg(plugin->inputs[device_idx]->name);
    } else {
      stack_.inplace_start("TVMUtils::ToTVMDevice")
          .call_arg(plugin->outputs[i]->device)
          .inplace_end();
    }
    stack_.inplace_end();
  }
  stack_.func_end("output_sinfo");

  // infer layout
  stack_.func_def("InferLayout" + plugin->name, "InferLayoutOutput")
      .func_arg("inputs", "const Array<Expr>&")
      .func_arg("var_layout_map", "const VarLayoutMap&")
      .func_start()
      .comment("define attrs");
  for (size_t i = 0; i < plugin->attrs.size(); i++) {
    const auto& attr = plugin->attrs[i];
    const String& anno = IsListType(attr->type) ? "Tuple" : "PrimValue";
    stack_
        .func_call("Downcast<" + anno + ">",
                   DocUtils::ToDeclare("const auto&", "attr_" + attr->name))
        .call_arg(DocUtils::ToIndex("inputs", i + plugin->inputs.size()));
  }
  stack_.declare("Array<NLayout>", "arg_layouts")
      .declare("Array<NLayout>", "output_layouts")
      .comment("extract meta attrs")
      .func_call(attr_name + "_from_exprs", "const " + attr_name + "& meta_attr");
  for (const auto& a : plugin->attrs) {
    stack_.call_arg("attr_" + a->name);
  }
  stack_.comment("extract meta inputs")
      .declare("std::vector<MetaTensor>", "input_metas")
      .for_start("i", 0, plugin->inputs.size())
      .func_call("LayoutUtils::InferLayoutDecision",
                 DocUtils::ToDeclare("const auto&", "in_layout"))
      .call_arg(DocUtils::ToIndex("inputs", "i"))
      .call_arg("var_layout_map")
      .func_call("push_back", "", "arg_layouts")
      .call_arg("in_layout")
      .func_call("push_back", "", "input_metas")
      .inplace_start("TVMUtils::ToMetaTensor")
      .call_arg(DocUtils::ToIndex("inputs", "i"))
      .call_arg("in_layout")
      .inplace_end()
      .for_end()
      .comment("add fake layout for attrs")
      .for_start("i", 0, plugin->attrs.size())
      .func_call("push_back", "", "arg_layouts")
      .inplace_start("LayoutDecision")
      .call_arg(DocUtils::ToStr(""))
      .inplace_end()
      .for_end();
  stack_.declare("std::vector<MetaTensor>", "output_metas");
  CodeGenSafeCall(plugin->externs["infer_output"], infer_args, "output_metas");
  stack_.for_start("i", 0, plugin->outputs.size())
      .func_call("push_back", "", "output_layouts")
      .inplace_start("LayoutDecision")
      .call_arg(DocUtils::ToAttrAccess(DocUtils::ToIndex("output_metas", "i"), "layout_name()"))
      .inplace_end()
      .for_end()
      .declare("Array<NLayout>", "input_layouts")
      .func_call("push_back", "", "input_layouts")
      .inplace_start("LayoutDecision")
      .call_arg(DocUtils::ToStr(""))
      .inplace_end()
      .func_call("push_back", "", "input_layouts")
      .call_arg("arg_layouts")
      .func_call("InferLayoutOutput", DocUtils::ToDeclare("const auto&", "infer_output"))
      .call_arg("input_layouts")
      .call_arg("output_layouts")
      .call_arg("Attrs()");
  stack_.func_end("infer_output");

  // register funcs
  stack_.func_call("TVM_REGISTER_GLOBAL")
      .call_arg(DocUtils::ToStr("msc.plugin.op.InferStructInfo" + plugin->name))
      .method_call("set_body_typed")
      .call_arg("InferStructInfo" + plugin->name)
      .line()
      .func_call("TVM_REGISTER_GLOBAL")
      .call_arg(DocUtils::ToStr("msc.plugin.op.InferLayout" + plugin->name))
      .method_call("set_body_typed")
      .call_arg("InferLayout" + plugin->name)
      .line();
}

void TVMPluginCodeGen::CodeGenOpRuntime(const Plugin& plugin) {
  ICHECK(!plugin->externs.count("infer_buffer")) << "infer_buffer is not supported for tvm runtime";
  const auto& attr_name = MetaAttrCls(plugin);
  const auto& func_name = ComputeName(plugin);
  String device_cond = "";
  for (size_t i = 0; i < plugin->inputs.size(); i++) {
    String device_type = "";
    if (plugin->inputs[i]->device == "cuda" || plugin->inputs[i]->device == "default") {
      device_type = "DLDeviceType::kDLCUDA";
    } else {
      device_type = "DLDeviceType::kDLCPU";
    }
    device_cond = device_cond + "TVMUtils::OnDevice(" + plugin->inputs[i]->name + ", " +
                  device_type + ")" + (i == plugin->inputs.size() - 1 ? "" : " && ");
  }
  stack_.func_def(func_name).func_arg("args", "TVMArgs").func_arg("ret", "TVMRetValue*");
  stack_.func_start().comment("define tensors");
  for (size_t i = 0; i < plugin->inputs.size(); i++) {
    stack_.assign(plugin->inputs[i]->name, DocUtils::ToIndex("args", i), "DLTensor*");
  }
  stack_.comment("extract meta attrs")
      .assign("pos", plugin->inputs.size(), "size_t")
      .func_call(attr_name + "_from_args", "const " + attr_name + "& meta_attr")
      .call_arg("args")
      .call_arg("pos");
  for (size_t i = 0; i < plugin->outputs.size(); i++) {
    stack_.assign(plugin->outputs[i]->name, DocUtils::ToIndex("args", "pos + " + std::to_string(i)),
                  "DLTensor*");
  }
  stack_.comment("do the compute").cond_if(device_cond);
  CodeGenCompute(plugin, "cuda");
  stack_.cond_else();
  CodeGenCompute(plugin, "cpu");
  stack_.cond_end().func_end();
  // register the compute
  stack_.func_call("TVM_REGISTER_GLOBAL")
      .call_arg(DocUtils::ToStr(plugin->name))
      .method_call("set_body")
      .call_arg(func_name)
      .line();
}

void TVMPluginCodeGen::CodeGenCmake(const std::set<String>& devices) {
  Map<String, String> flags;
  flags.Set("PLUGIN_SUPPORT_TVM", "");
  CodeGenPreCmake(devices, flags);
  stack_.line("set(CMAKE_CXX_STANDARD 17)")
      .line("set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -Wno-macro-redefined\")")
      .line()
      .line("set(TVM_ROOT " + config()->tvm_root + ")")
      .line("find_library(TVM_LIB NAMES tvm HINTS ${TVM_ROOT}/build NO_DEFAULT_PATH)");
  Array<String> includes, libs;
  includes.push_back("${TVM_ROOT}/include");
  includes.push_back("${TVM_ROOT}/3rdparty/dmlc-core/include");
  includes.push_back("${TVM_ROOT}/3rdparty/dlpack/include");
  includes.push_back("${TVM_ROOT}/3rdparty/compiler-rt");
  libs.push_back("${TVM_LIB}");
  CodeGenPostCmake(devices, includes, libs);
}

void TVMPluginCodeGen::CodeGenManagerDepends() {
  BasePluginCodeGen<TVMPluginCodeGenConfig>::CodeGenManagerDepends();
  stack_.line("from tvm import relax")
      .line("from tvm.relax import call_dps_packed")
      .line("from tvm.contrib.msc.plugin import utils as plugin_utils")
      .line("from tvm.contrib.msc.core import utils as msc_utils")
      .line();
}

void TVMPluginCodeGen::CodeGenManagerMethods() {
  BasePluginCodeGen<TVMPluginCodeGenConfig>::CodeGenManagerMethods();
  stack_.func_def("setup")
      .func_arg("self", "object")
      .func_start()
      .for_start("lib", "os.listdir(self._lib_folder)")
      .assign("lib_file", "os.path.join(self._lib_folder, lib)")
      .func_call("CDLL", "", "ctypes")
      .call_arg("lib_file")
      .for_end()
      .line("from tvm.contrib.msc.plugin.op import _ffi_api")
      .assign(DocUtils::ToAttrAccess("self", "_ffi_api"), "_ffi_api")
      .func_end();
}

void TVMPluginCodeGen::CodeGenOpBuilder(const Plugin& plugin) {
  stack_.func_def(plugin->name).func_arg("self", "object");
  for (const auto& t : plugin->inputs) {
    stack_.func_arg(t->name, "relax.Expr");
  }
  for (const auto& attr : plugin->attrs) {
    stack_.func_arg(attr->name, ToPyType(attr->type), attr->default_value);
  }
  stack_.func_arg("name", "str", "\"" + plugin->name + "\"").func_start();
  Array<String> args;
  for (const auto& t : plugin->inputs) {
    args.push_back(t->name);
  }
  for (const auto& a : plugin->attrs) {
    stack_.func_call("plugin_utils.to_expr", a->name).call_arg(a->name);
    args.push_back(a->name);
  }
  stack_.func_call("relax.Tuple", "args")
      .call_arg(DocUtils::ToList(args))
      .func_call("InferStructInfo" + plugin->name, "out_sinfo", "self._ffi_api");
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
      .call_arg("name");
  stack_.func_end("op").comment(GetPyComment(plugin), true);
}

void TVMPluginCodeGen::CodeGenCompute(const Plugin& plugin, const String& device) {
  if (plugin->externs.count(device + "_compute")) {
    // compute with dtype
    auto prepare_tensor = [this](const PluginTensor& tensor, const Map<String, String>& dtypes,
                                 size_t idx, const String& collect) {
      const String& t_name = "d_" + tensor->name;
      const String& t_dtype = dtypes.count(tensor->name) ? dtypes[tensor->name] : tensor->dtype;
      const String& tensor_type = "DataTensor<" + t_dtype + ">";
      const String& anno = collect == "input" ? "const " + tensor_type + "&" : tensor_type;
      stack_.func_call("TVMUtils::To" + tensor_type, DocUtils::ToDeclare(anno, t_name))
          .call_arg(tensor->name)
          .call_arg(collect == "input");
      return t_name;
    };
    for (const auto& dtypes : GetDtypeMatrix(plugin)) {
      const auto& tensor_dtypes = GetTensorDtypes(plugin, dtypes);
      Array<String> compute_args;
      String dtype_cond = "";
      for (size_t i = 0; i < plugin->inputs.size(); i++) {
        const auto& t_name = plugin->inputs[i]->name;
        dtype_cond = dtype_cond + "TVMUtils::ToMetaType(" + t_name +
                     "->dtype) == DataUtils::ToMetaType(\"" + dtypes.at(i) + "\")";
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
      ICHECK(plugin->buffers.size() == 0) << "Plugin with buffers is not supported in tvm";
      compute_args.push_back("meta_attr");
      if (device == "cuda") {
        stack_.assign("stream", "runtime::CUDAThreadEntry::ThreadLocal()->stream", "auto");
        compute_args.push_back("stream");
      }
      CodeGenSafeCall(plugin->externs[device + "_compute"], compute_args);
      stack_.cond_end();
    }
  } else {
    stack_.comment("Skip compute on " + device);
  }
}

TVM_REGISTER_GLOBAL("msc.plugin.GetTVMPluginSources")
    .set_body_typed([](const String& codegen_config, const String& print_config,
                       const String& codegen_type) -> Map<String, String> {
      TVMPluginCodeGen codegen = TVMPluginCodeGen(codegen_config);
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
