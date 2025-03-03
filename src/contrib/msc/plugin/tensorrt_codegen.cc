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
 * \file src/contrib/msc/plugin/tensorrt_codegen.cc
 */
#include "tensorrt_codegen.h"

#include <set>
namespace tvm {
namespace contrib {
namespace msc {

void TensorRTPluginCodeGen::CodeGenAttrDeclare(const Plugin& plugin) {
  BasePluginCodeGen<TensorRTPluginCodeGenConfig>::CodeGenAttrDeclare(plugin);
  const auto& attr_name = MetaAttrCls(plugin);
  // serialize size for attr
  stack_.comment("serialize size").func_def(attr_name + "_serialize_size", "size_t");
  // serialize method for attr
  stack_.comment("serialize method")
      .func_def(attr_name + "_serialize", "char*")
      .func_arg("meta_attr", "const " + attr_name + "&")
      .func_arg("buffer", "char*");
  // deserialize method for attr
  stack_.comment("deserialize method")
      .func_def(attr_name + "_deserialize", "const char*")
      .func_arg("meta_attr", attr_name + "&")
      .func_arg("buffer", "const char*");
  // attr to field
  stack_.comment("meta attr to field")
      .func_def(attr_name + "_to_fields")
      .func_arg("fields", "std::vector<PluginField>&");
  // attr from field
  stack_.comment("meta attr from field")
      .func_def(attr_name + "_from_fields", "const " + attr_name)
      .func_arg("fields", "const PluginField*");
}

void TensorRTPluginCodeGen::CodeGenAttrDefine(const Plugin& plugin) {
  const auto& attr_name = MetaAttrCls(plugin);
  // serialize size for attr
  stack_.func_def(attr_name + "_serialize_size", "size_t").func_start().assign("size", 0, "size_t");
  for (const auto& a : plugin->attrs) {
    stack_.comment("attr " + a->name + "(" + a->type + ")");
    if (IsListType(a->type)) {
      LOG_FATAL << "attribute type " << a->type << " is not supported";
      const auto& ele_type = GetEleType(a->type);
      stack_.assign("size", "size + sizeof(size_t)")
          .for_start("a", DocUtils::ToAttrAccess("meta_attr", a->name))
          .assign("size", "size + sizeof(" + ToCppType(ele_type) + ")")
          .for_end();
    } else {
      stack_.assign("size", "size + sizeof(" + ToCppType(a->type) + ")");
    }
  }
  stack_.func_end("size");
  // serialize method for attr
  stack_.func_def(attr_name + "_serialize", "char*")
      .func_arg("meta_attr", "const " + attr_name + "&")
      .func_arg("buffer", "char*")
      .func_start()
      .assign("start", "buffer", "const char*");
  for (const auto& a : plugin->attrs) {
    stack_.func_call("TRTUtils::ValToBuffer")
        .call_arg("buffer")
        .call_arg(DocUtils::ToAttrAccess("meta_attr", a->name));
  }
  stack_.func_call(attr_name + "_serialize_size", DocUtils::ToDeclare("size_t", "expected"))
      .line("assert(buffer == start + expected);")
      .func_end("buffer");
  // deserialize method for attr
  stack_.func_def(attr_name + "_deserialize", "const char*")
      .func_arg("meta_attr", attr_name + "&")
      .func_arg("buffer", "const char*")
      .func_start()
      .assign("start", "buffer", "const char*");
  for (const auto& a : plugin->attrs) {
    stack_.func_call("TRTUtils::ValFromBuffer")
        .call_arg("buffer")
        .call_arg(DocUtils::ToAttrAccess("meta_attr", a->name));
  }
  stack_.func_call(attr_name + "_serialize_size", DocUtils::ToDeclare("size_t", "expected"))
      .line("assert(buffer == start + expected);")
      .func_end("buffer");
  // attr to field
  stack_.func_def(attr_name + "_to_fields")
      .func_arg("fields", "std::vector<PluginField>&")
      .func_start();
  for (const auto& a : plugin->attrs) {
    stack_.func_call("emplace_back", "", "fields")
        .inplace_start("TRTUtils::ToField")
        .call_arg(DocUtils::ToStr(a->name))
        .call_arg(DocUtils::ToStr(a->type))
        .inplace_end();
  }
  stack_.func_end();
  // attr from field
  stack_.func_def(attr_name + "_from_fields", "const " + attr_name)
      .func_arg("fields", "const PluginField*")
      .func_start()
      .declare(attr_name, "meta_attr")
      .for_start("i", 0, plugin->attrs.size());
  for (size_t i = 0; i < plugin->attrs.size(); i++) {
    const auto& attr = plugin->attrs[i];
    const String& cond = "strcmp(fields[i].name, \"" + attr->name + "\") == 0";
    if (i == 0) {
      stack_.switch_start(cond);
    } else {
      stack_.switch_case(cond);
    }
    stack_.func_call("TRTUtils::FromField")
        .call_arg(DocUtils::ToIndex("fields", "i"))
        .call_arg(DocUtils::ToAttrAccess("meta_attr", attr->name));
  }
  stack_.switch_end().for_end().func_end("meta_attr");
}

void TensorRTPluginCodeGen::CodeGenOpHeader(const Plugin& plugin) {
  BasePluginCodeGen<TensorRTPluginCodeGenConfig>::CodeGenOpHeader(plugin);
  stack_.line("using namespace nvinfer1;").line();
}

void TensorRTPluginCodeGen::CodeGenOpDeclare(const Plugin& plugin) {
  if (!IsMixPrecision(plugin)) {
    // static plugin op
    const auto& op_static = OpCls(plugin, false);
    stack_.class_def(op_static + " :  public IPluginV2").class_start().scope_start("public:");
    CodegenOpCommonMethods(plugin, false, true);
    stack_.comment("special methods for " + op_static)
        .func_def("getOutputDimensions", "Dims")
        .func_decorator("noexcept override")
        .func_arg("index", "int")
        .func_arg("in_dims", "const Dims*")
        .func_arg("n_inputs", "int")
        .func_def("configureWithFormat")
        .func_decorator("noexcept override")
        .func_arg("in_dims", "const Dims*")
        .func_arg("n_inputs", "int")
        .func_arg("out_dims", "const Dims*")
        .func_arg("n_outputs", "int")
        .func_arg("dtype", "DataType")
        .func_arg("format", "PluginFormat")
        .func_arg("max_batch", "int")
        .func_def("supportsFormat", "bool")
        .func_decorator("const noexcept override")
        .func_arg("dtype", "DataType")
        .func_arg("format", "PluginFormat")
        .func_def("getWorkspaceSize", "size_t")
        .func_decorator("const noexcept override")
        .func_arg("max_batch", "int")
        .func_def("enqueue", "int")
        .func_decorator("noexcept override")
        .func_arg("batch_size", "int")
        .func_arg("inputs", "const void* const*")
        .func_arg("outputs", "void* const*")
        .func_arg("workspace", "void*")
        .func_arg("stream", "cudaStream_t")
        .scope_end();
    CodegenOpMembers(plugin, false);
    stack_.class_end();

    // static plugin creator
    CodegenCreator(plugin, false, true);
  }
  // dynamic plugin op
  const auto& op_dynamic = OpCls(plugin, true);
  stack_.class_def(op_dynamic + " :  public IPluginV2DynamicExt")
      .class_start()
      .scope_start("public:");
  CodegenOpCommonMethods(plugin, true, true);
  stack_.comment("special methods for " + op_dynamic)
      .func_def("getOutputDataType", "DataType")
      .func_decorator("const noexcept override")
      .func_arg("index", "int")
      .func_arg("in_types", "const DataType*")
      .func_arg("n_inputs", "int")
      .func_def("getOutputDimensions", "DimsExprs")
      .func_decorator("noexcept override")
      .func_arg("index", "int")
      .func_arg("in_dims", "const DimsExprs*")
      .func_arg("n_inputs", "int")
      .func_arg("builder", "IExprBuilder&")
      .func_def("configurePlugin")
      .func_decorator("noexcept override")
      .func_arg("in_descs", "const DynamicPluginTensorDesc*")
      .func_arg("n_inputs", "int")
      .func_arg("out_descs", "const DynamicPluginTensorDesc*")
      .func_arg("n_outputs", "int")
      .func_def("supportsFormatCombination", "bool")
      .func_decorator("noexcept override")
      .func_arg("pos", "int")
      .func_arg("io_desc", "const PluginTensorDesc*")
      .func_arg("n_inputs", "int")
      .func_arg("n_outputs", "int")
      .func_def("getWorkspaceSize", "size_t")
      .func_decorator("const noexcept override")
      .func_arg("in_descs", "const PluginTensorDesc*")
      .func_arg("n_inputs", "int")
      .func_arg("out_descs", "const PluginTensorDesc*")
      .func_arg("n_outputs", "int")
      .func_def("enqueue", "int")
      .func_decorator("noexcept override")
      .func_arg("input_descs", "const PluginTensorDesc*")
      .func_arg("output_descs", "const PluginTensorDesc*")
      .func_arg("inputs", "const void* const*")
      .func_arg("outputs", "void* const*")
      .func_arg("workspace", "void*")
      .func_arg("stream", "cudaStream_t")
      .scope_end();
  CodegenOpMembers(plugin, true);
  stack_.class_end();

  // dynamic plugin creator
  CodegenCreator(plugin, true, true);
}

void TensorRTPluginCodeGen::CodeGenOpDefine(const Plugin& plugin) {
  if (!IsMixPrecision(plugin)) {
    // static op
    const auto& op_static = OpCls(plugin, false);
    CodegenOpCommonMethods(plugin, false, false);
    // getOutputDimensions
    stack_.func_def(op_static + "::getOutputDimensions", "Dims")
        .func_decorator("noexcept")
        .func_arg("index", "int")
        .func_arg("in_dims", "const Dims*")
        .func_arg("n_inputs", "int")
        .func_start();
    CodegenOutputInfer(plugin, false);
    stack_
        .func_call("shape", DocUtils::ToDeclare("MetaShape", "out_shape"),
                   DocUtils::ToIndex("output_metas_", "index"))
        .func_call("TRTUtils::ToDims", DocUtils::ToDeclare("Dims", "out_dims"))
        .call_arg("out_shape")
        .func_end("out_dims");
    // configureWithFormat
    stack_.func_def(op_static + "::configureWithFormat")
        .func_decorator("noexcept")
        .func_arg("in_dims", "const Dims*")
        .func_arg("n_inputs", "int")
        .func_arg("out_dims", "const Dims*")
        .func_arg("n_outputs", "int")
        .func_arg("dtype", "DataType")
        .func_arg("format", "PluginFormat")
        .func_arg("max_batch", "int")
        .func_start()
        .assign("dtype_", "dtype")
        .line("assert(n_outputs == " + std::to_string(plugin->outputs.size()) + ");");
    CodegenOutputInfer(plugin, false);
    stack_.func_end();
    // supportsFormat
    stack_.func_def(op_static + "::supportsFormat", "bool")
        .func_decorator("const noexcept")
        .func_arg("dtype", "DataType")
        .func_arg("format", "PluginFormat")
        .func_start()
        .declare("bool", "support");
    size_t cnt = 0;
    for (const auto& dtypes : GetDtypeMatrix(plugin)) {
      const String& cond = "dtype_ == TRTUtils::ToDataType(\"" + dtypes.at(0) + "\")";
      if (cnt == 0) {
        stack_.switch_start(cond);
      } else {
        stack_.switch_case(cond);
      }
      stack_.assign("support", true);
      cnt++;
    }
    stack_.switch_case().assign("support", false).switch_end().func_end("support");
    // getWorkspaceSize
    stack_.func_def(op_static + "::getWorkspaceSize", "size_t")
        .func_decorator("const noexcept")
        .func_arg("max_batch", "int")
        .func_start()
        .assign("size", 0, "size_t");
    if (plugin->externs.count("infer_buffer")) {
      CodegenBufferInfer(plugin);
    }
    stack_.func_end("size");
    // enqueue
    stack_.func_def(op_static + "::enqueue", "int")
        .func_decorator("noexcept")
        .func_arg("batch_size", "int")
        .func_arg("inputs", "const void* const*")
        .func_arg("outputs", "void* const*")
        .func_arg("workspace", "void*")
        .func_arg("stream", "cudaStream_t")
        .func_start();
    CodegenEnqueue(plugin, false);
    stack_.func_end(0);

    // static creator
    CodegenCreator(plugin, false, false);
  }
  // dynamic op
  const auto& op_dynamic = OpCls(plugin, true);
  CodegenOpCommonMethods(plugin, true, false);
  // getOutputDataType
  stack_.func_def(op_dynamic + "::getOutputDataType", "DataType")
      .func_decorator("const noexcept")
      .func_arg("index", "int")
      .func_arg("in_types", "const DataType*")
      .func_arg("n_inputs", "int")
      .func_start()
      .declare("DataType", "dtype");
  for (size_t i = 0; i < plugin->outputs.size(); i++) {
    if (i == 0) {
      stack_.switch_start("index == " + std::to_string(i));
    } else {
      stack_.switch_case("index == " + std::to_string(i));
    }
    int ref = plugin->FindDtypeRefIdx(plugin->outputs[i]);
    if (ref >= 0) {
      stack_.assign("dtype", DocUtils::ToIndex("in_types", ref));
    } else {
      stack_.func_call("TRTUtils::ToDataType", "dtype")
          .call_arg(DocUtils::ToStr(plugin->outputs[i]->dtype));
    }
  }
  stack_.switch_end().func_end("dtype");
  // getOutputDimensions
  stack_.func_def(op_dynamic + "::getOutputDimensions", "DimsExprs")
      .func_decorator("noexcept")
      .func_arg("index", "int")
      .func_arg("in_dims", "const DimsExprs*")
      .func_arg("n_inputs", "int")
      .func_arg("builder", "IExprBuilder&")
      .func_start();
  CodegenOutputInfer(plugin, false);
  stack_
      .func_call("shape", DocUtils::ToDeclare("MetaShape", "out_shape"),
                 DocUtils::ToIndex("output_metas_", "index"))
      .func_call("TRTUtils::ToDimsExprs", DocUtils::ToDeclare("DimsExprs", "out_dims"))
      .call_arg("out_shape")
      .call_arg("builder")
      .func_end("out_dims");
  // configurePlugin
  stack_.func_def(op_dynamic + "::configurePlugin")
      .func_decorator("noexcept")
      .func_arg("in_descs", "const DynamicPluginTensorDesc*")
      .func_arg("n_inputs", "int")
      .func_arg("out_descs", "const DynamicPluginTensorDesc*")
      .func_arg("n_outputs", "int")
      .func_start()
      .line("assert(n_outputs == " + std::to_string(plugin->outputs.size()) + ");");
  CodegenOutputInfer(plugin, true);
  stack_.func_end();
  // supportsFormatCombination
  stack_.func_def(op_dynamic + "::supportsFormatCombination", "bool")
      .func_decorator("noexcept")
      .func_arg("pos", "int")
      .func_arg("io_desc", "const PluginTensorDesc*")
      .func_arg("n_inputs", "int")
      .func_arg("n_outputs", "int")
      .func_start()
      .declare("bool", "support");
  size_t cnt = 0;
  for (const auto& dtypes : GetDtypeMatrix(plugin)) {
    String cond;
    for (size_t i = 0; i < plugin->inputs.size(); i++) {
      cond = cond + "io_desc[" + std::to_string(i) + "].type == TRTUtils::ToDataType(\"" +
             dtypes.at(i) + "\")";
      cond = cond + (i == plugin->inputs.size() - 1 ? "" : " && ");
    }
    if (cnt == 0) {
      stack_.switch_start(cond);
    } else {
      stack_.switch_case(cond);
    }
    stack_.assign("support", true);
    cnt++;
  }
  stack_.switch_case().assign("support", false).switch_end().func_end("support");
  // getWorkspaceSize
  stack_.func_def(op_dynamic + "::getWorkspaceSize", "size_t")
      .func_decorator("const noexcept")
      .func_arg("in_descs", "const PluginTensorDesc*")
      .func_arg("n_inputs", "int")
      .func_arg("out_descs", "const PluginTensorDesc*")
      .func_arg("n_outputs", "int")
      .func_start()
      .assign("size", 0, "size_t");
  if (plugin->externs.count("infer_buffer")) {
    CodegenBufferInfer(plugin);
  }
  stack_.func_end("size");
  // enqueue
  stack_.func_def(op_dynamic + "::enqueue", "int")
      .func_decorator("noexcept")
      .func_arg("input_descs", "const PluginTensorDesc*")
      .func_arg("output_descs", "const PluginTensorDesc*")
      .func_arg("inputs", "const void* const*")
      .func_arg("outputs", "void* const*")
      .func_arg("workspace", "void*")
      .func_arg("stream", "cudaStream_t")
      .func_start();
  CodegenEnqueue(plugin, true);
  stack_.func_end(0);

  // dynamic creator
  CodegenCreator(plugin, true, false);
}

void TensorRTPluginCodeGen::CodeGenCmake(const std::set<String>& devices) {
  Map<String, String> flags;
  flags.Set("PLUGIN_SUPPORT_TENSORRT", "");
  flags.Set("TRT_MAJOR", std::to_string(config()->version[0]));
  flags.Set("TRT_MINOR", std::to_string(config()->version[1]));
  flags.Set("TRT_PATCH", std::to_string(config()->version[2]));
  CodeGenPreCmake(devices, flags);
  stack_
      .line("find_path(TRT_INCLUDE_DIR NvInfer.h HINTS " + config()->tensorrt_root +
            " PATH_SUFFIXES include)")
      .line("find_library(TRT_LIBS nvinfer HINTS " + config()->tensorrt_root +
            " PATH_SUFFIXES lib)")
      .line("set(CMAKE_CXX_FLAGS \"${CMAKE_CXX_FLAGS} -Wno-terminate\")");
  Array<String> includes, libs;
  includes.push_back("${TRT_INCLUDE_DIR}");
  libs.push_back("${TRT_LIBS}");
  CodeGenPostCmake(devices, includes, libs);
}

void TensorRTPluginCodeGen::CodeGenManagerMethods() {
  BasePluginCodeGen<TensorRTPluginCodeGenConfig>::CodeGenManagerMethods();
  stack_.func_def("setup")
      .func_arg("self", "object")
      .func_start()
      .for_start("lib", "os.listdir(self._lib_folder)")
      .assign("lib_file", "os.path.join(self._lib_folder, lib)")
      .func_call("CDLL", "", "ctypes")
      .call_arg("lib_file")
      .for_end()
      .func_end();
}

void TensorRTPluginCodeGen::CodegenOpCommonMethods(const Plugin& plugin, bool dynamic,
                                                   bool in_declare) {
  const auto& op_cls = OpCls(plugin, dynamic);
  const String& plugin_cls = dynamic ? "IPluginV2DynamicExt" : "IPluginV2";
  if (in_declare) {
    stack_.comment("common methods for " + op_cls);
    stack_.constructor_def(op_cls).constructor_arg("name", "const std::string&");
    for (const auto& a : plugin->attrs) {
      stack_.constructor_arg(a->name, "const " + ToCppType(a->type) + "&");
    }
    stack_.constructor_arg("layouts", "const std::vector<std::string>&")
        .constructor_def(op_cls)
        .constructor_arg("name", "const std::string&")
        .constructor_arg("buffer", "const void*")
        .constructor_arg("length", "size_t")
        .assign(op_cls + "()", "delete")
        .line()
        .constructor_def("~" + op_cls)
        .func_def("getSerializationSize", "size_t")
        .func_decorator("const noexcept override")
        .func_def("serialize")
        .func_decorator("const noexcept override")
        .func_arg("buffer", "void*")
        .func_def("getPluginType", "const char*")
        .func_decorator("const noexcept override")
        .func_def("getPluginVersion", "const char*")
        .func_decorator("const noexcept override")
        .func_def("getPluginNamespace", "const char*")
        .func_decorator("const noexcept override")
        .func_def("getNbOutputs", "int")
        .func_decorator("const noexcept override")
        .func_def("setPluginNamespace")
        .func_decorator("noexcept override")
        .func_arg("name_space", "const char*")
        .func_def("initialize", "int")
        .func_decorator("noexcept override")
        .func_def("terminate")
        .func_decorator("noexcept override")
        .func_def("destroy")
        .func_decorator("noexcept override")
        .func_def("clone", plugin_cls + "*")
        .func_decorator("const noexcept override");
  } else {
    const auto& attr_name = MetaAttrCls(plugin);
    // constructor from attrs
    stack_.constructor_def(op_cls + "::" + op_cls).constructor_arg("name", "const std::string&");
    for (const auto& a : plugin->attrs) {
      stack_.constructor_arg(a->name, "const " + ToCppType(a->type) + "&");
    }
    stack_.constructor_arg("layouts", "const std::vector<std::string>&")
        .constructor_start()
        .assign("name_", "name");
    for (const auto& a : plugin->attrs) {
      stack_.assign(DocUtils::ToAttrAccess("meta_attr_", a->name), a->name);
    }
    stack_.line("assert(layouts.size() == " + std::to_string(plugin->inputs.size()) + ");")
        .assign("layouts_", "layouts");
    stack_.constructor_end();
    // constructor from data
    stack_.constructor_def(op_cls + "::" + op_cls)
        .constructor_arg("name", "const std::string&")
        .constructor_arg("buffer", "const void*")
        .constructor_arg("length", "size_t")
        .constructor_start()
        .assign("name_", "name")
        .func_call("static_cast<const char*>", DocUtils::ToDeclare("const char*", "char_buf"))
        .call_arg("buffer")
        .assign("start_buf", "char_buf", "const char*")
        .func_call(attr_name + "_deserialize", "char_buf")
        .call_arg("meta_attr_")
        .call_arg("char_buf")
        .func_call("TRTUtils::ValFromBuffer")
        .call_arg("char_buf")
        .call_arg("dtype_")
        .func_call("TRTUtils::ValFromBuffer")
        .call_arg("char_buf")
        .call_arg("layouts_")
        .line("assert(layouts_.size() == " + std::to_string(plugin->inputs.size()) + ");")
        .line("assert(char_buf == (start_buf + length));")
        .constructor_end();
    // deconstructor
    stack_.constructor_def(op_cls + "::~" + op_cls)
        .constructor_start()
        .comment("ignore deconstruct of " + op_cls)
        .constructor_end();
    // getSerializationSize
    stack_.func_def(op_cls + "::getSerializationSize", "size_t")
        .func_decorator("const noexcept")
        .func_start()
        .assign("size", attr_name + "_serialize_size()", "size_t")
        .assign("size", "size + sizeof(dtype_)")
        .assign("size", "size + sizeof(size_t)")
        .for_start("layout", "layouts_")
        .assign("size", "size + sizeof(size_t) + layout.size() * sizeof(char)")
        .for_end()
        .func_end("size");
    // serialize
    stack_.func_def(op_cls + "::serialize")
        .func_decorator("const noexcept")
        .func_arg("buffer", "void*")
        .func_start()
        .func_call("static_cast<char*>", DocUtils::ToDeclare("char*", "char_buf"))
        .call_arg("buffer")
        .assign("start_buf", "char_buf", "const char*")
        .func_call(attr_name + "_serialize", "char_buf")
        .call_arg("meta_attr_")
        .call_arg("char_buf")
        .func_call("TRTUtils::ValToBuffer")
        .call_arg("char_buf")
        .call_arg("dtype_")
        .func_call("TRTUtils::ValToBuffer")
        .call_arg("char_buf")
        .call_arg("layouts_")
        .line("assert(char_buf == (start_buf + getSerializationSize()));")
        .func_end();
    // getPluginType
    const String& plugin_type = plugin->name + (dynamic ? "_dynamic" : "");
    stack_.func_def(op_cls + "::getPluginType", "const char*")
        .func_decorator("const noexcept")
        .func_start()
        .func_end(DocUtils::ToStr(plugin_type));
    // getPluginVersion
    stack_.func_def(op_cls + "::getPluginVersion", "const char*")
        .func_decorator("const noexcept")
        .func_start()
        .func_end(DocUtils::ToStr("1"));
    // getPluginNamespace
    stack_.func_def(op_cls + "::getPluginNamespace", "const char*")
        .func_decorator("const noexcept")
        .func_start()
        .func_call("c_str", DocUtils::ToDeclare("const char*", "name"),
                   DocUtils::ToDoc("name_space_"))
        .func_end("name");
    // getNbOutputs
    stack_.func_def(op_cls + "::getNbOutputs", "int")
        .func_decorator("const noexcept")
        .func_start()
        .func_end(plugin->outputs.size());
    // setPluginNamespace
    stack_.func_def(op_cls + "::setPluginNamespace")
        .func_decorator("noexcept")
        .func_arg("name_space", "const char*")
        .func_start()
        .assign("name_space_", "name_space")
        .func_end();
    // initialize
    stack_.func_def(op_cls + "::initialize", "int")
        .func_decorator("noexcept")
        .func_start()
        .func_end(0);
    // terminate
    stack_.func_def(op_cls + "::terminate")
        .func_decorator("noexcept")
        .func_start()
        .comment("Ignore teminate for " + plugin->name)
        .func_end();
    // destroy
    stack_.func_def(op_cls + "::destroy")
        .func_decorator("noexcept")
        .func_start()
        .line("delete this;")
        .func_end();
    // clone
    stack_.func_def(op_cls + "::clone", plugin_cls + "*")
        .func_decorator("const noexcept")
        .func_start()
        .func_call("new " + op_cls, DocUtils::ToDeclare(plugin_cls + "*", "plugin"))
        .call_arg("name_");
    for (const auto& a : plugin->attrs) {
      stack_.call_arg(DocUtils::ToAttrAccess("meta_attr_", a->name));
    }
    stack_.call_arg("layouts_").func_end("plugin");
  }
}

void TensorRTPluginCodeGen::CodegenOpMembers(const Plugin& plugin, bool dynamic) {
  stack_.scope_start("private:")
      .declare("std::string", "name_")
      .declare("std::string", "name_space_")
      .declare("DataType", "dtype_", 0, false)
      .declare_arg("DataType::kFLOAT")
      .declare(MetaAttrCls(plugin), "meta_attr_")
      .declare("std::vector<std::string>", "layouts_")
      .declare("std::vector<MetaTensor>", "input_metas_")
      .declare("std::vector<MetaTensor>", "output_metas_");
  if (plugin->externs.count("infer_buffer")) {
    stack_.declare("std::vector<MetaTensor>", "buffer_metas_");
  }
  stack_.scope_end().line();
}

void TensorRTPluginCodeGen::CodegenCreator(const Plugin& plugin, bool dynamic, bool in_declare) {
  const auto& creator_cls = CreatorCls(plugin, dynamic);
  const String& plugin_cls = dynamic ? "IPluginV2DynamicExt" : "IPluginV2";
  if (in_declare) {
    stack_.class_def(creator_cls + " :  public IPluginCreator")
        .class_start()
        .scope_start("public:")
        .constructor_def(creator_cls)
        .func_def("getPluginName", "const char*")
        .func_decorator("const noexcept override")
        .func_def("getPluginVersion", "const char*")
        .func_decorator("const noexcept override")
        .func_def("getPluginNamespace", "const char*")
        .func_decorator("const noexcept override")
        .func_def("getFieldNames", "const PluginFieldCollection*")
        .func_decorator("noexcept override")
        .func_def("setPluginNamespace")
        .func_decorator("noexcept override")
        .func_arg("name_space", "const char*")
        .func_def("createPlugin", plugin_cls + "*")
        .func_decorator("noexcept override")
        .func_arg("name", "const char*")
        .func_arg("collection", "const PluginFieldCollection*")
        .func_def("deserializePlugin", plugin_cls + "*")
        .func_decorator("noexcept override")
        .func_arg("name", "const char*")
        .func_arg("data", "const void*")
        .func_arg("length", "size_t")
        .scope_end()
        .scope_start("private:")
        .declare("static PluginFieldCollection", "collection_")
        .declare("static std::vector<PluginField>", "fields_")
        .declare("std::string", "name_space_")
        .scope_end()
        .line()
        .class_end();
  } else {
    const String& attr_name = MetaAttrCls(plugin);
    // static members
    stack_.comment("static members and register for " + plugin->name)
        .declare("PluginFieldCollection", creator_cls + "::collection_")
        .declare("std::vector<PluginField>", creator_cls + "::fields_")
        .func_call("REGISTER_TENSORRT_PLUGIN")
        .call_arg(creator_cls)
        .line();
    // constructor
    stack_.constructor_def(creator_cls + "::" + creator_cls)
        .constructor_start()
        .func_call(attr_name + "_to_fields")
        .call_arg("fields_");
    for (const auto& t : plugin->inputs) {
      stack_.func_call("emplace_back", "", "fields_")
          .inplace_start("TRTUtils::ToField")
          .call_arg(DocUtils::ToStr("layout_" + t->name))
          .call_arg(DocUtils::ToStr("string"))
          .inplace_end();
    }
    const auto& nb_fields_doc = DocUtils::ToAttrAccess("collection_", "nbFields");
    const auto& fields_doc = DocUtils::ToAttrAccess("collection_", "fields");
    stack_.func_call("size", nb_fields_doc, DocUtils::ToDoc("fields_"))
        .func_call("data", fields_doc, DocUtils::ToDoc("fields_"))
        .constructor_end();
    // getPluginName
    const String& plugin_type = plugin->name + (dynamic ? "_dynamic" : "");
    stack_.func_def(creator_cls + "::getPluginName", "const char*")
        .func_decorator("const noexcept")
        .func_start()
        .func_end(DocUtils::ToStr(plugin_type));
    // getPluginVersion
    stack_.func_def(creator_cls + "::getPluginVersion", "const char*")
        .func_decorator("const noexcept")
        .func_start()
        .func_end(DocUtils::ToStr("1"));
    // getPluginNamespace
    stack_.func_def(creator_cls + "::getPluginNamespace", "const char*")
        .func_decorator("const noexcept")
        .func_start()
        .func_call("c_str", DocUtils::ToDeclare("const char*", "name"),
                   DocUtils::ToDoc("name_space_"))
        .func_end("name");
    // getFieldNames
    stack_.func_def(creator_cls + "::getFieldNames", "const PluginFieldCollection*")
        .func_decorator("noexcept")
        .func_start()
        .func_end("&collection_");
    // setPluginNamespace
    stack_.func_def(creator_cls + "::setPluginNamespace")
        .func_decorator("noexcept")
        .func_arg("name_space", "const char*")
        .func_start()
        .assign("name_space_", "name_space")
        .func_end();
    // createPlugin
    size_t fields_size = plugin->attrs.size() + plugin->inputs.size();
    const auto& op_cls = OpCls(plugin, dynamic);
    stack_.func_def(creator_cls + "::createPlugin", plugin_cls + "*")
        .func_decorator("noexcept")
        .func_arg("name", "const char*")
        .func_arg("collection", "const PluginFieldCollection*")
        .func_start()
        .line("assert(collection->nbFields == " + std::to_string(fields_size) + ");")
        .assign("fields", DocUtils::ToAttrAccess(DocUtils::ToPtr("collection"), "fields"),
                "const PluginField*")
        .func_call(attr_name + "_from_fields", DocUtils::ToDeclare("const auto&", "meta_attr"))
        .call_arg("fields")
        .declare("std::vector<std::string>", "layouts")
        .func_call("resize", "", "layouts")
        .call_arg(plugin->inputs.size())
        .for_start("i", plugin->attrs.size(), fields_size);
    for (size_t i = 0; i < plugin->inputs.size(); i++) {
      const auto& tensor = plugin->inputs[i];
      const String& cond = "strcmp(fields[i].name, \"layout_" + tensor->name + "\") == 0";
      if (i == 0) {
        stack_.switch_start(cond);
      } else {
        stack_.switch_case(cond);
      }
      stack_.func_call("TRTUtils::FromField")
          .call_arg(DocUtils::ToIndex("fields", "i"))
          .call_arg(DocUtils::ToIndex("layouts", i));
    }
    stack_.switch_end()
        .for_end()
        .func_call("new " + op_cls, DocUtils::ToDeclare(op_cls + "*", "plugin"))
        .call_arg("name");
    for (const auto& a : plugin->attrs) {
      stack_.call_arg(DocUtils::ToAttrAccess("meta_attr", a->name));
    }
    stack_.call_arg("layouts")
        .func_call("setPluginNamespace", NullOpt, DocUtils::ToPtr("plugin"))
        .inplace_start("c_str", NullOpt, DocUtils::ToDoc("name_space_"))
        .inplace_end()
        .func_end("plugin");
    // deserializePlugin
    stack_.func_def(creator_cls + "::deserializePlugin", plugin_cls + "*")
        .func_decorator("noexcept")
        .func_arg("name", "const char*")
        .func_arg("data", "const void*")
        .func_arg("length", "size_t")
        .func_start()
        .func_call("new " + op_cls, DocUtils::ToDeclare(op_cls + "*", "plugin"))
        .call_arg("name")
        .call_arg("data")
        .call_arg("length")
        .func_call("setPluginNamespace", NullOpt, DocUtils::ToPtr("plugin"))
        .inplace_start("c_str", NullOpt, DocUtils::ToDoc("name_space_"))
        .inplace_end()
        .func_end("plugin");
  }
}

void TensorRTPluginCodeGen::CodegenOutputInfer(const Plugin& plugin, bool as_desc) {
  Array<String> infer_args{"input_metas_", "meta_attr_", "false"};
  stack_.line("assert(n_inputs == " + std::to_string(plugin->inputs.size()) + ");")
      .func_call("resize", "", "input_metas_")
      .call_arg(plugin->inputs.size())
      .for_start("i", 0, plugin->inputs.size())
      .func_call("TRTUtils::ToMetaTensor", DocUtils::ToIndex("input_metas_", "i"));
  if (as_desc) {
    stack_.call_arg(DocUtils::ToIndex("in_descs", "i"));
  } else {
    stack_.call_arg(DocUtils::ToIndex("in_dims", "i")).call_arg("dtype_");
  }
  stack_.call_arg(DocUtils::ToIndex("layouts_", "i")).for_end();
  CodeGenSafeCall(plugin->externs["infer_output"], infer_args, "output_metas_");
}

void TensorRTPluginCodeGen::CodegenBufferInfer(const Plugin& plugin) {
  Array<String> infer_args{"input_metas_", "meta_attr_", "false"};
  CodeGenSafeCall(plugin->externs["infer_buffer"], infer_args, "buffer_metas_");
  stack_.for_start("b", "buffer_metas_")
      .assign("size", "size + max_batch * b.size(false)")
      .for_end();
}

void TensorRTPluginCodeGen::CodegenEnqueue(const Plugin& plugin, bool dynamic) {
  ICHECK(plugin->externs.count("cuda_compute")) << "cuda_compute is needed fo TensorRT plugin";
  auto prepare_tensor = [this, &dynamic](const PluginTensor& tensor,
                                         const Map<String, String>& dtypes, size_t idx,
                                         const String& collect) {
    const String& t_name = "d_" + tensor->name;
    const String& t_dtype = dtypes.count(tensor->name) ? dtypes[tensor->name] : tensor->dtype;
    const String& tensor_type = "DataTensor<" + t_dtype + ">";
    const String& anno = collect == "input" ? "const " + tensor_type + "&" : tensor_type;
    stack_.func_call("TRTUtils::To" + tensor_type, DocUtils::ToDeclare(anno, t_name));
    const auto& t_meta = DocUtils::ToIndex(collect + "_metas_", idx);
    if (dynamic) {
      stack_.call_arg(t_meta).call_arg(DocUtils::ToIndex(collect + "_descs", idx));
    } else {
      stack_.call_arg(t_meta).call_arg("batch_size");
    }
    if (collect == "input") {
      stack_.call_arg(DocUtils::ToIndex("inputs", idx));
    } else if (collect == "output") {
      stack_.call_arg(DocUtils::ToIndex("outputs", idx));
    } else {
      stack_.call_arg("workspace + offset");
    }
    return t_name;
  };
  for (const auto& dtypes : GetDtypeMatrix(plugin)) {
    const auto& tensor_dtypes = GetTensorDtypes(plugin, dtypes);
    Array<String> compute_args;
    String dtype_cond = "";
    if (dynamic) {
      for (size_t i = 0; i < plugin->inputs.size(); i++) {
        dtype_cond = dtype_cond + "input_descs[" + std::to_string(i) +
                     "].type == TRTUtils::ToDataType(\"" + dtypes.at(i) + "\")";
        dtype_cond = dtype_cond + (i == plugin->inputs.size() - 1 ? "" : " && ");
      }
    } else {
      dtype_cond = "dtype_ == TRTUtils::ToDataType(\"" + dtypes.at(0) + "\")";
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
    if (plugin->buffers.size() > 0) {
      stack_.assign("offset", 0, "size_t");
      for (size_t i = 0; i < plugin->buffers.size(); i++) {
        const String& t_name = prepare_tensor(plugin->outputs[i], tensor_dtypes, i, "buffer");
        compute_args.push_back(t_name);
        const String& size_name = "size_" + plugin->buffers[i]->name;
        stack_
            .func_call("size", DocUtils::ToDeclare("size_t", size_name),
                       DocUtils::ToIndex("buffer_metas_", i))
            .call_arg(false)
            .assign("offset", "offset + batch_size * " + size_name);
      }
    }
    compute_args.push_back("meta_attr_");
    compute_args.push_back("stream");
    CodeGenSafeCall(plugin->externs["cuda_compute"], compute_args);
    stack_.cond_end();
  }
}

TVM_REGISTER_GLOBAL("msc.plugin.GetTensorRTPluginSources")
    .set_body_typed([](const String& codegen_config, const String& print_config,
                       const String& codegen_type) -> Map<String, String> {
      TensorRTPluginCodeGen codegen = TensorRTPluginCodeGen(codegen_config);
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
