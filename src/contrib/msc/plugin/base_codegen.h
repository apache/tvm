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
 * \file src/contrib/msc/plugin/base_codegen.h
 * \brief The codegen for Plugin.
 */
#ifndef TVM_CONTRIB_MSC_PLUGIN_BASE_CODEGEN_H_
#define TVM_CONTRIB_MSC_PLUGIN_BASE_CODEGEN_H_

#include <dmlc/json.h>
#include <tvm/script/printer/doc.h>

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "../core/codegen/code_stack.h"
#include "../core/ir/plugin.h"
#include "../core/printer/cpp_printer.h"
#include "../core/printer/python_printer.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

/*!
 * \brief CodeGen for Plugin
 */
template <typename ConfigType>
class BasePluginCodeGen {
 public:
  /*!
   * \brief The constructor of BasePluginCodeGen
   * \param config the options for codegen.
   */
  explicit BasePluginCodeGen(const std::string& config = "") {
    config_.reset(new ConfigType());
    if (config.size() > 0) {
      std::istringstream is(config);
      dmlc::JSONReader reader(&is);
      reader.Read(config_.get());
    }
  }

  virtual ~BasePluginCodeGen() = default;

  /*! \brief Get plugin sources*/
  virtual const Map<String, String> GetBuildSources(const std::string& print_options = "") {
    Map<String, String> sources;
    // plugin sources
    for (const auto& name : ListPluginNames()) {
      const auto& plugin = GetPlugin(name);
      // attr declare
      const String& attr_macro = "TVM_CONTRIB_MSC_" + StringUtils::Upper(plugin->name) + "_ATTR_H_";
      this->stack_.line("#ifndef " + attr_macro)
          .line("#define " + attr_macro)
          .line()
          .line("#include \"plugin_utils.h\"")
          .line();
      StartNamespace();
      CodeGenAttrDeclare(plugin);
      EndNamespace();
      this->stack_.line("#endif  // " + attr_macro);
      sources.Set(plugin->name + "_attr.h", ToCppSource(print_options));
      // attr define
      this->stack_.line("#include \"" + plugin->name + "_attr.h\"").line();
      StartNamespace();
      CodeGenAttrDefine(plugin);
      EndNamespace();
      sources.Set(plugin->name + "_attr.cc", ToCppSource(print_options));
      // op decalre
      const String& op_macro = "TVM_CONTRIB_MSC_" + StringUtils::Upper(plugin->name) + "_OP_H_";
      this->stack_.line("#ifndef " + op_macro).line("#define " + op_macro).line();
      CodeGenOpHeader(plugin);
      StartNamespace();
      CodeGenOpDeclare(plugin);
      EndNamespace();
      this->stack_.line("#endif  // " + op_macro);
      sources.Set(plugin->name + "_op.h", ToCppSource(print_options));
      // op define
      this->stack_.line("#include \"" + plugin->name + "_op.h\"").line();
      StartNamespace();
      CodeGenOpDefine(plugin);
      EndNamespace();
      sources.Set(plugin->name + "_op.cc", ToCppSource(print_options));
      // op runtime
      if (this->config()->with_runtime) {
        CodeGenOpHeader(plugin);
        StartNamespace();
        CodeGenOpRuntime(plugin);
        EndNamespace();
        sources.Set(plugin->name + "_runtime.cc", ToCppSource(print_options));
      }
    }
    // cmakelists
    std::set<String> devices;
    for (const auto& name : ListPluginNames()) {
      const auto& plugin = GetPlugin(name);
      for (const auto& pair : plugin->externs) {
        if (StringUtils::EndsWith(pair.first, "_compute")) {
          devices.insert(StringUtils::Replace(pair.first, "_compute", ""));
        }
      }
    }
    CodeGenCmake(devices);
    sources.Set("CMakeLists.txt", ToCppSource(print_options));
    return sources;
  }

  /*! \brief Get manager sources*/
  virtual const Map<String, String> GetManagerSources(const std::string& print_options = "") {
    Map<String, String> sources;
    CodeGenManagerDepends();
    this->stack_.class_def("PluginManager(object)").class_start();
    CodeGenManagerMethods();
    for (const auto& name : ListPluginNames()) {
      CodeGenOpBuilder(GetPlugin(name));
    }
    if (this->config()->need_convert) {
      Map<Plugin, String> symbols;
      this->stack_.func_def("get_convert_map")
          .func_decorator("classmethod")
          .func_arg("cls", "object")
          .func_start();
      CodeGenConvertDepends();
      for (const auto& name : ListPluginNames()) {
        const auto& plugin = GetPlugin(name);
        const auto& symbol = CodeGenOpConvert(plugin);
        symbols.Set(plugin, symbol);
      }
      this->stack_.assign("converters", "{}");
      for (const auto& pair : symbols) {
        this->stack_.assign(DocUtils::ToIndex("converters", DocUtils::ToStr(pair.second)),
                            ConverterName(pair.first));
      }
      this->stack_.func_end("converters");
    }
    this->stack_.class_end();
    sources.Set("manager.py", ToPySource(print_options));
    return sources;
  }

 protected:
  /*! \brief Header of plugin files*/
  virtual void CodeGenOpHeader(const Plugin& plugin) {
    this->stack_.line("#include \"" + plugin->name + "_attr.h\"");
    std::set<String> include_headers;
    for (const auto& pair : plugin->externs) {
      if (pair.second->header.size() > 0 && !include_headers.count(pair.second->header)) {
        this->stack_.line("#include \"" + pair.second->header + "\"");
        include_headers.insert(pair.second->header);
      }
    }
    this->stack_.line();
  }

  /*! \brief Start the namespace*/
  void StartNamespace() {
    this->stack_.line("namespace tvm {")
        .line("namespace contrib {")
        .line("namespace msc {")
        .line("namespace plugin {")
        .line();
  }

  /*! \brief End the namespace*/
  void EndNamespace() {
    this->stack_.line("}  // namespace plugin")
        .line("}  // namespace msc")
        .line("}  // namespace contrib")
        .line("}  // namespace tvm");
  }

  /*! \brief Codegen safe call extern*/
  void CodeGenSafeCall(const PluginExtern& extern_func,
                       const Array<String>& call_args = Array<String>(), const String& ret = "") {
    this->stack_.scope_start("try {").func_call(extern_func->name, ret);
    for (const auto& arg : call_args) {
      this->stack_.call_arg(arg);
    }
    this->stack_.scope_end()
        .scope_start("} catch (const std::exception& exc) {")
        .line("std::cerr << \"Failed to run extern " + extern_func->name +
              " : \" << exc.what() << std::endl;")
        .line("throw std::runtime_error(\"Failed to run extern " + extern_func->name + "\");")
        .scope_end()
        .line("}");
  }

  /*! \brief Codegen plugin attr declare*/
  virtual void CodeGenAttrDeclare(const Plugin& plugin) {
    this->stack_.struct_start(MetaAttrCls(plugin)).comment("define attributes");
    for (const auto& attr : plugin->attrs) {
      this->stack_.declare(ToCppType(attr->type), attr->name);
      if (attr->default_value.size() > 0) {
        this->stack_.declare_arg(attr->default_value);
      }
    }
    this->stack_.line()
        .comment("print method")
        .func_def("operator<<", "friend std::ostream&")
        .func_arg("out", "std::ostream&")
        .func_arg("attrs", "const " + MetaAttrCls(plugin) + "&")
        .func_start()
        .line("out << \"[" + MetaAttrCls(plugin) + "] : \";");
    for (const auto& attr : plugin->attrs) {
      this->stack_.line("out << \"| " + attr->name + "(" + attr->type + ")=\" << attrs." +
                        attr->name + ";");
    }
    this->stack_.func_end("out").struct_end();
  }

  /*! \brief Codegen plugin attr define*/
  virtual void CodeGenAttrDefine(const Plugin& plugin) {}

  /*! \brief Codegen plugin op declare*/
  virtual void CodeGenOpDeclare(const Plugin& plugin) = 0;

  /*! \brief Codegen plugin op define*/
  virtual void CodeGenOpDefine(const Plugin& plugin) = 0;

  /*! \brief Codegen plugin runtime*/
  virtual void CodeGenOpRuntime(const Plugin& plugin) {}

  /*! \brief Codegen cmake file*/
  virtual void CodeGenCmake(const std::set<String>& devices) {
    CodeGenPreCmake(devices);
    CodeGenPostCmake(devices);
  }

  /*! \brief Codegen cmake start*/
  void CodeGenPreCmake(const std::set<String>& devices,
                       const Map<String, String>& extra_flags = Map<String, String>()) {
    const auto& p_name = this->config()->project_name;
    stack_.line("cmake_minimum_required(VERSION " + this->config()->cmake_version + " FATAL_ERROR)")
        .line("project(" + p_name + ")");
    if (devices.count("cuda")) {
      stack_.line("find_package(CUDA)").line("add_definitions(-DPLUGIN_ENABLE_CUDA)");
    }
    stack_.line();
    for (const auto& pair : extra_flags) {
      if (pair.second.size() == 0) {
        stack_.line("add_definitions(-D" + pair.first + ")");
      } else {
        stack_.line("add_definitions(-D" + pair.first + "=" + pair.second + ")");
      }
    }
    for (const auto& pair : this->config()->flags) {
      if (pair.second.size() == 0) {
        stack_.line("add_definitions(-D" + pair.first + ")");
      } else {
        stack_.line("add_definitions(-D" + pair.first + "=" + pair.second + ")");
      }
    }
    stack_.line();
  }

  /*! \brief Codegen cmake end*/
  void CodeGenPostCmake(const std::set<String>& devices,
                        const Array<String>& extra_includes = Array<String>(),
                        const Array<String>& extra_libs = Array<String>()) {
    const auto& p_name = this->config()->project_name;
    stack_.line()
        .line("file(GLOB_RECURSE PLUGIN_HEADERS src/*.h)")
        .line("file(GLOB_RECURSE PLUGIN_CC_SRCS src/*.cc)");
    if (devices.count("cuda")) {
      stack_.line("file(GLOB_RECURSE PLUGIN_CU_SRCS src/*.cu)");
    }
    if (devices.count("cuda")) {
      stack_.line("cuda_add_library(" + p_name + " SHARED ${PLUGIN_CC_SRCS} ${PLUGIN_CU_SRCS})");
    } else {
      stack_.line("add_library(" + p_name + " SHARED ${PLUGIN_CC_SRCS})");
    }
    // define includes
    String includes = StringUtils::Join(extra_includes, " ");
    if (this->config()->includes.size() > 0) {
      includes = includes + " " + StringUtils::Join(this->config()->includes, " ");
    }
    if (includes.size() > 0) {
      stack_.line("target_include_directories(" + p_name + " PUBLIC " + includes + ")");
    }
    // define libs
    String link_libs = StringUtils::Join(extra_libs, " ");
    const auto& libs = StringUtils::Join(this->config()->libs, " ");
    if (libs.size() > 0) {
      link_libs = link_libs + " " + libs;
    }
    if (link_libs.size() > 0) {
      stack_.line("target_link_libraries(" + p_name + " " + link_libs + ")");
    }
    const auto& install_dir = this->config()->install_dir;
    if (install_dir.size() > 0) {
      stack_.line()
          .line("SET(LIBRARY_OUTPUT_PATH " + install_dir + "/lib)")
          .line("file(COPY ${PLUGIN_HEADERS} DESTINATION " + install_dir + "/include)");
      if (this->config()->libs.size() > 0) {
        stack_.line("file(COPY " + libs + " DESTINATION " + install_dir + "/lib)");
      }
    }
  }

  /*! \brief Codegen manager depends*/
  virtual void CodeGenManagerDepends() {
    this->stack_.line("import os")
        .line("import shutil")
        .line("import ctypes")
        .line("from typing import Any, List, Dict")
        .line();
  }

  /*! \brief Codegen manager methods*/
  virtual void CodeGenManagerMethods() {
    // init method
    stack_.func_def("__init__")
        .func_arg("self", "object")
        .func_arg("root", "str", "None")
        .func_start()
        .cond_if("root is None")
        .assign("root", "os.path.dirname(__name__)")
        .cond_end()
        .assign(DocUtils::ToAttrAccess("self", "_lib_folder"), "os.path.join(root, \"lib\")")
        .func_call("assert")
        .inplace_start("os.path.isdir")
        .call_arg(DocUtils::ToAttrAccess("self", "_lib_folder"))
        .inplace_end()
        .assign(DocUtils::ToAttrAccess("self", "_include_folder"),
                "os.path.join(root, \"include\")")
        .func_call("assert")
        .inplace_start("os.path.isdir")
        .call_arg(DocUtils::ToAttrAccess("self", "_include_folder"))
        .inplace_end()
        .assign(DocUtils::ToAttrAccess("self", "_manager_file"),
                "os.path.join(root, \"manager.py\")")
        .func_call("assert")
        .inplace_start("os.path.isfile")
        .call_arg(DocUtils::ToAttrAccess("self", "_manager_file"))
        .inplace_end()
        .func_call("setup", "", "self")
        .func_end();
    // list headers
    this->stack_.func_def("list_includes")
        .func_arg("self", "object")
        .func_arg("as_abs", "bool", "False")
        .func_start()
        .assign("includes", "[]")
        .for_start("f", "os.listdir(self._include_folder)")
        .cond_if("as_abs")
        .func_call("append", "", "includes")
        .inplace_start("os.path.join")
        .call_arg(DocUtils::ToAttrAccess("self", "_include_folder"))
        .call_arg("f")
        .inplace_end()
        .cond_else()
        .func_call("append", "", "includes")
        .call_arg("f")
        .cond_end()
        .for_end()
        .func_end("includes");
    // copy the headers
    this->stack_.func_def("copy_includes")
        .func_arg("self", "object")
        .func_arg("dst", "str")
        .func_start()
        .cond_if("not os.path.isdir(dst)")
        .func_call("makedirs", "", "os")
        .call_arg("dst")
        .cond_end()
        .for_start("header", "os.listdir(self._include_folder)")
        .func_call("shutil.copyfile")
        .inplace_start("os.path.join")
        .call_arg(DocUtils::ToAttrAccess("self", "_include_folder"))
        .call_arg("header")
        .inplace_end()
        .inplace_start("os.path.join")
        .call_arg("dst")
        .call_arg("header")
        .inplace_end()
        .for_end()
        .func_end();
    // list libs
    this->stack_.func_def("list_libs")
        .func_arg("self", "object")
        .func_arg("as_abs", "bool", "False")
        .func_start()
        .assign("libs", "[]")
        .for_start("f", "os.listdir(self._lib_folder)")
        .cond_if("as_abs")
        .func_call("append", "", "libs")
        .inplace_start("os.path.join")
        .call_arg(DocUtils::ToAttrAccess("self", "_lib_folder"))
        .call_arg("f")
        .inplace_end()
        .cond_else()
        .func_call("append", "", "libs")
        .call_arg("f")
        .cond_end()
        .for_end()
        .func_end("libs");
    // copy the libs
    this->stack_.func_def("copy_libs")
        .func_arg("self", "object")
        .func_arg("dst", "str")
        .func_start()
        .cond_if("not os.path.isdir(dst)")
        .func_call("makedirs", "", "os")
        .call_arg("dst")
        .cond_end()
        .for_start("lib", "os.listdir(self._lib_folder)")
        .func_call("shutil.copyfile")
        .inplace_start("os.path.join")
        .call_arg(DocUtils::ToAttrAccess("self", "_lib_folder"))
        .call_arg("lib")
        .inplace_end()
        .inplace_start("os.path.join")
        .call_arg("dst")
        .call_arg("lib")
        .inplace_end()
        .for_end()
        .func_end();
    // export method
    this->stack_.func_def("export")
        .func_arg("self", "object")
        .func_arg("dst", "str")
        .func_start()
        .func_call("copy_includes", "", "self")
        .inplace_start("os.path.join")
        .call_arg("dst")
        .call_arg(DocUtils::ToStr("include"))
        .inplace_end()
        .func_call("copy_libs", "", "self")
        .inplace_start("os.path.join")
        .call_arg("dst")
        .call_arg(DocUtils::ToStr("lib"))
        .inplace_end()
        .func_call("shutil.copyfile")
        .call_arg(DocUtils::ToAttrAccess("self", "_manager_file"))
        .inplace_start("os.path.join")
        .call_arg("dst")
        .call_arg(DocUtils::ToStr("manager.py"))
        .inplace_end()
        .func_end();
    // get op names
    this->stack_.func_def("get_op_names", "List[str]")
        .func_arg("self", "object")
        .func_start()
        .assign("names", "[]");
    for (const auto& name : ListPluginNames()) {
      this->stack_.func_call("append", "", "names").call_arg(DocUtils::ToStr(name));
    }
    this->stack_.func_end("names");
    // get ops info
    this->stack_.func_def("get_ops_info", "dict")
        .func_arg("self", "object")
        .func_start()
        .assign("info", "{}");
    for (const auto& name : ListPluginNames()) {
      ICHECK(this->config()->ops_info.count(name)) << "Can not find op info for " << name;
      const auto& info = this->config()->ops_info[name];
      this->stack_.assign(DocUtils::ToIndex("info", DocUtils::ToStr(name)), info);
    }
    this->stack_.func_end("info");
  }

  /*! \brief Codegen manager for plugin*/
  virtual void CodeGenOpBuilder(const Plugin& plugin) {}

  /*! \brief Codegen convert depends*/
  virtual void CodeGenConvertDepends() {
    this->stack_.line("from tvm import relax")
        .line("from tvm.relax import call_dps_packed")
        .line("from tvm.contrib.msc.plugin import utils as plugin_utils")
        .line("from tvm.contrib.msc.plugin.op import _ffi_api as _plugin_api")
        .line("from tvm.contrib.msc.core import utils as msc_utils")
        .line();
  }

  /*! \brief Codegen convert function for plugin*/
  virtual const String CodeGenOpConvert(const Plugin& plugin) { return plugin->name; }

  /*! \brief Change code stack to cpp source*/
  const String ToCppSource(const std::string& print_options = "") {
    CppPrinter printer(print_options);
    for (const auto& d : this->stack_.GetDocs()) {
      printer.Append(d);
    }
    this->stack_.Reset();
    return printer.GetString();
  }

  /*! \brief Change code stack to python source*/
  const String ToPySource(const std::string& print_options = "") {
    PythonPrinter printer(print_options);
    for (const auto& d : this->stack_.GetDocs()) {
      printer.Append(d);
    }
    this->stack_.Reset();
    return printer.GetString();
  }

  std::vector<std::unordered_map<int, String>> GetDtypeMatrix(const Plugin& plugin) {
    std::vector<std::unordered_map<int, String>> matrix;
    if (plugin->support_dtypes.size() == 0) {
      std::unordered_map<int, String> dtypes;
      for (size_t i = 0; i < plugin->inputs.size(); i++) {
        dtypes[i] = plugin->inputs[i]->dtype;
      }
      matrix.push_back(dtypes);
    } else {
      Array<String> templates;
      Array<Array<String>> condidates;
      for (const auto& pair : plugin->support_dtypes) {
        templates.push_back(pair.first);
        condidates.push_back(pair.second);
      }
      for (const auto& t_dtypes : ArrayUtils::Product(condidates)) {
        std::unordered_map<int, String> dtypes;
        for (size_t i = 0; i < templates.size(); i++) {
          for (size_t in_idx = 0; in_idx < plugin->inputs.size(); in_idx++) {
            if (plugin->inputs[in_idx]->dtype == templates[i]) {
              dtypes[in_idx] = t_dtypes[i];
            }
          }
        }
        for (size_t i = 0; i < plugin->inputs.size(); i++) {
          if (dtypes.count(i)) {
            continue;
          }
          dtypes[i] = plugin->inputs[i]->dtype;
        }
        matrix.push_back(dtypes);
      }
    }
    return matrix;
  }

  const Map<String, String> GetTensorDtypes(const Plugin& plugin,
                                            const std::unordered_map<int, String>& dtypes) {
    Map<String, String> tensor_dtypes;
    for (const auto& pair : dtypes) {
      const String& ref_dtype = plugin->inputs[pair.first]->dtype;
      for (const auto& t : plugin->inputs) {
        if (t->dtype == ref_dtype) {
          tensor_dtypes.Set(t->name, pair.second);
        }
      }
      for (const auto& t : plugin->outputs) {
        if (t->dtype == ref_dtype) {
          tensor_dtypes.Set(t->name, pair.second);
        }
      }
      for (const auto& t : plugin->buffers) {
        if (t->dtype == ref_dtype) {
          tensor_dtypes.Set(t->name, pair.second);
        }
      }
    }
    return tensor_dtypes;
  }

  /*! \brief Change plugin comment in python*/
  const String GetPyComment(const Plugin& plugin) {
    String comment = "Python wrapper for " + plugin->name + "\nInputs\n------";
    for (const auto& t : plugin->inputs) {
      comment = comment + "\n" + t->name + ": " + t->dtype + "\n  " + t->describe;
    }
    comment = comment + "\nOutputs\n-------";
    for (const auto& t : plugin->outputs) {
      comment = comment + "\n" + t->name + ": " + t->dtype + "\n  " + t->describe;
    }
    if (plugin->attrs.size() > 0) {
      comment = comment + "\nAttributes\n-----------";
      for (const auto& a : plugin->attrs) {
        comment = comment + "\n" + a->name + ": " + ToPyType(a->type) + "\n  " + a->describe;
      }
    }
    return comment;
  }

  /*! \brief Get class name for meta attrs*/
  const String MetaAttrCls(const Plugin& plugin) const { return plugin->name + "MetaAttr"; }

  /*! \brief Get converter name for plugin*/
  const String ConverterName(const Plugin& plugin) const { return plugin->name + "Converter"; }

  /*! \brief Check if the type is list type. */
  bool IsListType(const String& type) { return StringUtils::StartsWith(type, "list"); }

  /*! \brief Get type of element. */
  const String GetEleType(const String& type) {
    if (!IsListType(type)) {
      return "";
    }
    return StringUtils::Replace(StringUtils::Replace(type, "list(", ""), ")", "");
  }

  /*! \brief Type name in cpp*/
  virtual const String ToCppType(const String& type) {
    if (IsListType(type)) {
      const auto& ele_type = GetEleType(type);
      return "std::vector<" + ToCppType(ele_type) + ">";
    }
    if (type == "int64") {
      return "int64_t";
    }
    if (type == "int32" || type == "int") {
      return "int32_t";
    }
    if (type == "int8") {
      return "int8_t";
    }
    if (type == "string") {
      return "std::string";
    }
    return type;
  }

  /*! \brief Type name in python*/
  virtual const String ToPyType(const String& type) {
    if (IsListType(type)) {
      const auto& ele_type = GetEleType(type);
      return "List[" + ToPyType(ele_type) + "]";
    }
    if (type == "int64" || type == "int32" || type == "int" || type == "int8") {
      return "int";
    }
    if (type == "string") {
      return "str";
    }
    return type;
  }

  /*!
   * \brief Compare version with version in config
   * 0 for same version, 1 for greater version, -1 for less version
   */
  int CompareVersion(size_t major, size_t minor, size_t patch) {
    return CommonUtils::CompareVersion(this->config()->version, {major, minor, patch});
  }

  /*! \brief The config of plugin codegen*/
  const std::shared_ptr<ConfigType> config() { return config_; }

  /*! \brief The stack of codes*/
  CodeStack stack_;

 private:
  std::shared_ptr<ConfigType> config_;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_PLUGIN_BASE_CODEGEN_H_
