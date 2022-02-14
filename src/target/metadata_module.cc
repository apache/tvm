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
 * \file metadata_module.cc
 * \brief Defines functions that build MetadataModules for C++ and C runtimes.
 */
#include "metadata_module.h"

#include <tvm/relay/runtime.h>

#include <vector>

#include "../relay/backend/utils.h"
#include "../runtime/const_loader_module.h"
#include "../runtime/meta_data.h"
#include "llvm/llvm_module.h"
#include "source/source_module.h"

namespace tvm {
namespace codegen {

static runtime::Module CreateCrtMetadataModule(
    runtime::Module target_module, Target target, relay::Runtime runtime,
    relay::backend::ExecutorCodegenMetadata metadata,
    Array<runtime::Module> non_crt_exportable_modules,
    Array<runtime::Module> crt_exportable_modules,
    const std::unordered_map<std::string, runtime::NDArray>& const_var_ndarray) {
  if (!non_crt_exportable_modules.empty()) {
    std::string non_exportable_modules;
    for (unsigned int i = 0; i < non_crt_exportable_modules.size(); i++) {
      if (i > 0) {
        non_exportable_modules += ", ";
      }
      auto mod = non_crt_exportable_modules[i];
      auto pf_sym = mod.GetFunction("get_symbol");
      if (pf_sym != nullptr) {
        non_exportable_modules += pf_sym().operator std::string();
      } else {
        non_exportable_modules +=
            std::string{"(module type_key="} + mod->type_key() + std::string{")"};
      }
    }
    CHECK(false) << "These " << non_crt_exportable_modules.size()
                 << " modules are not exportable to C-runtime: " << non_exportable_modules;
  }

  if (target->kind->name == "c") {
    crt_exportable_modules.push_back(target_module);
    target_module =
        CreateCSourceCrtMetadataModule(crt_exportable_modules, target, runtime, metadata);
  } else if (target->kind->name == "llvm") {
#ifdef TVM_LLVM_VERSION
    crt_exportable_modules.push_back(target_module);
    target_module = CreateLLVMCrtMetadataModule(crt_exportable_modules, target, runtime);
#else   // TVM_LLVM_VERSION
    LOG(FATAL) << "TVM was not built with LLVM enabled.";
#endif  // TVM_LLVM_VERSION
  }

  return target_module;
}

static runtime::metadata::Metadata ConvertMetaData(
    relay::backend::ExecutorCodegenMetadata metadata) {
  std::vector<runtime::metadata::TensorInfo> inputs;
  CHECK(metadata.defined());
  CHECK(metadata->inputs.defined());
  CHECK(metadata->inputs.size() == metadata->input_tensor_types.size());
  for (size_t i = 0; i < metadata->inputs.size(); ++i) {
    LOG(INFO) << i;
    auto ttype = metadata->input_tensor_types[i];
    auto v = metadata->inputs[i];
    CHECK(v.defined());
    CHECK(ttype.defined());
    inputs.push_back(
        runtime::metadata::TensorInfo(make_object<target::metadata::InMemoryTensorInfoNode>(
            v->name_hint, relay::backend::ShapeToJSON(ttype->shape), ttype->dtype)));
  }

  LOG(INFO) << "MAKE METADATA? ";
  std::vector<runtime::metadata::TensorInfo> outputs;
  auto output_ttypes = metadata->output_tensor_types;
  for (unsigned int i = 0; i < output_ttypes.size(); i++) {
    auto ttype = output_ttypes[i];
    std::stringstream name;
    name << "output" << i;
    outputs.push_back(
        runtime::metadata::TensorInfo(make_object<target::metadata::InMemoryTensorInfoNode>(
            name.str(), relay::backend::ShapeToJSON(ttype->shape), ttype->dtype)));
  }
  std::vector<std::string> devices_vector;
  for (auto d : metadata->devices) {
    devices_vector.push_back(d.operator std::string());
  }
  auto n = make_object<target::metadata::InMemoryMetadataNode>(
      kMetadataVersion, inputs, outputs, devices_vector, runtime::kTvmExecutorAot,
      metadata->mod_name, metadata->interface_api, metadata->unpacked_api);

  return runtime::metadata::Metadata(std::move(n));
}

static runtime::Module CreateCppMetadataModule(
    runtime::Module target_module, Target target, relay::Runtime runtime,
    relay::backend::ExecutorCodegenMetadata metadata,
    const std::unordered_map<std::string, std::vector<std::string>>& const_vars_by_symbol,
    Array<runtime::Module> non_crt_exportable_modules,
    Array<runtime::Module> crt_exportable_modules,
    const std::unordered_map<std::string, runtime::NDArray>& const_var_ndarray) {
  if (!non_crt_exportable_modules.empty()) {
    runtime::Module const_loader_mod =
        runtime::ConstLoaderModuleCreate(const_var_ndarray, const_vars_by_symbol);
    const_loader_mod.Import(target_module);
    for (const auto& it : non_crt_exportable_modules) {
      const_loader_mod.Import(it);
    }
    target_module = const_loader_mod;
  }

  runtime::metadata::Metadata metadata_tmp = ConvertMetaData(metadata);

  LOG(INFO) << "MAKE METADATA: " << metadata_tmp;

  if (metadata->executor == runtime::kTvmExecutorAot && runtime->name == relay::kTvmRuntimeCpp) {
    if (target->kind->name == "c") {
      auto metadata_module = CreateCSourceCppMetadataModule(metadata_tmp);
      metadata_module->Import(target_module);
      target_module = metadata_module;
    }
#ifdef TVM_LLVM_VERSION
    else if (target->kind->name == "llvm") {
      auto metadata_module = CreateLLVMCppMetadataModule(metadata_tmp, target, runtime);
      metadata_module->Import(target_module);
      target_module = metadata_module;
    }
#endif  // TVM_LLVM_VERSION
    else {
      CHECK(false) << "Don't know how to create MetadataModule for target type " << target->str();
    }
  }

  return target_module;
}

/*!
 * \brief Create a metadata module wrapper. The helper is used by different
 *        codegens, such as graph executor codegen and the vm compiler.
 *
 * \param params The metadata for initialization of all modules.
 * \param target_module the internal module that is compiled by tvm.
 * \param ext_modules The external modules that needs to be imported inside the metadata
 * module(s).
 * \param target The target that all the modules are compiled for
 * \return The created metadata module that manages initialization of metadata.
 */
runtime::Module CreateMetadataModule(
    const std::unordered_map<std::string, runtime::NDArray>& const_var_ndarray,
    tvm::runtime::Module target_module, const Array<runtime::Module>& ext_modules, Target target,
    tvm::relay::Runtime runtime, relay::backend::ExecutorCodegenMetadata metadata) {
  // Here we split modules into two groups:
  //  1. Those modules which can be exported to C-runtime. These are DSO-exportable
  //     (i.e. llvm or c) modules which return nothing from get_const_vars().
  //  2. Other modules.
  Array<runtime::Module> crt_exportable_modules;
  Array<runtime::Module> non_crt_exportable_modules;

  auto DSOExportable = [](tvm::runtime::Module& mod) {
    return !std::strcmp(mod->type_key(), "llvm") || !std::strcmp(mod->type_key(), "c");
  };

  bool is_targeting_crt = runtime->name == "crt";

  // Wrap all submodules in the initialization wrapper.
  std::unordered_map<std::string, std::vector<std::string>> const_vars_by_symbol;
  for (tvm::runtime::Module mod : ext_modules) {
    auto pf_sym = mod.GetFunction("get_symbol");
    auto pf_var = mod.GetFunction("get_const_vars");
    std::vector<std::string> symbol_const_vars;
    if (pf_sym != nullptr && pf_var != nullptr) {
      String symbol = pf_sym();
      Array<String> variables = pf_var();
      for (size_t i = 0; i < variables.size(); i++) {
        symbol_const_vars.push_back(variables[i].operator std::string());
      }
      ICHECK_EQ(const_vars_by_symbol.count(symbol), 0U) << "Found duplicated symbol: " << symbol;
      const_vars_by_symbol[symbol] = symbol_const_vars;
    }
    // We only need loading of serialized constant data
    // if there are constants present and required by the
    // runtime module to be initialized by the binary
    // metadata module. If not rest of the modules are
    // wrapped in c-source metadata module.

    // TODO(@manupa-arm) : we should be able to use csource_metadata
    // if the variables are empty when all the runtime modules implement get_func_names
    if (symbol_const_vars.empty() && is_targeting_crt && DSOExportable(mod) &&
        (target->kind->name == "c" || target->kind->name == "llvm")) {
      crt_exportable_modules.push_back(mod);
    } else {
      non_crt_exportable_modules.push_back(mod);
    }
  }

  if (is_targeting_crt) {
    LOG(INFO) << "Create CRT metadata: " << metadata.defined();
    return CreateCrtMetadataModule(target_module, target, runtime, metadata,
                                   non_crt_exportable_modules, crt_exportable_modules,
                                   const_var_ndarray);
  } else {
    return CreateCppMetadataModule(target_module, target, runtime, metadata, const_vars_by_symbol,
                                   non_crt_exportable_modules, crt_exportable_modules,
                                   const_var_ndarray);
  }
}

}  // namespace codegen

}  // namespace tvm
