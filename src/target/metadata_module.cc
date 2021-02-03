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

#include "../runtime/meta_data.h"
#include "llvm/llvm_module.h"
#include "source/source_module.h"

namespace tvm {
namespace codegen {

/*!
 * \brief Create a metadata module wrapper. The helper is used by different
 *        codegens, such as graph runtime codegen and the vm compiler.
 *
 * \param params The metadata for initialization of all modules.
 * \param target_module the internal module that is compiled by tvm.
 * \param ext_modules The external modules that needs to be imported inside the metadata
 * module(s).
 * \param target The target that all the modules are compiled for
 * \return The created metadata module that manages initialization of metadata.
 */
runtime::Module CreateMetadataModule(
    const std::unordered_map<std::string, runtime::NDArray>& params,
    tvm::runtime::Module target_module, const Array<runtime::Module>& ext_modules, Target target) {
  Array<tvm::runtime::Module> csource_modules;
  Array<tvm::runtime::Module> binary_modules;

  auto DSOExportable = [](tvm::runtime::Module& mod) {
    return !std::strcmp(mod->type_key(), "llvm") || !std::strcmp(mod->type_key(), "c");
  };

  // Wrap all submodules in the initialization wrapper.
  std::unordered_map<std::string, std::vector<std::string>> sym_metadata;
  for (tvm::runtime::Module mod : ext_modules) {
    auto pf_sym = mod.GetFunction("get_symbol");
    auto pf_var = mod.GetFunction("get_const_vars");
    std::vector<std::string> arrays;
    if (pf_sym != nullptr && pf_var != nullptr) {
      String symbol = pf_sym();
      Array<String> variables = pf_var();
      for (size_t i = 0; i < variables.size(); i++) {
        arrays.push_back(variables[i].operator std::string());
      }
      ICHECK_EQ(sym_metadata.count(symbol), 0U) << "Found duplicated symbol: " << symbol;
      sym_metadata[symbol] = arrays;
    }
    // We only need loading of serialized constant data
    // if there are constants present and required by the
    // runtime module to be initialized by the binary
    // metadata module. If not rest of the modules are
    // wrapped in c-source metadata module.

    // TODO(@manupa-arm) : we should be able to use csource_metadata
    // if the variables are empty when all the runtime modules implement get_func_names
    if (arrays.empty() && DSOExportable(mod) && target->kind->name == "c") {
      csource_modules.push_back(mod);
    } else {
      binary_modules.push_back(mod);
    }
  }

  if (target.defined() &&
      target->GetAttr<String>("runtime").value_or(String("")) == kTvmRuntimeCrt) {
    if (target->kind->name == "c") {
      csource_modules.push_back(target_module);
      target_module = CreateCSourceCrtMetadataModule(csource_modules, target);
    } else if (target->kind->name == "llvm") {
      binary_modules.push_back(target_module);
      target_module = CreateLLVMCrtMetadataModule(binary_modules, target);
    }
  } else {
    if (!binary_modules.empty()) {
      runtime::Module binary_meta_mod = runtime::MetadataModuleCreate(params, sym_metadata);
      binary_meta_mod.Import(target_module);
      for (const auto& it : binary_modules) {
        binary_meta_mod.Import(it);
      }
      return binary_meta_mod;
    }
  }
  return target_module;
}

}  // namespace codegen
}  // namespace tvm
