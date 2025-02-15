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

#include <vector>

#include "source/codegen_source_base.h"

namespace tvm {
namespace codegen {

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
    tvm::runtime::Module target_module, const Array<runtime::Module>& ext_modules, Target target) {
  // Here we split modules into two groups:
  //  1. Those modules which can be exported to C-runtime. These are DSO-exportable
  //     (i.e. llvm or c) modules which return nothing from get_const_vars().
  //  2. Other modules.
  Array<runtime::Module> crt_exportable_modules;
  Array<runtime::Module> non_crt_exportable_modules;

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
  }

  return target_module;
}
}  // namespace codegen
}  // namespace tvm