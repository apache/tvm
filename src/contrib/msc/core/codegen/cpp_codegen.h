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
 * \file src/contrib/msc/core/codegen/cpp_codegen.h
 * \brief CPP codegen for MSCGraph.
 */
#ifndef TVM_CONTRIB_MSC_CORE_CODEGEN_CPP_CODEGEN_H_
#define TVM_CONTRIB_MSC_CORE_CODEGEN_CPP_CODEGEN_H_

#include <dmlc/json.h>
#include <tvm/script/printer/doc.h>

#include <string>

#include "../printer/cpp_printer.h"
#include "base_codegen.h"
#include "code_stack.h"
#include "codegen_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

template <typename ConfigType, typename HelperType>
class CppCodeGen : public BaseCodeGen<ConfigType, HelperType> {
 public:
  /*!
   * \brief The constructor of PyCodeGen
   * \param graph the graph to be generated.
   * \param config the options for codegen.
   */
  explicit CppCodeGen(const MSCGraph& graph, const std::string& config = "")
      : BaseCodeGen<ConfigType, HelperType>(graph, config) {}

  /*! \brief Stack the docs for the class declare*/
  virtual void CodeGenClassDeclare() = 0;

  /*! \brief Stack the docs for the class define*/
  virtual void CodeGenClassDefine() = 0;

  /*! \brief Stack the docs for the main func*/
  virtual void CodeGenMain() = 0;

  /*! \brief Stack the docs for the class define*/
  virtual void CodeGenCmake() = 0;

  /*! \brief Get sources*/
  virtual const Map<String, String> GetSources(const std::string& print_options = "") {
    Map<String, String> sources;
    auto add_source = [&print_options, &sources, this](const String& file) {
      CppPrinter printer(print_options);
      for (const auto& d : this->stack_.GetDocs()) {
        printer.Append(d);
      }
      sources.Set(file, printer.GetString());
      this->stack_.Reset();
    };
    // class declare
    CodeGenClassDeclare();
    add_source(this->graph()->name + ".h");
    // class define
    CodeGenClassDefine();
    add_source(this->graph()->name + ".cc");
    // main func
    CodeGenMain();
    add_source("main.cc");
    // cmakelists
    CodeGenCmake();
    add_source("CMakeLists.txt");
    return sources;
  }

 protected:
  void StartNamespace() {
    this->stack_.line("namespace tvm {").line("namespace contrib {").line("namespace msc {").line();
  }

  void EndNamespace() {
    this->stack_.line()
        .line("} // namespace tvm")
        .line("} // namespace contrib")
        .line("} // namespace msc")
        .line();
  }
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_CODEGEN_CPP_CODEGEN_H_
