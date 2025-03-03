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

#include <set>
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
      : BaseCodeGen<ConfigType, HelperType>(graph, config) {
    for (const auto& output : graph->GetOutputs()) {
      graph_outputs_.insert(output);
    }
  }

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
  /*! \brief Describe the prim*/
  virtual const String DescribePrim(const MSCPrim& prim) {
    // binary ops
    DESCRIBE_PRIM_BINARY("Min", "std::min", true)
    DESCRIBE_PRIM_BINARY("Max", "std::max", true)
    // special
    if (prim->optype == "if_then_else") {
      return "(" + this->DescribePrim(prim->ParentAt(0)) + "?" +
             this->DescribePrim(prim->ParentAt(1)) + ":" + this->DescribePrim(prim->ParentAt(2)) +
             ")";
    }
    return BaseCodeGen<ConfigType, HelperType>::DescribePrim(prim);
  }

  /*! \brief Stack the docs for the node*/
  virtual void CodeGenNode(const MSCJoint& node, bool use_tools) {
    this->stack_.comment(this->Comment(node));
    // process inputs and weights by tools
    if (use_tools) {
      const auto* pf = runtime::Registry::Get("msc_tool.codegen_tensor");
      ICHECK(pf != nullptr) << "Cannot find codegen_tensor func.";
      for (size_t i = 0; i < node->inputs.size(); i++) {
        const auto& input = node->InputAt(i);
        const Array<String>& lines = (*pf)(GetTensorCtx(input), input->name, node->name,
                                           this->config()->tools_scope, this->config()->tools_tag);
        for (const auto& l : lines) {
          this->stack_.line(l);
        }
      }
      for (const auto& pair : node->weights) {
        const Array<String>& lines = (*pf)(GetTensorCtx(pair.second), pair.second->name, node->name,
                                           this->config()->tools_scope, this->config()->tools_tag);
        for (const auto& l : lines) {
          this->stack_.line(l);
        }
      }
    }
    for (const auto& d : this->GetOpCodes(node)) {
      this->stack_.line(d);
    }
    // process graph outputs by tools
    if (use_tools) {
      const auto* pf = runtime::Registry::Get("msc_tool.codegen_tensor");
      ICHECK(pf != nullptr) << "Cannot find codegen_tensor func.";
      for (size_t i = 0; i < node->outputs.size(); i++) {
        int index = static_cast<int>(i);
        if (graph_outputs_.count(node->OutputAt(index))) {
          const auto& output = node->OutputAt(index);
          const Array<String>& lines =
              (*pf)(GetTensorCtx(output), output->name, node->name, this->config()->tools_scope,
                    this->config()->tools_tag);
          for (const auto& l : lines) {
            this->stack_.line(l);
          }
        }
      }
    }
  }

  /*! \brief Get the tensor context for codegen_tensor*/
  virtual const Map<String, String> GetTensorCtx(const MSCTensor& tensor) {
    Map<String, String> tensor_ctx;
    MSCJoint producer;
    if (this->graph()->weight_holders.count(tensor->name)) {
      producer = this->graph()->FindProducer(tensor);
      for (const auto& pair : producer->weights) {
        if (pair.second == tensor) {
          tensor_ctx.Set("tensor", this->IdxWeightBase(producer, pair.first));
          break;
        }
      }
      ICHECK(tensor_ctx.count("tensor"))
          << "Can not find weight " << tensor << " from " << producer;
    } else {
      const auto& pair = this->graph()->FindProducerAndIdx(tensor);
      producer = pair.first;
      tensor_ctx.Set("tensor", this->IdxOutputBase(pair.first, pair.second));
    }
    tensor_ctx.Set("producer", this->IdxNodeBase(producer));
    return tensor_ctx;
  }

  /*! \brief Get the step context for codegen_step*/
  virtual const Map<String, String> GetStepCtx() {
    Map<String, String> step_ctx;
    std::string version = "";
    for (size_t i = 0; i < this->config()->version.size(); i++) {
      version += std::to_string(this->config()->version[i]) +
                 (i < this->config()->version.size() - 1 ? "." : "");
    }
    step_ctx.Set("version", version);
    return step_ctx;
  }

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

 private:
  std::set<MSCTensor> graph_outputs_;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_CODEGEN_CPP_CODEGEN_H_
