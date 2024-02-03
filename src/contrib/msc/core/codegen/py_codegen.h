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
 * \file src/contrib/msc/core/codegen/py_codegen.h
 * \brief Python codegen for MSCGraph.
 */
#ifndef TVM_CONTRIB_MSC_CORE_CODEGEN_PY_CODEGEN_H_
#define TVM_CONTRIB_MSC_CORE_CODEGEN_PY_CODEGEN_H_

#include <dmlc/json.h>
#include <tvm/script/printer/doc.h>

#include <set>
#include <string>

#include "../printer/python_printer.h"
#include "base_codegen.h"
#include "code_stack.h"
#include "codegen_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

template <typename ConfigType, typename HelperType>
class PyCodeGen : public BaseCodeGen<ConfigType, HelperType> {
 public:
  /*!
   * \brief The constructor of PyCodeGen
   * \param graph the graph to be generated.
   * \param config the options for codegen.
   */
  explicit PyCodeGen(const MSCGraph& graph, const std::string& config = "")
      : BaseCodeGen<ConfigType, HelperType>(graph, config) {
    for (const auto& output : graph->GetOutputs()) {
      graph_outputs_.insert(output);
    }
  }

  /*! \brief Stack the docs for the script*/
  virtual void CodeGenScript() {
    CodeGenHeader();
    this->stack_.line().comment("Define the helpers");
    CodeGenHelper();
    this->stack_.line().comment("Define the graph");
    CodeGenGraph();
    if (this->config()->need_test) {
      this->stack_.line().comment("Define the test");
      CodeGenTest();
    }
  }

  /*! \brief Get sources*/
  virtual const Map<String, String> GetSources(const std::string& print_options = "") {
    Map<String, String> sources;
    PythonPrinter printer(print_options);
    CodeGenScript();
    for (const auto& d : this->stack_.GetDocs()) {
      printer.Append(d);
    }
    sources.Set(this->graph()->name + ".py", printer.GetString());
    return sources;
  }

 protected:
  /*! \brief Stack the docs for the header*/
  virtual void CodeGenHeader() {
    this->stack_.line("import os")
        .line("import numpy as np")
        .line("from typing import List, Dict, Any")
        .line("import tvm");
    if (this->config()->use_tools) {
      this->stack_.line("from tvm.contrib.msc.core import tools as msc_tools");
    }
    this->stack_.line("from tvm.contrib.msc.core import utils as msc_utils");
  }

  /*! \brief Stack the docs for the helpers*/
  virtual void CodeGenHelper() {
    if (this->config()->need_test) {
      this->stack_.func_def("load_data", "np.ndarray")
          .func_arg("name", "str")
          .func_arg("shape", "List[int]")
          .func_arg("dtype", "str")
          .func_start()
          .func_call("os.path.join", "path")
          .call_arg(DocUtils::ToStr(this->config()->baseline_folder))
          .call_arg("name + \".bin\"")
          .cond_if("os.path.isfile(path)")
          .func_call("np.fromfile", "data")
          .call_arg("path")
          .call_arg("dtype", "dtype")
          .method_call("reshape")
          .call_arg("shape")
          .cond_else()
          .func_call("np.ones", "data")
          .call_arg("(shape)")
          .method_call("astype")
          .call_arg("dtype")
          .cond_end()
          .func_end("data");
    }
  }

  /*! \brief Stack the docs for the test*/
  void CodeGenTest() {
    this->stack_.cond_if("__name__ == \"__main__\"")
        .comment("Prepare test datas")
        .assign("inputs", "{}")
        .assign("golden", "{}");
    for (const auto& i : this->graph()->input_names) {
      const auto& input = this->graph()->FindTensor(i);
      this->stack_
          .func_call("load_data", DocUtils::ToIndex("inputs", DocUtils::ToStr(input->alias)))
          .call_arg(DocUtils::ToStr(input->alias))
          .call_arg(DocUtils::ToList(input->shape, true))
          .call_arg(DocUtils::ToStr(runtime::DLDataType2String(input->dtype)));
    }
    for (const auto& o : this->graph()->output_names) {
      const auto& output = this->graph()->FindTensor(o);
      this->stack_
          .func_call("load_data", DocUtils::ToIndex("golden", DocUtils::ToStr(output->alias)))
          .call_arg(DocUtils::ToStr(output->alias))
          .call_arg(DocUtils::ToList(output->shape, true))
          .call_arg(DocUtils::ToStr(runtime::DLDataType2String(output->dtype)));
    }
    this->stack_.comment("Build and inference the graph");
    CodeGenInference();
    this->stack_.func_call("msc_utils.compare_arrays")
        .call_arg("golden")
        .call_arg("outputs")
        .call_arg(DocUtils::ToStr("detail"), "verbose")
        .cond_end();
  }

  /*! \brief Stack the docs for the node*/
  virtual void CodeGenNode(const MSCJoint& node, bool use_tools) {
    this->stack_.comment(this->Comment(node));
    // process inputs and weights by tools
    if (use_tools) {
      for (size_t i = 0; i < node->inputs.size(); i++) {
        const auto& input = node->InputAt(i);
        this->stack_.func_call("msc_tools.process_tensor", this->IdxInputBase(node, i, true))
            .call_arg(this->IdxInputBase(node, i, false))
            .call_arg(DocUtils::ToStr(input->name))
            .call_arg(DocUtils::ToStr(node->name))
            .call_arg(DocUtils::ToStr(this->config()->tools_scope))
            .call_arg(DocUtils::ToStr(this->config()->tools_tag));
      }
      for (const auto& pair : node->weights) {
        this->stack_
            .func_call("msc_tools.process_tensor", this->IdxWeightBase(node, pair.first, true))
            .call_arg(this->IdxWeightBase(node, pair.first, false))
            .call_arg(DocUtils::ToStr(pair.second->name))
            .call_arg(DocUtils::ToStr(node->name))
            .call_arg(DocUtils::ToStr(this->config()->tools_scope))
            .call_arg(DocUtils::ToStr(this->config()->tools_tag));
      }
    }
    for (const auto& d : this->GetOpCodes(node)) {
      this->stack_.line(d);
    }
    // process graph outputs by tools
    if (use_tools) {
      for (size_t i = 0; i < node->outputs.size(); i++) {
        int index = static_cast<int>(i);
        if (graph_outputs_.count(node->OutputAt(index))) {
          this->stack_.func_call("msc_tools.process_tensor", this->IdxOutputBase(node, index, true))
              .call_arg(this->IdxOutputBase(node, index, false))
              .call_arg(DocUtils::ToStr(node->OutputAt(index)->name))
              .call_arg(DocUtils::ToStr("exit"))
              .call_arg(DocUtils::ToStr(this->config()->tools_scope))
              .call_arg(DocUtils::ToStr(this->config()->tools_tag));
        }
      }
    }
  }

  /*! \brief Stack the docs for the graph*/
  virtual void CodeGenGraph() = 0;

  /*! \brief Stack the docs for the graph inference*/
  virtual void CodeGenInference() = 0;

  /*! \brief Get tensor type of the framework*/
  virtual const String TensorType() const { return "np.ndarray"; }

 private:
  std::set<MSCTensor> graph_outputs_;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_CODEGEN_PY_CODEGEN_H_
