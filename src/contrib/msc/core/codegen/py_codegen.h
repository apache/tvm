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

#include <string>

#include "../printer/python_printer.h"
#include "base_codegen.h"
#include "code_stack.h"
#include "codegen_utils.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

template <typename ConfigType>
class PyCodeGen : public BaseCodeGen<ConfigType> {
 public:
  /*!
   * \brief The constructor of PyCodeGen
   * \param graph the graph to be generated.
   * \param config the options for codegen.
   */
  explicit PyCodeGen(const MSCGraph& graph, const std::string& config = "")
      : BaseCodeGen<ConfigType>(graph, config) {}

  /*! \brief Stack the docs for the script*/
  virtual const Array<Doc> GetDocs() {
    CodeGenHeader();
    this->stack_.line().comment("Define the helpers");
    CodeGenHelper();
    this->stack_.line().comment("Define the graph");
    CodeGenGraph();
    if (this->config()->need_test) {
      this->stack_.line().comment("Define the test");
      CodeGenTest();
    }
    return this->stack_.GetDocs();
  }

  /*! \brief Get sources*/
  virtual const Map<String, String> GetSources(const std::string& print_options = "") {
    Map<String, String> sources;
    PythonPrinter printer(print_options);
    for (const auto& d : this->GetDocs()) {
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
        .line("from typing import List, Dict")
        .line("import tvm")
        .line("from tvm.contrib.msc.core import utils as msc_utils");
  }

  /*! \brief Stack the docs for the helpers*/
  virtual void CodeGenHelper() {
    this->stack_.func_def("process_tensor", TensorType())
        .func_arg("tensor", TensorType())
        .func_arg("name", "str")
        .func_arg("consumer", "str")
        .func_start()
        .func_end("tensor");
    if (this->config()->need_test) {
      this->stack_.func_def("load_data", "np.ndarray")
          .func_arg("name", "str")
          .func_arg("shape", "List[int]")
          .func_arg("dtype", "str")
          .func_start()
          .call_start("os.path.join")
          .call_str_arg(this->config()->baseline_folder)
          .call_arg("name + \".bin\"")
          .call_end("path")
          .cond_if("os.path.isfile(path)")
          .call_start("np.fromfile")
          .call_arg("path")
          .call_arg("dtype", "dtype")
          .call_end("data")
          .inplace_start("reshape")
          .call_arg("shape")
          .inplace_end()
          .cond_else()
          .call_start("np.ones")
          .call_arg("(shape)")
          .call_end("data")
          .inplace_start("astype")
          .call_arg("dtype")
          .inplace_end()
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
      this->stack_.call_start("load_data")
          .call_str_arg(input->alias)
          .call_list_arg(input->shape, "", true)
          .call_str_arg(runtime::DLDataType2String(input->dtype))
          .call_end("inputs[\"" + input->alias + "\"]");
    }
    for (const auto& o : this->graph()->output_names) {
      const auto& output = this->graph()->FindTensor(o);
      this->stack_.call_start("load_data")
          .call_str_arg(output->alias)
          .call_list_arg(output->shape, "", true)
          .call_str_arg(runtime::DLDataType2String(output->dtype))
          .call_end("golden[\"" + output->alias + "\"]");
    }
    this->stack_.comment("Build and inference the graph");
    CodeGenInference();
    this->stack_.line("msc_utils.compare_arrays(golden, outputs, verbose=\"detail\")").cond_end();
  }

  /*! \brief Stack the docs for the node*/
  virtual void CodeGenNode(const MSCJoint& node) {
    this->stack_.comment(this->Comment(node));
    if (this->config()->need_process) {
      for (size_t i = 0; i < node->inputs.size(); i++) {
        const auto& input = node->InputAt(i);
        this->stack_.call_start("process_tensor")
            .call_arg(this->IdxInput(node, i, true))
            .call_str_arg(input->name)
            .call_str_arg(node->name)
            .call_end(this->IdxInput(node, i, false));
      }
      for (const auto& pair : node->weights) {
        this->stack_.call_start("process_tensor")
            .call_arg(this->IdxWeight(node, pair.first, true))
            .call_str_arg(pair.second->name)
            .call_str_arg(node->name)
            .call_end(this->IdxWeight(node, pair.first, false));
      }
    }
    for (const auto& d : this->GetOpCodes(node)) {
      this->stack_.line(d);
    }
  }

  /*! \brief Stack the docs for the graph*/
  virtual void CodeGenGraph() = 0;

  /*! \brief Stack the docs for the graph inference*/
  virtual void CodeGenInference() = 0;

  /*! \brief Get tensor type of the framework*/
  virtual const String TensorType() const { return "np.ndarray"; }
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm
#endif  // TVM_CONTRIB_MSC_CORE_CODEGEN_PY_CODEGEN_H_
