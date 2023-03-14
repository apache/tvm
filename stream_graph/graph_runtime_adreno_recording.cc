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


#include "./stream_graph_executor.h"
#include "../../opencl/opencl_common.h"
namespace tvm {
namespace runtime {

class GraphExecutorAdrenoStream : public StreamGraphExecutor {
 public:
  /*!
   * \brief Begin recording of operations
   */
  void StartCapture()
  {
    reinterpret_cast<OpenCLModuleNode*>(module_.operator->())->StartRecording();
  }
  /*!
   * \brief End recording. All operations will be saved
   */
  void EndCapture()
  {
    reinterpret_cast<OpenCLModuleNode*>(module_.operator->())->EndRecording();
  }
  /*!
   * \brief Run recorded operations in streaming mode
   */
  void RunGraph()
  {
    reinterpret_cast<OpenCLModuleNode*>(module_.operator->())->RunRecording();
  }

  /*!
   * \brief GetFunction Get the function based on input.
   * \param name The function which needs to be invoked.
   * \param sptr_to_self Packed function pointer.
   */
  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
    if (name == "run_graph") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->RunGraph(); });
    } else if (name == "start_capture") {
      return PackedFunc(
          [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->StartCapture(); });
    } else if (name == "end_capture") {
      return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->EndCapture(); });
    } else {
      return GraphExecutor::GetFunction(name, sptr_to_self);
    }
  }
};

Module GraphExecutorAdrenoStreamCreate(const std::string& sym_json, const tvm::runtime::Module& m,
                                    const std::vector<Device>& devs,
                                    PackedFunc lookup_linked_param_func) {
  auto exec = make_object<GraphExecutorAdrenoStream>();
  exec->Init(sym_json, m, devs, lookup_linked_param_func);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.graph_executor_adreno_recording.create")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      ICHECK_GE(args.num_args, 4)
          << "The expected number of arguments for graph_executor.create is "
             "at least 4, but it has "
          << args.num_args;
      PackedFunc lookup_linked_param_func;
      int dev_start_arg = 2;
      if (args[2].type_code() == kTVMPackedFuncHandle) {
        lookup_linked_param_func = args[2];
        dev_start_arg++;
      }

      *rv = GraphExecutorAdrenoStreamCreate(args[0], args[1], GetAllDevice(args, dev_start_arg),
                                         lookup_linked_param_func);
    });

}  // namespace runtime
}  // namespace tvm

