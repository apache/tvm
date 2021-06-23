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
 * \file pipeline_executor.cc
 */
#include "pipeline_executor.h"

namespace tvm {
namespace runtime {

/*! \bief Stop pipeline run. */
void SubGraphRuntime::Stop() { pipeline_stop(runtimes_); }
/*!
 * \brief Run all the operations one by one.
 */
void SubGraphRuntime::Run() { pipeline_run(runtimes_, input_int_map_); }

void SubGraphRuntime::Init(const Array<tvm::runtime::Module>& modules,
                           const std::string& pipeline_json) {
  std::istringstream is(pipeline_json);
  dmlc::JSONReader reader(&is);
  this->Load(&reader);
  outpuNumber_ = pipeline_init(modules, &runtimes_, pipeline_conf_, mod_conf_);
  return;
}

/*!
 * \brief set index-th input to the modIndx-th graph.
 * \param index The input index.
 * \param data_in The input data.
 * \param modIndx The runtime index.
 */
void SubGraphRuntime::SetInput(int index, DLTensor* data_in, int mod_idx) {
  if (1 == mod_idx) {
    runtimes_[0]->runtimePtr->SetInput(index, data_in);
  } else {
    pipeline_setinput(input_int_map_, index, data_in, mod_idx);
  }
}

/*!
 * \brief Get the number of outputs
 *
 * \return The number of outputs from last pipeline.
 */
int SubGraphRuntime::NumOutputs() const { return outpuNumber_; }

/*!
 * \brief Get the number of inputs
 *
 * \return The number of inputs to the first pipeline.
 */
int SubGraphRuntime::NumInputs() const {
  int inputsNum = 0;
  for (auto runtime : runtimes_) {
    inputsNum += runtime->runtimePtr->NumInputs();
  }
  return inputsNum;
}

/*!
 * \brief Return NDArray for given input index.
 * \param index The input index.
 *
 * \return NDArray corresponding to given input node index.
 */
NDArray SubGraphRuntime::GetInput(int index, int mIndx) const {
  auto gruntime = runtimes_[mIndx];
  return gruntime->runtimePtr->GetInput(index);
}

/*!
 * \brief Return input index for given input name.
 * \param name The input name.
 *
 * \return int corresponding to given input node name.
 */
int SubGraphRuntime::GetInputIndex(const string& name, int mIndx) const {
  auto gruntime = runtimes_[mIndx - 1];
  return gruntime->runtimePtr->GetInputIndex(name);
}

/*!
 * \brief Return NDArray Array for all output.
 *
 * \return NDArray Array for all output.
 */
Array<NDArray> SubGraphRuntime::GetOutput(bool syncPoll) {
  Array<NDArray> nd;
  if (pipeline_poll(&output_entry_, runtimes_, syncPoll)) {
    for (auto output : output_entry_) {
      nd.push_back(output);
    }
  }
  return nd;
}

PackedFunc SubGraphRuntime::GetFunction(const std::string& name,
                                        const ObjectPtr<Object>& sptr_to_self) {
  if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      // Default use first runtime index value.
      int modIndx = 0;
      if (args.num_args == 3) {
        modIndx = static_cast<int>(args[2]);
      }
      if (String::CanConvertFrom(args[0])) {
        int index = this->GetInputIndex(args[0].operator String(), modIndx);
        this->SetInput(index, args[1], modIndx);
      } else {
        this->SetInput(static_cast<int>(args[0]), args[1], modIndx);
      }
    });
  } else if (name == "get_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (args.num_args == 1) {
        *rv = this->GetOutput(static_cast<bool>(args[0]));
      } else {
        *rv = this->GetOutput();
      }
    });
  } else if (name == "get_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      int in_idx = 0, mod_idx = 0;
      if (args.num_args == 2) {
        mod_idx = args[1];
      }

      if (String::CanConvertFrom(args[0])) {
        int index = this->GetInputIndex(args[0].operator String(), mod_idx);
        *rv = this->GetInput(index, mod_idx);
      } else {
        in_idx = args[0];
        if (in_idx >= 0) {
          *rv = this->GetInput(in_idx, mod_idx);
        }
      }
    });
  } else if (name == "get_num_outputs") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->NumOutputs(); });
  } else if (name == "get_num_inputs") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->NumInputs(); });
  } else if (name == "run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->Run(); });
  } else if (name == "stop") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->Stop(); });
  } else {
    return PackedFunc();
  }
}

Module PipelineRuntimeCreate(const Array<tvm::runtime::Module>& m,
                             const std::string& pipeline_json) {
  auto exec = make_object<SubGraphRuntime>();
  exec->Init(m, pipeline_json);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.pipeline_executor.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = PipelineRuntimeCreate(args[0], args[1]);
});
}  // namespace runtime
}  // namespace tvm
