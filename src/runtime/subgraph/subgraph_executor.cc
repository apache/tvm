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
 * \file subgraph_runtime.cc
 */
#include "subgraph_executor.h"

#include <tvm/runtime/registry.h>

namespace tvm {
namespace runtime {

/*!
 *\bief Stop subgraph run.
 */
void SubGraphRuntime::Stop() { subgraph_stop(runtimes); }
/*!
 * \brief Run all the operations one by one.
 */
void SubGraphRuntime::Run() { subgraph_run(runtimes); }

void SubGraphRuntime::Init(const Array<tvm::runtime::Module>& modules) {
  subgraph_init(modules, &runtimes);
  SetupStorage();
  return;
}

void SubGraphRuntime::SetupStorage(void) {
  auto lastGraphRuntime = runtimes.back();
  int outputNum = lastGraphRuntime->runtimePtr->NumOutputs();
  for (int i = 0; i < outputNum; i++) {
    NDArray array = lastGraphRuntime->runtimePtr->GetOutput(i);
    auto dltensor = const_cast<DLTensor*>(array.operator->());
    vector<int64_t> shape;
    for (int i = 0; i < dltensor->ndim; i++) {
      shape.push_back(dltensor->shape[i]);
    }

    auto ndarray = NDArray::Empty(shape, dltensor->dtype, dltensor->device);
    ndarray.CreateView(shape, dltensor->dtype);
    output_entry_.push_back(ndarray);
  }
}

/*!
 * \brief set index-th input to the graph.
 * \param index The input index.
 * \param data_in The input data.
 */
void SubGraphRuntime::SetInput(int index, DLTensor* data_in) {
  auto gruntime = runtimes.front();
  gruntime->runtimePtr->SetInput(index, data_in);
}

void SubGraphRuntime::SetInput(const std::string& name, DLTensor* data_in) {
  auto gruntime = runtimes.front();
  gruntime->runtimePtr->SetInput(name, data_in);
}

/*!
 * \brief Get the number of outputs
 *
 * \return The number of outputs from last subgraph.
 */
int SubGraphRuntime::NumOutputs() const { return runtimes.back()->runtimePtr->NumOutputs(); }

/*!
 * \brief Get the number of inputs
 *
 * \return The number of inputs to the first subgraph.
 */
int SubGraphRuntime::NumInputs() const { return runtimes.front()->runtimePtr->NumInputs(); }

/*!
 * \brief Return NDArray for given input index.
 * \param index The input index.
 *
 * \return NDArray corresponding to given input node index.
 */
NDArray SubGraphRuntime::GetInput(int index, int mIndx) const {
  auto gruntime = runtimes[mIndx];
  return gruntime->runtimePtr->GetInput(index);
}

NDArray SubGraphRuntime::GetInput(const std::string& name, int mIndx) const {
  auto gruntime = runtimes[mIndx];
  return gruntime->runtimePtr->GetInput(name);
}

/*!
 * \brief Return NDArray Array for all output.
 *
 * \return NDArray Array for all output.
 */
Array<NDArray> SubGraphRuntime::GetOutput(bool syncPoll) {
  Array<NDArray> nd;
  if (subgraph_poll(&output_entry_, runtimes, syncPoll)) {
    for (auto output : output_entry_) {
      nd.push_back(output);
    }
  }
  return nd;
}

PackedFunc SubGraphRuntime::GetFunction(const std::string& name,
                                        const ObjectPtr<Object>& sptr_to_self) {
  // Return member functions during query.
  if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (String::CanConvertFrom(args[0])) {
        this->SetInput(args[0].operator String(), args[1]);
      } else {
        this->SetInput(static_cast<int>(args[0]), args[1]);
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
      int in_idx = 0, graph_idx = 0;
      if (args.num_args == 2) {
        graph_idx = args[1];
      }

      if (String::CanConvertFrom(args[0])) {
        *rv = this->GetInput(args[0].operator String(), graph_idx);
      } else {
        in_idx = args[0];
        if (in_idx >= 0) {
          *rv = this->GetInput(in_idx, graph_idx);
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

Module SubGraphRuntimeCreate(const Array<tvm::runtime::Module>& m) {
  auto exec = make_object<SubGraphRuntime>();
  exec->Init(m);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.subgraph_executor.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  *rv = SubGraphRuntimeCreate(args[0]);
});
}  // namespace runtime
}  // namespace tvm
