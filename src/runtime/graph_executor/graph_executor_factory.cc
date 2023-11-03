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
 * \file graph_executor_factory.cc
 * \brief Graph executor factory implementations
 */

#include "./graph_executor_factory.h"

#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include <iterator>
#include <vector>

namespace tvm {
namespace runtime {

GraphExecutorFactory::GraphExecutorFactory(
    const std::string& graph_json,
    const std::unordered_map<std::string, tvm::runtime::NDArray>& params,
    const std::string& module_name) {
  graph_json_ = graph_json;
  params_ = params;
  module_name_ = module_name;
}

PackedFunc GraphExecutorFactory::GetFunction(
    const String& name, const tvm::runtime::ObjectPtr<tvm::runtime::Object>& sptr_to_self) {
  if (name == module_name_) {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::vector<Device> devices;
      for (int i = 0; i < args.num_args; ++i) {
        devices.emplace_back(args[i].operator Device());
      }
      *rv = this->ExecutorCreate(devices);
    });
  } else if (name == "get_graph_json") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->graph_json_; });

  } else if (name == "get_graph_params") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      Map<String, tvm::runtime::NDArray> params;
      for (const auto& kv : params_) {
        params.Set(kv.first, kv.second);
      }
      *rv = params;
    });
  } else if (name == "debug_create") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_GE(args.size(), 2);
      std::string module_name = args[0].operator String();
      ICHECK(module_name == module_name_) << "Currently we only support single model for now.";
      std::vector<Device> devices;
      for (int i = 1; i < args.num_args; ++i) {
        devices.emplace_back(args[i].operator Device());
      }
      *rv = this->DebugExecutorCreate(devices);
    });
  } else if (name == "remove_params") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::unordered_map<std::string, tvm::runtime::NDArray> empty_params{};
      auto exec =
          make_object<GraphExecutorFactory>(this->graph_json_, empty_params, this->module_name_);
      exec->Import(this->imports_[0]);
      *rv = Module(exec);
    });
  } else if (name == "cuda_graph_create") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::vector<Device> devices;
      for (int i = 0; i < args.num_args; ++i) {
        devices.emplace_back(args[i].operator Device());
      }
      *rv = this->CudaGraphExecutorCreate(devices);
    });
  } else {
    return PackedFunc();
  }
}

void GraphExecutorFactory::SaveToBinary(dmlc::Stream* stream) {
  stream->Write(graph_json_);
  std::vector<std::string> names;
  std::vector<DLTensor*> arrays;
  for (const auto& v : params_) {
    names.emplace_back(v.first);
    arrays.emplace_back(const_cast<DLTensor*>(v.second.operator->()));
  }
  uint64_t sz = arrays.size();
  ICHECK(sz == names.size());
  stream->Write(sz);
  stream->Write(names);
  for (size_t i = 0; i < sz; ++i) {
    tvm::runtime::SaveDLTensor(stream, arrays[i]);
  }
  stream->Write(module_name_);
}

Module GraphExecutorFactory::ExecutorCreate(const std::vector<Device>& devs) {
  auto exec = make_object<GraphExecutor>();
  exec->Init(this->graph_json_, this->imports_[0], devs, PackedFunc());
  // set params
  SetParams(exec.get(), this->params_);
  return Module(exec);
}

Module GraphExecutorFactory::DebugExecutorCreate(const std::vector<Device>& devs) {
  const PackedFunc* pf = tvm::runtime::Registry::Get("tvm.graph_executor_debug.create");
  ICHECK(pf != nullptr) << "Cannot find function tvm.graph_executor_debug.create in registry. "
                           "Do you enable debug graph executor build?";
  // Debug executor create packed function will call GetAllContexs, so we unpack the devs.
  std::vector<int> unpacked_devs;
  for (const auto& dev : devs) {
    unpacked_devs.emplace_back(dev.device_type);
    unpacked_devs.emplace_back(dev.device_id);
  }
  size_t args_size = unpacked_devs.size() + 2;
  std::vector<TVMValue> values(args_size);
  std::vector<int> codes(args_size);
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  setter(0, this->graph_json_);
  setter(1, this->imports_[0]);
  for (size_t i = 0; i < unpacked_devs.size(); ++i) {
    setter(i + 2, unpacked_devs[i]);
  }
  TVMRetValue rv;
  pf->CallPacked(TVMArgs(values.data(), codes.data(), args_size), &rv);
  Module mod = rv.operator Module();
  // debug graph executor is one child class of graph executor.
  SetParams(const_cast<GraphExecutor*>(mod.as<GraphExecutor>()), this->params_);
  return mod;
}

Module GraphExecutorFactory::CudaGraphExecutorCreate(const std::vector<Device>& devs) {
  const PackedFunc* pf = tvm::runtime::Registry::Get("tvm.graph_executor_cuda_graph.create");
  ICHECK(pf != nullptr) << "Cannot find function tvm.graph_executor_cuda_graph.create in registry. "
                           "Did you set(USE_GRAPH_EXECUTOR_CUGRAPH=ON)?";
  std::vector<int> unpacked_devs;
  for (const auto& dev : devs) {
    unpacked_devs.emplace_back(dev.device_type);
    unpacked_devs.emplace_back(dev.device_id);
  }
  size_t args_size = unpacked_devs.size() + 2;
  std::vector<TVMValue> values(args_size);
  std::vector<int> codes(args_size);
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  setter(0, this->graph_json_);
  setter(1, this->imports_[0]);
  for (size_t i = 0; i < unpacked_devs.size(); ++i) {
    setter(i + 2, unpacked_devs[i]);
  }
  TVMRetValue rv;
  pf->CallPacked(TVMArgs(values.data(), codes.data(), args_size), &rv);
  Module mod = rv.operator Module();
  SetParams(const_cast<GraphExecutor*>(mod.as<GraphExecutor>()), this->params_);
  return mod;
}

Module GraphExecutorFactoryModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string graph_json;
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
  std::string module_name;
  ICHECK(stream->Read(&graph_json));
  uint64_t sz;
  ICHECK(stream->Read(&sz));
  std::vector<std::string> names;
  ICHECK(stream->Read(&names));
  ICHECK(sz == names.size());
  for (size_t i = 0; i < sz; ++i) {
    tvm::runtime::NDArray temp;
    temp.Load(stream);
    params[names[i]] = temp;
  }
  ICHECK(stream->Read(&module_name));
  auto exec = make_object<GraphExecutorFactory>(graph_json, params, module_name);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.graph_executor_factory.create")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      ICHECK_GE(args.num_args, 3) << "The expected number of arguments for "
                                     "graph_executor_factory.create needs at least 3, "
                                     "but it has "
                                  << args.num_args;
      // The argument order is graph_json, module, module_name, param0_name, param0_tensor,
      // [param1_name, param1_tensor], ...
      ICHECK_EQ((args.size() - 3) % 2, 0);
      std::unordered_map<std::string, tvm::runtime::NDArray> params;
      for (size_t i = 3; i < static_cast<size_t>(args.size()); i += 2) {
        std::string name = args[i].operator String();
        params[name] = args[i + 1].operator tvm::runtime::NDArray();
      }
      auto exec = make_object<GraphExecutorFactory>(args[0], params, args[2]);
      exec->Import(args[1]);
      *rv = Module(exec);
    });

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_GraphExecutorFactory")
    .set_body_typed(GraphExecutorFactoryModuleLoadBinary);

Module GraphRuntimeFactoryModuleLoadBinary(void* strm) {
  LOG(WARNING) << "You are loading a module which was built with GraphRuntimeFactory. "
               << "GraphRuntime has been renamed to GraphExecutor, and support for loading "
               << "GraphRuntimeFactory modules will be removed after the next TVM release. "
               << "Please rebuild the module before then to avoid breakage.";
  return GraphExecutorFactoryModuleLoadBinary(strm);
}

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_GraphRuntimeFactory")
    .set_body_typed(GraphRuntimeFactoryModuleLoadBinary);

}  // namespace runtime
}  // namespace tvm
