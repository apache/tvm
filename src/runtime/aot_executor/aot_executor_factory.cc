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
 * \file aot_executor_factory.cc
 * \brief AOT executor factory implementations
 */

#include "./aot_executor_factory.h"

#include <tvm/runtime/container/string.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include <iterator>
#include <vector>

namespace tvm {
namespace runtime {

AotExecutorFactory::AotExecutorFactory(
    const std::unordered_map<std::string, tvm::runtime::NDArray>& params,
    const std::string& module_name) {
  params_ = params;
  module_name_ = module_name;
}

PackedFunc AotExecutorFactory::GetFunction(
    const String& name, const tvm::runtime::ObjectPtr<tvm::runtime::Object>& sptr_to_self) {
  if (name == module_name_) {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      ICHECK_GT(args.num_args, 0) << "Must supply at least one device argument";
      std::vector<Device> devices;
      for (int i = 0; i < args.num_args; ++i) {
        devices.emplace_back(args[i].operator Device());
      }
      *rv = this->ExecutorCreate(devices);
    });
  } else if (name == "list_module_names") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      Array<String> names = {module_name_};
      *rv = names;
    });
  } else if (name == "remove_params") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::unordered_map<std::string, tvm::runtime::NDArray> empty_params{};
      auto exec = make_object<AotExecutorFactory>(empty_params, this->module_name_);
      exec->Import(this->imports_[0]);
      *rv = Module(exec);
    });
  } else {
    return PackedFunc();
  }
}

void AotExecutorFactory::SaveToBinary(dmlc::Stream* stream) {
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

Module AotExecutorFactory::ExecutorCreate(const std::vector<Device>& devs) {
  auto exec = make_object<AotExecutor>(this->imports_[0], devs);
  // set params
  SetParams(exec.get(), this->params_);
  return Module(exec);
}

Module AotExecutorFactoryModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
  std::string module_name;
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
  auto exec = make_object<AotExecutorFactory>(params, module_name);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.aot_executor_factory.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  ICHECK_GE(args.num_args, 2) << "The expected number of arguments for "
                                 "aot_executor_factory.create needs at least 2, "
                                 "but it has "
                              << args.num_args;
  // The argument order is module, module_name, param0_name, param0_tensor,
  // [param1_name, param1_tensor], ...
  ICHECK_EQ((args.size() - 2) % 2, 0);
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
  for (size_t i = 2; i < static_cast<size_t>(args.size()); i += 2) {
    std::string name = args[i].operator String();
    params[name] = args[i + 1].operator tvm::runtime::NDArray();
  }
  auto exec = make_object<AotExecutorFactory>(params, args[1]);
  exec->Import(args[0]);
  *rv = Module(exec);
});

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_AotExecutorFactory")
    .set_body_typed(AotExecutorFactoryModuleLoadBinary);

}  // namespace runtime
}  // namespace tvm
