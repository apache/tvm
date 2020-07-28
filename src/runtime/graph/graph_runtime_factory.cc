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
 * \file graph_runtime_factory.cc
 * \brief Graph runtime factory implementations
 */

#include "./graph_runtime_factory.h"

#include <tvm/node/container.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/registry.h>

#include <iterator>
#include <vector>

namespace tvm {
namespace runtime {

GraphRuntimeFactory::GraphRuntimeFactory(
    const std::string& graph_json,
    const std::unordered_map<std::string, tvm::runtime::NDArray>& params,
    const std::string& module_name) {
  graph_json_ = graph_json;
  params_ = params;
  module_name_ = module_name;
}

PackedFunc GraphRuntimeFactory::GetFunction(
    const std::string& name, const tvm::runtime::ObjectPtr<tvm::runtime::Object>& sptr_to_self) {
  if (name == module_name_) {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::vector<TVMContext> contexts;
      for (int i = 0; i < args.num_args; ++i) {
        contexts.emplace_back(args[i].operator TVMContext());
      }
      *rv = this->RuntimeCreate(contexts);
    });
  } else if (name == "debug_create") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK_GE(args.size(), 2);
      std::string module_name = args[0].operator String();
      CHECK(module_name == module_name_) << "Currently we only support single model for now.";
      std::vector<TVMContext> contexts;
      for (int i = 1; i < args.num_args; ++i) {
        contexts.emplace_back(args[i].operator TVMContext());
      }
      *rv = this->DebugRuntimeCreate(contexts);
    });
  } else if (name == "remove_params") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::unordered_map<std::string, tvm::runtime::NDArray> empty_params{};
      auto exec =
          make_object<GraphRuntimeFactory>(this->graph_json_, empty_params, this->module_name_);
      exec->Import(this->imports_[0]);
      *rv = Module(exec);
    });
  } else {
    return PackedFunc();
  }
}

void GraphRuntimeFactory::SaveToBinary(dmlc::Stream* stream) {
  stream->Write(graph_json_);
  std::vector<std::string> names;
  std::vector<DLTensor*> arrays;
  for (const auto& v : params_) {
    names.emplace_back(v.first);
    arrays.emplace_back(const_cast<DLTensor*>(v.second.operator->()));
  }
  uint64_t sz = arrays.size();
  CHECK(sz == names.size());
  stream->Write(sz);
  stream->Write(names);
  for (size_t i = 0; i < sz; ++i) {
    tvm::runtime::SaveDLTensor(stream, arrays[i]);
  }
  stream->Write(module_name_);
}

Module GraphRuntimeFactory::RuntimeCreate(const std::vector<TVMContext>& ctxs) {
  auto exec = make_object<GraphRuntime>();
  exec->Init(this->graph_json_, this->imports_[0], ctxs);
  // set params
  SetParams(exec.get(), this->params_);
  return Module(exec);
}

Module GraphRuntimeFactory::DebugRuntimeCreate(const std::vector<TVMContext>& ctxs) {
  const PackedFunc* pf = tvm::runtime::Registry::Get("tvm.graph_runtime_debug.create");
  CHECK(pf != nullptr) << "Cannot find function tvm.graph_runtime_debug.create in registry. "
                          "Do you enable debug graph runtime build?";
  // Debug runtime create packed function will call GetAllContexs, so we unpack the ctxs.
  std::vector<int> unpacked_ctxs;
  for (const auto& ctx : ctxs) {
    unpacked_ctxs.emplace_back(ctx.device_type);
    unpacked_ctxs.emplace_back(ctx.device_id);
  }
  size_t args_size = unpacked_ctxs.size() + 2;
  std::vector<TVMValue> values(args_size);
  std::vector<int> codes(args_size);
  runtime::TVMArgsSetter setter(values.data(), codes.data());
  setter(0, this->graph_json_);
  setter(1, this->imports_[0]);
  for (size_t i = 0; i < unpacked_ctxs.size(); ++i) {
    setter(i + 2, unpacked_ctxs[i]);
  }
  TVMRetValue rv;
  pf->CallPacked(TVMArgs(values.data(), codes.data(), args_size), &rv);
  Module mod = rv.operator Module();
  // debug graph runtime is one child class of graph runtime.
  SetParams(const_cast<GraphRuntime*>(mod.as<GraphRuntime>()), this->params_);
  return mod;
}

Module GraphRuntimeFactoryModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string graph_json;
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
  std::string module_name;
  CHECK(stream->Read(&graph_json));
  uint64_t sz;
  CHECK(stream->Read(&sz));
  std::vector<std::string> names;
  CHECK(stream->Read(&names));
  CHECK(sz == names.size());
  for (size_t i = 0; i < sz; ++i) {
    tvm::runtime::NDArray temp;
    temp.Load(stream);
    params[names[i]] = temp;
  }
  CHECK(stream->Read(&module_name));
  auto exec = make_object<GraphRuntimeFactory>(graph_json, params, module_name);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.graph_runtime_factory.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  CHECK_GE(args.num_args, 3) << "The expected number of arguments for "
                                "graph_runtime_factory.create needs at least 3, "
                                "but it has "
                             << args.num_args;
  // The argument order is graph_json, module, module_name, params.
  CHECK_EQ((args.size() - 3) % 2, 0);
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
  for (size_t i = 3; i < static_cast<size_t>(args.size()); i += 2) {
    std::string name = args[i].operator String();
    params[name] = args[i + 1].operator tvm::runtime::NDArray();
  }
  auto exec = make_object<GraphRuntimeFactory>(args[0], params, args[2]);
  exec->Import(args[1]);
  *rv = Module(exec);
});

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_GraphRuntimeFactory")
    .set_body_typed(GraphRuntimeFactoryModuleLoadBinary);

}  // namespace runtime
}  // namespace tvm
