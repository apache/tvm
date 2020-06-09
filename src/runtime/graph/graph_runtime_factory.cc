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

#include <tvm/runtime/registry.h>
#include <tvm/node/container.h>
#include <vector>
#include <algorithm>
#include <iterator>
#include "./graph_runtime_factory.h"
#include "./graph_runtime.h"

namespace tvm {
namespace runtime {

void GraphRuntimeFactory::Init(const std::string& kind,
                               const std::string& graph_json,
                               const std::unordered_map<std::string, tvm::runtime::NDArray>& params) {
  kind_ = kind;
  graph_json_ = graph_json;
  params_ = params;
}

void GraphRuntimeFactory::ImportModule(Module other, std::string module_name) {
  this->Import(other);
  module_names_.push_back(module_name);
}

PackedFunc GraphRuntimeFactory::GetFunction(const std::string& name,
                                            const tvm::runtime::ObjectPtr<tvm::runtime::Object>& sptr_to_self) {
  if (name == "runtime_create") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      std::vector<TVMContext> contexts;
      TVMContext ctx;
      // arg is: module, ctxs
      CHECK_EQ((args.size() - 1) % 2, 0);
      for (int i = 1; i < args.num_args; i += 2) {
        int dev_type = args[i];
        ctx.device_type = static_cast<DLDeviceType>(dev_type);
        ctx.device_id = args[i + 1];
        contexts.push_back(ctx);
      }
      *rv = this->RuntimeCreate(args[0], contexts);
    });
  } else if (name == "import_module") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args.size(), 2);
      this->ImportModule(args[0], args[1]);
    });
  } else if (name == "select_module") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK_EQ(args.size(), 1);
      *rv = this->SelectModule(args[0]);
    });
  } else if (name == "get_json") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = this->graph_json_;
    });
  } else if (name == "get_lib") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK_GT(this->imports().size(), 0);
      *rv = this->imports_[0];
    });
  } else if (name == "get_params") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      Map<String, tvm::runtime::NDArray> ret;
      for (const auto& kv : this->params_) {
        ret.Set(kv.first, kv.second);
      }
      *rv = ret;
    });
  } else {
    return PackedFunc();
  }
}

void GraphRuntimeFactory::SaveToBinary(dmlc::Stream* stream) {
  stream->Write(module_names_);
  stream->Write(kind_);
  stream->Write(graph_json_);
  std::vector<std::string> names;
  std::vector<DLTensor*> arrays;
  for (const auto& v : params_) {
    names.emplace_back(v.first);
    arrays.emplace_back(const_cast<DLTensor*>(v.second.operator->()));
  }
  stream->Write(names);
  uint64_t sz = arrays.size();
  CHECK(sz == names.size());
  stream->Write(sz);
  for (size_t i = 0; i < sz; ++i) {
    tvm::runtime::SaveDLTensor(stream, arrays[i]);
  }
}

Module GraphRuntimeFactory::RuntimeCreate(Module module, const std::vector<TVMContext> &ctxs) {
  auto factory_module = module.as<GraphRuntimeFactory>();
  CHECK(factory_module != nullptr);
  if (factory_module->GetKind() == "graph") {
    auto exec = make_object<GraphRuntime>();
    exec->Init(factory_module->GetJson(), factory_module->GetLib(), ctxs);
    exec->SetParams(factory_module->GetParams());
    return Module(exec);
  }

  return Module();
}

Module GraphRuntimeFactory::SelectModule(const std::string &name) {
  CHECK(std::find(module_names_.begin(), module_names_.end(), name) != module_names_.end());
  auto iter = std::find(module_names_.begin(), module_names_.end(), name);
  CHECK(iter != module_names_.end());
  if (iter == module_names_.begin()) {
    auto exec = make_object<GraphRuntimeFactory>();
    exec->Init(this->kind_, this->graph_json_, this->params_);
    exec->ImportModule(this->imports_[0], *iter);
    return Module(exec);
  } else {
    return this->imports_[std::distance(module_names_.begin(), iter)];
  }
}

Module GraphRuntimeFactoryModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::vector<std::string> module_names;
  std::string kind;
  std::string graph_json;
  CHECK(stream->Read(&module_names));
  CHECK(stream->Read(&kind));
  CHECK(stream->Read(&graph_json));
  std::vector<std::string> names;
  CHECK(stream->Read(&names));
  uint64_t sz;
  CHECK(stream->Read(&sz));
  CHECK(sz == names.size());
  std::unordered_map<std::string, tvm::runtime::NDArray> params;
  for (size_t i = 0; i < sz; ++i) {
    tvm::runtime::NDArray temp;
    temp.Load(stream);
    params[names[i]] = temp;
  }

  auto exec = make_object<GraphRuntimeFactory>();
  exec->Init(kind, graph_json, params);
  exec->SetModuleNames(module_names);
  return Module(exec);
}

Module RuntimeCreate(Module module, const std::vector<TVMContext> &ctxs) {
  auto mod = module.as<GraphRuntimeFactory>();
  CHECK(mod != nullptr);
  if (mod->GetKind() == "graph") {
    auto exec = make_object<GraphRuntime>();
    exec->Init(mod->GetJson(), mod->GetLib(), ctxs);
    exec->SetParams(mod->GetParams());
    return Module(exec);
  } else {
    LOG(ERROR) << "Doesn't support graph kind of " << mod->GetKind();
  }

  return Module();
}

TVM_REGISTER_GLOBAL("tvm.graph_runtime_factory.create")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    CHECK_GE(args.num_args, 4) << "The expected number of arguments for "
                                       "graph_runtime_factory.create needs at least 3, "
                                       "but it has "
                                    << args.num_args;
    auto exec = make_object<GraphRuntimeFactory>();
    // The argument order is graph_runtime_kind, graph_json, module, module_name, params.
    CHECK_EQ((args.size() - 4) % 2, 0);
    std::unordered_map<std::string, tvm::runtime::NDArray> params;
    for (size_t i = 4; i < static_cast<size_t>(args.size()); i += 2) {
      std::string name = args[i].operator String();
      params[name] = args[i + 1].operator tvm::runtime::NDArray();
    }
    exec->Init(args[0], args[1], params);
    exec->ImportModule(args[2], args[3]);
    *rv = Module(exec);
  });

TVM_REGISTER_GLOBAL("tvm.graph_runtime_factory.runtime_create")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  std::vector<TVMContext> contexts;
  TVMContext ctx;
  // arg is: module, ctxs
  CHECK_EQ((args.size() - 1) % 2, 0);
  for (int i = 1; i < args.num_args; i += 2) {
    int dev_type = args[i];
    ctx.device_type = static_cast<DLDeviceType>(dev_type);
    ctx.device_id = args[i + 1];
    contexts.push_back(ctx);
  }
  *rv = RuntimeCreate(args[0], contexts);
});

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_GraphRuntimeFactory")
.set_body_typed(GraphRuntimeFactoryModuleLoadBinary);

}  // namespace runtime
}  // namespace tvm
