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
 * \file module.cc
 * \brief TVM module system
 */
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <unordered_set>
#include <cstring>
#include "file_util.h"

namespace tvm {
namespace runtime {

void ModuleNode::Import(Module other) {
  // specially handle rpc
  if (!std::strcmp(this->type_key(), "rpc")) {
    static const PackedFunc* fimport_ = nullptr;
    if (fimport_ == nullptr) {
      fimport_ = runtime::Registry::Get("rpc.ImportRemoteModule");
      CHECK(fimport_ != nullptr);
    }
    (*fimport_)(GetRef<Module>(this), other);
    return;
  }
  // cyclic detection.
  std::unordered_set<const ModuleNode*> visited{other.operator->()};
  std::vector<const ModuleNode*> stack{other.operator->()};
  while (!stack.empty()) {
    const ModuleNode* n = stack.back();
    stack.pop_back();
    for (const Module& m : n->imports_) {
      const ModuleNode* next = m.operator->();
      if (visited.count(next)) continue;
      visited.insert(next);
      stack.push_back(next);
    }
  }
  CHECK(!visited.count(this))
      << "Cyclic dependency detected during import";
  this->imports_.emplace_back(std::move(other));
}

PackedFunc ModuleNode::GetFunction(const std::string& name, bool query_imports) {
  ModuleNode* self = this;
  PackedFunc pf = self->GetFunction(name, GetObjectPtr<Object>(this));
  if (pf != nullptr) return pf;
  if (query_imports) {
    for (Module& m : self->imports_) {
      pf = m->GetFunction(name, m.data_);
      if (pf != nullptr) return pf;
    }
  }
  return pf;
}

Module Module::LoadFromFile(const std::string& file_name,
                            const std::string& format) {
  std::string fmt = GetFileFormat(file_name, format);
  CHECK(fmt.length() != 0)
      << "Cannot deduce format of file " << file_name;
  if (fmt == "dll" || fmt == "dylib" || fmt == "dso") {
    fmt = "so";
  }
  std::string load_f_name = "runtime.module.loadfile_" + fmt;
  const PackedFunc* f = Registry::Get(load_f_name);
  CHECK(f != nullptr)
      << "Loader of " << format << "("
      << load_f_name << ") is not presented.";
  Module m = (*f)(file_name, format);
  return m;
}

void ModuleNode::SaveToFile(const std::string& file_name,
                            const std::string& format) {
  LOG(FATAL) << "Module[" << type_key() << "] does not support SaveToFile";
}

void ModuleNode::SaveToBinary(dmlc::Stream* stream) {
  LOG(FATAL) << "Module[" << type_key() << "] does not support SaveToBinary";
}

std::string ModuleNode::GetSource(const std::string& format) {
  LOG(FATAL) << "Module[" << type_key() << "] does not support GetSource";
  return "";
}

const PackedFunc* ModuleNode::GetFuncFromEnv(const std::string& name) {
  auto it = import_cache_.find(name);
  if (it != import_cache_.end()) return it->second.get();
  PackedFunc pf;
  for (Module& m : this->imports_) {
    pf = m.GetFunction(name, true);
    if (pf != nullptr) break;
  }
  if (pf == nullptr) {
    const PackedFunc* f = Registry::Get(name);
    CHECK(f != nullptr)
        << "Cannot find function " << name
        << " in the imported modules or global registry";
    return f;
  } else {
    import_cache_.insert(std::make_pair(name, std::make_shared<PackedFunc>(pf)));
    return import_cache_.at(name).get();
  }
}

bool RuntimeEnabled(const std::string& target) {
  std::string f_name;
  if (target == "cpu") {
    return true;
  } else if (target == "cuda" || target == "gpu") {
    f_name = "device_api.gpu";
  } else if (target == "cl" || target == "opencl" || target == "sdaccel") {
    f_name = "device_api.opencl";
  } else if (target == "gl" || target == "opengl") {
    f_name = "device_api.opengl";
  } else if (target == "mtl" || target == "metal") {
    f_name = "device_api.metal";
  } else if (target == "vulkan") {
    f_name = "device_api.vulkan";
  } else if (target == "stackvm") {
    f_name = "target.build.stackvm";
  } else if (target == "rpc") {
    f_name = "device_api.rpc";
  } else if (target == "micro_dev") {
    f_name = "device_api.micro_dev";
  } else if (target.length() >= 5 && target.substr(0, 5) == "nvptx") {
    f_name = "device_api.gpu";
  } else if (target.length() >= 4 && target.substr(0, 4) == "rocm") {
    f_name = "device_api.rocm";
  } else if (target.length() >= 4 && target.substr(0, 4) == "llvm") {
    const PackedFunc* pf = runtime::Registry::Get("codegen.llvm_target_enabled");
    if (pf == nullptr) return false;
    return (*pf)(target);
  } else {
    LOG(FATAL) << "Unknown optional runtime " << target;
  }
  return runtime::Registry::Get(f_name) != nullptr;
}

TVM_REGISTER_GLOBAL("runtime.RuntimeEnabled")
.set_body_typed(RuntimeEnabled);

TVM_REGISTER_GLOBAL("runtime.ModuleGetSource")
.set_body_typed([](Module mod, std::string fmt) {
  return mod->GetSource(fmt);
});

TVM_REGISTER_GLOBAL("runtime.ModuleImportsSize")
.set_body_typed([](Module mod) {
  return static_cast<int64_t>(mod->imports().size());
});

TVM_REGISTER_GLOBAL("runtime.ModuleGetImport")
.set_body_typed([](Module mod, int index) {
  return mod->imports().at(index);
});

TVM_REGISTER_GLOBAL("runtime.ModuleGetTypeKey")
.set_body_typed([](Module mod) {
  return std::string(mod->type_key());
});

TVM_REGISTER_GLOBAL("runtime.ModuleLoadFromFile")
.set_body_typed(Module::LoadFromFile);

TVM_REGISTER_GLOBAL("runtime.ModuleSaveToFile")
.set_body_typed([](Module mod, std::string name, std::string fmt) {
  mod->SaveToFile(name, fmt);
});

TVM_REGISTER_OBJECT_TYPE(ModuleNode);
}  // namespace runtime
}  // namespace tvm
