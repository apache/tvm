/*!
 *  Copyright (c) 2017 by Contributors
 * \file module.cc
 * \brief TVM module system
 */
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <unordered_set>
#include <cstring>
#include "./file_util.h"

namespace tvm {
namespace runtime {

PackedFunc Module::GetFunction(
    const std::string& name, bool query_imports) {
  PackedFunc pf = node_->GetFunction(name, node_);
  if (pf != nullptr) return pf;
  if (query_imports) {
    for (const Module& m : node_->imports_) {
      pf = m.node_->GetFunction(name, m.node_);
      if (pf != nullptr) return pf;
    }
  }
  return pf;
}

void Module::Import(Module other) {
  // specially handle rpc
  if (!std::strcmp((*this)->type_key(), "rpc")) {
    static const PackedFunc* fimport_ = nullptr;
    if (fimport_ == nullptr) {
      fimport_ = runtime::Registry::Get("contrib.rpc._ImportRemoteModule");
      CHECK(fimport_ != nullptr);
    }
    (*fimport_)(*this, other);
    return;
  }
  // cyclic detection.
  std::unordered_set<const ModuleNode*> visited{other.node_.get()};
  std::vector<const ModuleNode*> stack{other.node_.get()};
  while (!stack.empty()) {
    const ModuleNode* n = stack.back();
    stack.pop_back();
    for (const Module& m : n->imports_) {
      const ModuleNode* next = m.node_.get();
      if (visited.count(next)) continue;
      visited.insert(next);
      stack.push_back(next);
    }
  }
  CHECK(!visited.count(node_.get()))
      << "Cyclic dependency detected during import";
  node_->imports_.emplace_back(std::move(other));
}

Module Module::LoadFromFile(const std::string& file_name,
                            const std::string& format) {
  std::string fmt = GetFileFormat(file_name, format);
  CHECK(fmt.length() != 0)
      << "Cannot deduce format of file " << file_name;
  if (fmt == "dll" || fmt == "dylib" || fmt == "dso") {
    fmt = "so";
  }
  std::string load_f_name = "module.loadfile_" + fmt;
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
    pf = m.GetFunction(name, false);
    if (pf != nullptr) break;
  }
  if (pf == nullptr) {
    const PackedFunc* f = Registry::Get(name);
    CHECK(f != nullptr)
        << "Cannot find function " << name
        << " in the imported modules or global registry";
    return f;
  } else {
    std::unique_ptr<PackedFunc> f(new PackedFunc(pf));
    import_cache_[name] = std::move(f);
    return import_cache_.at(name).get();
  }
}

bool RuntimeEnabled(const std::string& target) {
  std::string f_name;
  if (target == "cpu") {
    return true;
  } else if (target == "cuda" || target == "gpu") {
    f_name = "device_api.gpu";
  } else if (target == "cl" || target == "opencl") {
    f_name = "device_api.opencl";
  } else if (target == "mtl" || target == "metal") {
    f_name = "device_api.metal";
  } else if (target == "stackvm") {
    f_name = "codegen.build_stackvm";
  } else if (target == "rpc") {
    f_name = "device_api.rpc";
  } else if (target == "vpi" || target == "verilog") {
    f_name = "device_api.vpi";
  } else if (target.length() >= 5 && target.substr(0, 5) == "nvptx") {
    f_name = "codegen.build_nvptx";
  } else if (target.length() >= 4 && target.substr(0, 4) == "rocm") {
    f_name = "codegen.build_rocm";
  } else if (target.length() >= 4 && target.substr(0, 4) == "llvm") {
    const PackedFunc* pf = runtime::Registry::Get("codegen.llvm_target_enabled");
    if (pf == nullptr) return false;
    return (*pf)(target);
  } else {
    LOG(FATAL) << "Unknown optional runtime " << target;
  }
  return runtime::Registry::Get(f_name) != nullptr;
}

TVM_REGISTER_GLOBAL("module._Enabled")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = RuntimeEnabled(args[0]);
    });

TVM_REGISTER_GLOBAL("module._GetSource")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = args[0].operator Module()->GetSource(args[1]);
    });

TVM_REGISTER_GLOBAL("module._ImportsSize")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = static_cast<int64_t>(
        args[0].operator Module()->imports().size());
    });

TVM_REGISTER_GLOBAL("module._GetImport")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = args[0].operator Module()->
        imports().at(args[1].operator int());
    });

TVM_REGISTER_GLOBAL("module._GetTypeKey")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = std::string(args[0].operator Module()->type_key());
    });

TVM_REGISTER_GLOBAL("module._LoadFromFile")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    *ret = Module::LoadFromFile(args[0], args[1]);
    });

TVM_REGISTER_GLOBAL("module._SaveToFile")
.set_body([](TVMArgs args, TVMRetValue *ret) {
    args[0].operator Module()->
        SaveToFile(args[1], args[2]);
    });
}  // namespace runtime
}  // namespace tvm
