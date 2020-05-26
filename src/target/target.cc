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
 *  Compile executable modules.
 * \file src/target/target.cc
 */
#include <dmlc/thread_local.h>
#include <tvm/node/repr_printer.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/expr.h>

#include <algorithm>
#include <stack>

namespace tvm {

using runtime::PackedFunc;
using runtime::TVMArgs;
using runtime::TVMRetValue;

TVM_REGISTER_NODE_TYPE(TargetNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TargetNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const TargetNode*>(node.get());
      p->stream << op->str();
    });

/*!
 * \brief Construct a Target node from the given name and options.
 * \param target_name The major target name. Should be one of
 * {"aocl", "aocl_sw_emu", "c", "cuda", "ext_dev", "hexagon", "hybrid", "llvm",
 *  "metal", "nvptx", "opencl", "rocm", "sdaccel", "stackvm", "vulkan"}
 * \param options Additional options appended to the target
 * \return The constructed Target
 */
Target CreateTarget(const std::string& target_name, const std::vector<std::string>& options) {
  auto t = make_object<TargetNode>();
  t->target_name = target_name;

  std::string libs_flag = "-libs=";
  std::string device_flag = "-device=";
  std::string keys_flag = "-keys=";
  for (auto& item : options) {
    t->options_array.push_back(item);

    if (item.find(libs_flag) == 0) {
      std::stringstream ss(item.substr(libs_flag.length()));
      std::string lib_item;
      while (std::getline(ss, lib_item, ',')) {
        t->libs_array.push_back(lib_item);
      }
    } else if (item.find(device_flag) == 0) {
      t->device_name = item.substr(device_flag.length());
      t->keys_array.push_back(t->device_name);
    } else if (item.find(keys_flag) == 0) {
      std::stringstream ss(item.substr(keys_flag.length()));
      std::string key_item;
      while (std::getline(ss, key_item, ',')) {
        t->keys_array.push_back(key_item);
      }
    }
  }

  if (t->device_name.length() > 0) {
    t->keys_array.push_back(t->device_name);
  }
  t->device_type = kDLCPU;
  t->thread_warp_size = 1;
  if (target_name == "c" && t->device_name == "micro_dev") {
    t->device_type = kDLMicroDev;
  } else if (target_name == "c" || target_name == "llvm") {
    t->keys_array.push_back("cpu");
  } else if (target_name == "cuda" || target_name == "nvptx") {
    t->device_type = kDLGPU;
    t->keys_array.push_back("cuda");
    t->keys_array.push_back("gpu");
    t->max_num_threads = 1024;
    t->thread_warp_size = 32;
  } else if (target_name == "rocm" || target_name == "opencl") {
    // For now assume rocm schedule for opencl
    if (target_name == "opencl") {
      t->device_type = kDLOpenCL;
    } else {
      t->device_type = kDLROCM;
    }
    t->keys_array.push_back(target_name);
    t->keys_array.push_back("gpu");
    t->max_num_threads = 256;
    if (t->device_name == "intel_graphics") {
      t->thread_warp_size = 16;
    }
  } else if (target_name == "metal" || target_name == "vulkan" || target_name == "webgpu") {
    if (target_name == "metal") {
      t->device_type = kDLMetal;
    } else if (target_name == "vulkan") {
      t->device_type = kDLVulkan;
    } else {
      t->device_type = kDLWebGPU;
    }
    t->keys_array.push_back(target_name);
    t->keys_array.push_back("gpu");
    t->max_num_threads = 256;
  } else if (target_name == "sdaccel") {
    t->device_type = kDLOpenCL;
    t->keys_array.push_back("sdaccel");
    t->keys_array.push_back("hls");
  } else if (target_name == "aocl" || target_name == "aocl_sw_emu") {
    t->device_type = kDLAOCL;
    t->keys_array.push_back("aocl");
    t->keys_array.push_back("hls");
  } else if (target_name == "stackvm") {
    t->device_type = kDLCPU;
  } else if (target_name == "ext_dev") {
    t->device_type = kDLExtDev;
  } else if (target_name == "hybrid") {
    t->device_type = kDLCPU;
  } else if (target_name == "hexagon") {
    t->keys_array.push_back("hexagon");
    t->device_type = kDLHexagon;
  } else if (target_name == "webgpu") {
    t->keys_array.push_back("webgpu");
    t->device_type = kDLWebGPU;
  } else {
    LOG(ERROR) << "Unknown target name " << target_name << "; falling back to stackvm";
    return target::stackvm();
  }

  return Target(t);
}

TVM_REGISTER_GLOBAL("target.TargetCreate").set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string target_name = args[0];
  std::vector<std::string> options;
  for (int i = 1; i < args.num_args; ++i) {
    std::string arg = args[i];
    options.push_back(arg);
  }

  *ret = CreateTarget(target_name, options);
});

TVM_REGISTER_GLOBAL("target.TargetFromString").set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string target_str = args[0];
  *ret = Target::Create(target_str);
});

std::vector<std::string> TargetNode::keys() const {
  std::vector<std::string> result;
  for (auto& expr : keys_array) {
    result.push_back(expr);
  }
  return result;
}

std::vector<std::string> TargetNode::options() const {
  std::vector<std::string> result;
  for (auto& expr : options_array) {
    result.push_back(expr);
  }
  return result;
}

std::unordered_set<std::string> TargetNode::libs() const {
  std::unordered_set<std::string> result;
  for (auto& expr : libs_array) {
    result.insert(expr);
  }
  return result;
}

const std::string& TargetNode::str() const {
  if (str_repr_.length() != 0) return str_repr_;
  std::ostringstream result;
  result << target_name;
  for (const auto& x : options()) {
    result << " " << x;
  }
  str_repr_ = result.str();
  return str_repr_;
}

bool StartsWith(const std::string& str, const std::string& pattern) {
  return str.compare(0, pattern.length(), pattern) == 0;
}

std::string GetDeviceName(const std::string& target_str) {
  std::istringstream ss(target_str);
  std::string target_name;
  ss >> target_name;

  std::string item;
  while (ss >> item) {
    if (StartsWith(item, "-device=")) {
      return item.substr(std::string("-device=").length());
    }
  }

  return "";
}

Target Target::Create(const std::string& target_str) {
  if (target_str.length() == 0) {
    LOG(ERROR) << "target_str must not be empty";
  }

  std::istringstream ss(target_str);
  std::string target_name;

  ss >> target_name;
  auto device_name = GetDeviceName(target_str);

  std::vector<std::string> options;
  std::string item;
  while (ss >> item) {
    options.push_back(item);
  }

  return CreateTarget(target_name, options);
}

/*! \brief Entry to hold the Target context stack. */
struct TVMTargetThreadLocalEntry {
  /*! \brief The current target context */
  std::stack<tvm::Target> context_stack;
};

/*! \brief Thread local store to hold the Target context stack. */
typedef dmlc::ThreadLocalStore<TVMTargetThreadLocalEntry> TVMTargetThreadLocalStore;

void Target::EnterWithScope() {
  TVMTargetThreadLocalEntry* entry = TVMTargetThreadLocalStore::Get();
  entry->context_stack.push(*this);
}

void Target::ExitWithScope() {
  TVMTargetThreadLocalEntry* entry = TVMTargetThreadLocalStore::Get();
  CHECK(!entry->context_stack.empty());
  CHECK(entry->context_stack.top().same_as(*this));
  entry->context_stack.pop();
}

tvm::Target Target::Current(bool allow_not_defined) {
  TVMTargetThreadLocalEntry* entry = TVMTargetThreadLocalStore::Get();
  if (entry->context_stack.size() > 0) {
    return entry->context_stack.top();
  }
  CHECK(allow_not_defined)
      << "Target context required. Please set it by constructing a TargetContext";

  return Target();
}

TVM_REGISTER_GLOBAL("target.GetCurrentTarget").set_body([](TVMArgs args, TVMRetValue* ret) {
  bool allow_not_defined = args[0];
  *ret = Target::Current(allow_not_defined);
});
class Target::Internal {
 public:
  static void EnterScope(Target target) { target.EnterWithScope(); }
  static void ExitScope(Target target) { target.ExitWithScope(); }
};

TVM_REGISTER_GLOBAL("target.EnterTargetScope").set_body_typed(Target::Internal::EnterScope);

TVM_REGISTER_GLOBAL("target.ExitTargetScope").set_body_typed(Target::Internal::ExitScope);

namespace target {
std::vector<std::string> MergeOptions(std::vector<std::string> opts,
                                      const std::vector<std::string>& new_opts) {
  opts.insert(opts.end(), new_opts.begin(), new_opts.end());
  return opts;
}

Target llvm(const std::vector<std::string>& options) { return CreateTarget("llvm", options); }

Target cuda(const std::vector<std::string>& options) { return CreateTarget("cuda", options); }

Target rocm(const std::vector<std::string>& options) { return CreateTarget("rocm", options); }

Target opencl(const std::vector<std::string>& options) { return CreateTarget("opencl", options); }

Target metal(const std::vector<std::string>& options) { return CreateTarget("metal", options); }

Target mali(const std::vector<std::string>& options) {
  return CreateTarget("opencl", MergeOptions(options, {"-device=mali"}));
}

Target intel_graphics(const std::vector<std::string>& options) {
  return CreateTarget("opencl", MergeOptions(options, {"-device=intel_graphics"}));
}

Target stackvm(const std::vector<std::string>& options) { return CreateTarget("stackvm", options); }

Target ext_dev(const std::vector<std::string>& options) { return CreateTarget("ext_dev", options); }

Target hexagon(const std::vector<std::string>& options) { return CreateTarget("hexagon", options); }
}  // namespace target
}  // namespace tvm
