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
#include <tvm/target/target_id.h>
#include <tvm/tir/expr.h>

#include <algorithm>
#include <stack>

namespace tvm {

using runtime::PackedFunc;
using runtime::TVMArgs;
using runtime::TVMRetValue;

Target Target::CreateTarget(const std::string& name, const std::vector<std::string>& options) {
  TargetId id = TargetId::Get(name);
  ObjectPtr<TargetNode> target = make_object<TargetNode>();
  target->id = id;
  // tag is always empty
  target->tag = "";
  // parse attrs
  target->attrs = id->ParseAttrsFromRaw(options);
  String device_name = target->GetAttr<String>("device", "").value();
  // set up keys
  {
    std::vector<String> keys;
    // user provided keys
    if (Optional<Array<String>> user_keys = target->GetAttr<Array<String>>("keys")) {
      keys = std::vector<String>(user_keys.value().begin(), user_keys.value().end());
      target->attrs.erase("keys");
    }
    // add `device_name`
    if (!device_name.empty()) {
      keys.push_back(device_name);
    }
    // add default keys
    for (const auto& key : target->id->default_keys) {
      keys.push_back(key);
    }
    // de-duplicate keys
    size_t new_size = 0;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (keys[i] == "") {
        continue;
      }
      keys[new_size++] = keys[i];
      for (size_t j = i + 1; j < keys.size(); ++j) {
        if (keys[j] == keys[i]) {
          keys[j] = String("");
        }
      }
    }
    keys.resize(new_size);
    target->keys = std::move(keys);
  }
  return Target(target);
}

TVM_REGISTER_NODE_TYPE(TargetNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TargetNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const TargetNode*>(node.get());
      p->stream << op->str();
    });

TVM_REGISTER_GLOBAL("target.TargetCreate").set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string name = args[0];
  std::vector<std::string> options;
  for (int i = 1; i < args.num_args; ++i) {
    std::string arg = args[i];
    options.push_back(arg);
  }

  *ret = Target::CreateTarget(name, options);
});

TVM_REGISTER_GLOBAL("target.TargetFromString").set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string target_str = args[0];
  *ret = Target::Create(target_str);
});

std::vector<std::string> TargetNode::GetKeys() const {
  std::vector<std::string> result;
  for (auto& expr : keys) {
    result.push_back(expr);
  }
  return result;
}

std::unordered_set<std::string> TargetNode::GetLibs() const {
  Optional<Array<String>> libs = this->GetAttr<Array<String>>("libs");
  if (!libs.defined()) {
    return {};
  }
  std::unordered_set<std::string> result;
  for (const auto& item : libs.value()) {
    result.insert(item);
  }
  return result;
}

const std::string& TargetNode::str() const {
  if (str_repr_.empty()) {
    std::ostringstream os;
    os << id->name;
    if (!this->keys.empty()) {
      os << " -keys=";
      bool is_first = true;
      for (const String& s : keys) {
        if (is_first) {
          is_first = false;
        } else {
          os << ',';
        }
        os << s;
      }
    }
    if (Optional<String> attrs_str = id->StringifyAttrsToRaw(attrs)) {
      os << ' ' << attrs_str.value();
    }
    str_repr_ = os.str();
  }
  return str_repr_;
}

bool StartsWith(const std::string& str, const std::string& pattern) {
  return str.compare(0, pattern.length(), pattern) == 0;
}

Target Target::Create(const String& target_str) {
  std::vector<std::string> splits;
  std::istringstream is(target_str);
  for (std::string s; is >> s; splits.push_back(s)) {
  }
  CHECK(!splits.empty()) << "ValueError: Cannot parse empty target string: \"" << target_str
                         << "\"";
  return CreateTarget(splits[0], {splits.begin() + 1, splits.end()});
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

Target llvm(const std::vector<std::string>& options) {
  return Target::CreateTarget("llvm", options);
}

Target cuda(const std::vector<std::string>& options) {
  return Target::CreateTarget("cuda", options);
}

Target rocm(const std::vector<std::string>& options) {
  return Target::CreateTarget("rocm", options);
}

Target opencl(const std::vector<std::string>& options) {
  return Target::CreateTarget("opencl", options);
}

Target metal(const std::vector<std::string>& options) {
  return Target::CreateTarget("metal", options);
}

Target mali(const std::vector<std::string>& options) {
  return Target::CreateTarget("opencl", MergeOptions(options, {"-device=mali"}));
}

Target intel_graphics(const std::vector<std::string>& options) {
  return Target::CreateTarget(
      "opencl", MergeOptions(options, {"-device=intel_graphics", "-thread_warp_size=16"}));
}

Target stackvm(const std::vector<std::string>& options) {
  return Target::CreateTarget("stackvm", options);
}

Target ext_dev(const std::vector<std::string>& options) {
  return Target::CreateTarget("ext_dev", options);
}

Target hexagon(const std::vector<std::string>& options) {
  return Target::CreateTarget("hexagon", options);
}
}  // namespace target
}  // namespace tvm
