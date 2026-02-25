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
#include <tvm/ffi/extra/json.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/transform.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/target/tag.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>
#include <tvm/tir/expr.h>

#include <algorithm>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {

TVM_FFI_STATIC_INIT_BLOCK() { TargetNode::RegisterReflection(); }

class TargetInternal {
 public:
  static void EnterScope(Target target) { target.EnterWithScope(); }
  static void ExitScope(Target target) { target.ExitWithScope(); }
  static ffi::Map<ffi::String, ffi::Any> ToConfig(Target target) { return target->ToConfig(); }
  static ObjectPtr<TargetNode> FromString(const ffi::String& tag_or_config_or_target_str);
  static ObjectPtr<TargetNode> FromConfigString(const ffi::String& config_str);
  static ObjectPtr<TargetNode> FromConfig(ffi::Map<ffi::String, ffi::Any> config);
  static void ConstructorDispatcher(ffi::PackedArgs args, ffi::Any* rv);
  static Target WithHost(const Target& target, const Target& target_host) {
    ObjectPtr<TargetNode> n = ffi::make_object<TargetNode>(*target.get());
    n->host = target_host;
    return (Target)n;
  }

 private:
  static std::unordered_map<ffi::String, ffi::Any> QueryDevice(int device_id,
                                                               const TargetNode* target);
};

/**********  Helper functions  **********/

Target Target::WithHost(const Target& target, const Target& host) {
  return TargetInternal::WithHost(target, host);
}

void CheckAndUpdateHostConsistency(Target* target, Target* host) {
  *target = Target(*target, *host);
  *host = (*target)->GetHost().value_or(Target());
}

static std::vector<ffi::String> DeduplicateKeys(const std::vector<ffi::String>& keys) {
  std::vector<ffi::String> new_keys;
  for (size_t i = 0; i < keys.size(); ++i) {
    bool found = false;
    for (size_t j = 0; j < i; ++j) {
      if (keys[i] == keys[j]) {
        found = true;
        break;
      }
    }
    if (!found) {
      new_keys.push_back(keys[i]);
    }
  }
  return new_keys;
}

static TargetKind GetTargetKind(const ffi::String& name) {
  ffi::Optional<TargetKind> kind = TargetKind::Get(name);
  if (!kind.defined()) {
    TVM_FFI_THROW(TypeError) << "Target kind \"" + name + "\" is not defined";
  }
  return kind.value();
}

const std::string& TargetNode::str() const {
  if (str_repr_.empty()) {
    str_repr_ = std::string(ffi::json::Stringify(ToConfig()));
  }
  return str_repr_;
}

/**********  Small member methods  **********/

Target::Target(const ffi::String& tag_or_config_or_target_str) {
  ObjectPtr<Object> target;
  try {
    target = TargetInternal::FromString(tag_or_config_or_target_str);
  } catch (const Error& e) {
    std::ostringstream os;
    os << ". Target creation from string failed: " << tag_or_config_or_target_str;
    throw Error("ValueError", e.message() + os.str(), e.backtrace());
  }
  data_ = std::move(target);
}

Target::Target(const ffi::Map<ffi::String, ffi::Any>& config) {
  ObjectPtr<Object> target;
  try {
    target = TargetInternal::FromConfig(config);
  } catch (const Error& e) {
    std::ostringstream os;
    os << ". Target creation from config dict failed: " << config;
    throw Error("ValueError", std::string(e.message()) + os.str(), e.backtrace());
  }
  data_ = std::move(target);
}

Target::Target(Target target, Target host) {
  ObjectPtr<TargetNode> n = ffi::make_object<TargetNode>(*target.get());
  n->host = std::move(host);
  data_ = std::move(n);
}

Target::Target(TargetKind kind, ffi::Optional<ObjectRef> host, ffi::String tag,
               ffi::Array<ffi::String> keys, ffi::Map<ffi::String, ffi::Any> attrs) {
  auto data = ffi::make_object<TargetNode>();
  data->kind = std::move(kind);
  data->host = std::move(host);
  data->tag = std::move(tag);
  data->keys = std::move(keys);
  data->attrs = std::move(attrs);
  data_ = std::move(data);
}

ffi::Map<ffi::String, ffi::Any> TargetNode::ToConfig() const {
  ffi::Map<ffi::String, ffi::Any> result = {
      {"kind", this->kind->name},
      {"tag", this->tag},
      {"keys", this->keys},
  };
  if (this->host.defined()) {
    result.Set("host", this->GetHost().value_or(Target())->ToConfig());
  }
  for (const auto& kv : attrs) {
    result.Set(kv.first, kv.second);
  }
  return result;
}

ffi::Optional<Target> TargetNode::GetHost() const { return this->host.as<Target>(); }

Target Target::WithoutHost() const {
  if ((*this)->GetHost()) {
    auto output = ffi::make_object<TargetNode>(*get());
    output->host = std::nullopt;
    return Target(output);
  } else {
    return *this;
  }
}

int TargetNode::GetTargetDeviceType() const {
  if (ffi::Optional<Integer> device_type = GetAttr<Integer>("target_device_type")) {
    return Downcast<Integer>(device_type)->value;
  }
  return kind->default_device_type;
}

bool TargetNode::HasKey(const std::string& query_key) const {
  return std::any_of(keys.begin(), keys.end(),
                     [&query_key](const auto& key) { return key == query_key; });
}

/*! \brief Entry to hold the Target context stack. */
struct TVMTargetThreadLocalEntry {
  /*! \brief The current target context */
  std::stack<Target> context_stack;
};

/*! \brief Thread local store to hold the Target context stack. */
static TVMTargetThreadLocalEntry* TVMTargetThreadLocalStoreGet() {
  static thread_local TVMTargetThreadLocalEntry inst;
  return &inst;
}

void Target::EnterWithScope() {
  TVMTargetThreadLocalEntry* entry = TVMTargetThreadLocalStoreGet();
  entry->context_stack.push(*this);
}

void Target::ExitWithScope() {
  TVMTargetThreadLocalEntry* entry = TVMTargetThreadLocalStoreGet();
  TVM_FFI_ICHECK(!entry->context_stack.empty());
  TVM_FFI_ICHECK(entry->context_stack.top().same_as(*this));
  entry->context_stack.pop();
}

Target Target::Current(bool allow_not_defined) {
  TVMTargetThreadLocalEntry* entry = TVMTargetThreadLocalStoreGet();
  if (entry->context_stack.size() > 0) {
    return entry->context_stack.top();
  }
  TVM_FFI_ICHECK(allow_not_defined)
      << "Target context required. Please set it by constructing a TargetContext";

  return Target();
}

/**********  Creation  **********/

void TargetInternal::ConstructorDispatcher(ffi::PackedArgs args, ffi::Any* rv) {
  if (args.size() == 1) {
    const auto& arg = args[0];
    if (auto opt_target = arg.as<Target>()) {
      *rv = Target(opt_target.value());
    } else if (auto opt_str = arg.try_cast<ffi::String>()) {
      *rv = Target(opt_str.value());
    } else if (auto opt_map = arg.try_cast<ffi::Map<ffi::String, ffi::Any>>()) {
      *rv = Target(opt_map.value());
    } else {
      TVM_FFI_THROW(TypeError) << "Cannot create target with type: " << args[0].GetTypeKey();
    }
    return;
  } else if (args.size() == 2) {
    if (args[0].as<Target>().has_value() && args[1].as<Target>().has_value()) {
      Target target = args[0].cast<Target>();
      Target host = args[1].cast<Target>();
      *rv = Target(target, host);
    } else {
      TVM_FFI_THROW(ValueError) << "Invalid type of arguments. Expect 2 Target arguments.";
    }
    return;
  }
  TVM_FFI_THROW(ValueError) << "Invalid number of arguments. Expect 1 or 2, but gets: "
                            << args.size();
}

ObjectPtr<TargetNode> TargetInternal::FromString(const ffi::String& tag_or_config_or_target_str) {
  if (ffi::Optional<Target> target = TargetTag::Get(tag_or_config_or_target_str)) {
    Target value = target.value();
    return ffi::details::ObjectUnsafe::ObjectPtrFromObjectRef<TargetNode>(value);
  }
  if (!tag_or_config_or_target_str.empty() && tag_or_config_or_target_str.data()[0] == '{') {
    return TargetInternal::FromConfigString(tag_or_config_or_target_str);
  }
  // Treat as bare kind name (e.g. "llvm", "cuda"). Reject strings with spaces.
  std::string s(tag_or_config_or_target_str);
  if (s.find(' ') != std::string::npos) {
    TVM_FFI_THROW(ValueError)
        << "Cannot parse target string \"" << s
        << "\". CLI target string form (e.g. \"llvm -mcpu=xxx\") is no longer supported. "
        << "Please use JSON dict form (e.g. {\"kind\": \"llvm\", \"mcpu\": \"xxx\"}) instead.";
  }
  return TargetInternal::FromConfig(ffi::Map<ffi::String, ffi::Any>{{"kind", ffi::String(s)}});
}

ObjectPtr<TargetNode> TargetInternal::FromConfigString(const ffi::String& config_str) {
  ffi::String error_msg;
  ffi::json::Value parsed = ffi::json::Parse(config_str, &error_msg);
  if (error_msg.size() > 0) {
    TVM_FFI_THROW(ValueError) << "Failed to parse target JSON config: " << error_msg;
  }
  auto config = parsed.as<ffi::Map<ffi::String, ffi::Any>>();
  if (!config.has_value()) {
    TVM_FFI_THROW(ValueError) << "Target JSON config must be a dict, got: " << config_str;
  }
  return TargetInternal::FromConfig(config.value());
}

ObjectPtr<TargetNode> TargetInternal::FromConfig(ffi::Map<ffi::String, ffi::Any> config) {
  const ffi::String kKind = "kind";
  const ffi::String kTag = "tag";
  const ffi::String kKeys = "keys";
  const ffi::String kDeviceName = "device";
  const ffi::String kHost = "host";
  const ffi::String kFromDevice = "from_device";
  ObjectPtr<TargetNode> target = ffi::make_object<TargetNode>();

  // Step 0: If "tag" is present without "kind", look up the tag config and merge overrides on top
  if (!config.count(kKind) && config.count(kTag)) {
    auto tag_name = config[kTag].try_cast<ffi::String>();
    TVM_FFI_ICHECK(tag_name.has_value())
        << "Expect type of field \"tag\" is String, but get type: " << config[kTag].GetTypeKey();
    auto tag_config = TargetTag::GetConfig(tag_name.value());
    TVM_FFI_ICHECK(tag_config.has_value()) << "Unknown target tag: " << tag_name.value();
    // Start from the tag's base config, then apply user overrides
    ffi::Map<ffi::String, ffi::Any> merged = tag_config.value();
    for (const auto& kv : config) {
      if (kv.first != kTag) {
        merged.Set(kv.first, kv.second);
      }
    }
    merged.Set(kTag, ffi::String(tag_name.value()));
    config = std::move(merged);
  }

  // Step 1: Parse 'kind' (needed to look up the schema, but kept in config for canonicalizer)
  if (config.count(kKind)) {
    if (auto kind = config[kKind].try_cast<ffi::String>()) {
      target->kind = GetTargetKind(kind.value());
    } else {
      TVM_FFI_THROW(TypeError) << "Expect type of field \"kind\" is String, but get type: "
                               << config[kKind].GetTypeKey();
    }
  } else {
    TVM_FFI_THROW(ValueError) << "Field \"kind\" is not found";
  }

  // Step 2: Extract "host" before schema validation (needs special recursive parsing)
  if (config.count(kHost)) {
    target->host = ffi::Function(ConstructorDispatcher)(config[kHost]).cast<Target>();
    config.erase(kHost);
  } else {
    target->host = std::nullopt;
  }

  // Step 3: Use ConfigSchema to validate types, apply defaults, and run canonicalizer
  // Note: structural keys (kind, tag, keys, device) pass through to canonicalizer
  ffi::Map<ffi::String, ffi::Any> resolved = target->kind->schema_.Resolve(config);

  // Step 4: Extract structural fields from resolved config
  if (resolved.count(kTag)) {
    if (auto tag = resolved[kTag].try_cast<ffi::String>()) {
      target->tag = tag.value();
    }
    resolved.erase(kTag);
  } else {
    target->tag = "";
  }

  {
    std::vector<ffi::String> keys;
    bool has_keys = resolved.count(kKeys);
    if (has_keys) {
      ffi::Array<ffi::String> cfg_keys = Downcast<ffi::Array<ffi::String>>(resolved.at(kKeys));
      for (const ffi::String& key : cfg_keys) {
        keys.push_back(key);
      }
    }
    if (resolved.count(kDeviceName)) {
      if (auto device = resolved.at(kDeviceName).try_cast<ffi::String>()) {
        keys.push_back(device.value());
      }
    }
    if (!has_keys) {
      for (const auto& key : target->kind->default_keys) {
        keys.push_back(key);
      }
    }
    target->keys = DeduplicateKeys(keys);
    resolved.erase(kKeys);
  }

  // Step 5: Build attrs from resolved entries (excluding structural keys)
  resolved.erase(kKind);
  std::unordered_map<ffi::String, ffi::Any> attrs;
  for (const auto& kv : resolved) {
    attrs[kv.first] = kv.second;
  }

  // Step 6: If requested, query attributes from the device. User-specified
  // parameters take precedence over queried parameters.
  int64_t from_device_id = -1;
  if (auto it = attrs.find(kFromDevice); it != attrs.end()) {
    from_device_id = it->second.cast<int64_t>();
    attrs.erase(it);
  }

  if (from_device_id >= 0) {
    auto device_params = QueryDevice(from_device_id, target.get());
    for (const auto& kv : device_params) {
      if (attrs.count(kv.first) == 0) {
        attrs[kv.first] = kv.second;
      }
    }
  }

  target->attrs = attrs;
  return target;
}

std::unordered_map<ffi::String, ffi::Any> TargetInternal::QueryDevice(int device_id,
                                                                      const TargetNode* target) {
  std::unordered_map<ffi::String, ffi::Any> output;

  Device device{static_cast<DLDeviceType>(target->GetTargetDeviceType()), device_id};

  auto api = runtime::DeviceAPI::Get(device, true);
  if (!api) {
    LOG(INFO) << "Requested reading the parameters for " << target->kind->name << " from device_id "
              << device_id << ", but support for this runtime wasn't enabled at compile-time.  "
              << "Using default target parameters.";
    return output;
  }

  ffi::Any ret;
  api->GetAttr(device, runtime::kExist, &ret);
  bool device_exists = ret.cast<bool>();
  if (!device_exists) {
    TVM_FFI_ICHECK(device_exists) << "Requested reading the parameters for " << target->kind->name
                                  << " from device_id " << device_id << ", but device_id "
                                  << device_id
                                  << " doesn't exist.  Using default target parameters.";
    return output;
  }

  for (const auto& e : target->kind->schema_.ListOptions()) {
    ffi::Any ret;
    api->GetTargetProperty(device, e.key, &ret);
    if (ret.type_index() != kTVMFFINone) {
      output[e.key] = ret;
    }
  }

  return output;
}

/**********  Registry  **********/

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def_packed("target.Target", TargetInternal::ConstructorDispatcher)
      .def("target.TargetEnterScope", TargetInternal::EnterScope)
      .def("target.TargetExitScope", TargetInternal::ExitScope)
      .def("target.TargetCurrent", Target::Current)
      .def("target.TargetExport", TargetInternal::ToConfig)
      .def("target.WithHost", TargetInternal::WithHost)
      .def("target.TargetGetDeviceType",
           [](const Target& target) { return target->GetTargetDeviceType(); })
      .def("target.TargetGetFeature",
           [](const Target& target, const ffi::String& feature_key) -> Any {
             ffi::String full_key = "feature." + std::string(feature_key);
             auto it = target->attrs.find(full_key);
             if (it != target->attrs.end()) {
               return (*it).second;
             }
             return Any();
           });
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TargetNode>([](const ObjectRef& obj, ReprPrinter* p) {
      p->stream << Downcast<Target>(obj)->str();
    });

}  // namespace tvm
