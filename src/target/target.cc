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
  static ffi::Map<ffi::String, ffi::Any> Export(Target target) { return target->Export(); }
  static const TargetKindNode::ValueTypeInfo& FindTypeInfo(const TargetKind& kind,
                                                           const std::string& key);
  static Any ParseType(const Any& obj, const TargetKindNode::ValueTypeInfo& info);
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

template <class T>
static T ObjTypeCheck(const Any& obj, const std::string& expected_type) {
  auto opt = obj.try_cast<T>();
  if (!opt.has_value()) {
    TVM_FFI_THROW(TypeError) << "Expects type \"" << expected_type << "\", but gets \""
                             << obj.GetTypeKey() << "\" for object: " << obj;
  }
  return opt.value();
}

static TargetKind GetTargetKind(const ffi::String& name) {
  ffi::Optional<TargetKind> kind = TargetKind::Get(name);
  if (!kind.defined()) {
    TVM_FFI_THROW(TypeError) << "Target kind \"" + name + "\" is not defined";
  }
  return kind.value();
}

const TargetKindNode::ValueTypeInfo& TargetInternal::FindTypeInfo(const TargetKind& kind,
                                                                  const std::string& key) {
  auto it = kind->key2vtype_.find(key);
  if (it == kind->key2vtype_.end()) {
    std::ostringstream os;
    os << ": Cannot recognize \'" << key << "\'. Candidates are: ";
    bool is_first = true;
    for (const auto& kv : kind->key2vtype_) {
      if (is_first) {
        is_first = false;
      } else {
        os << ", ";
      }
      os << kv.first;
    }
    TVM_FFI_THROW(TypeError) << os.str();
  }
  return it->second;
}

/**********  Parsing  **********/

Any TargetInternal::ParseType(const Any& obj, const TargetKindNode::ValueTypeInfo& info) {
  if (info.type_index == ffi::TypeIndex::kTVMFFIInt) {
    // Parsing integer
    return ObjTypeCheck<int64_t>(obj, "int64_t");
  } else if (info.type_index == ffi::TypeIndex::kTVMFFIBool) {
    // Parsing boolean
    return ObjTypeCheck<bool>(obj, "bool");
  } else if (info.type_index == ffi::TypeIndex::kTVMFFIStr) {
    // Parsing string
    return ObjTypeCheck<ffi::String>(obj, "String");
  } else if (info.type_index == Target::ContainerType::RuntimeTypeIndex()) {
    // Parsing target
    if (auto opt = obj.as<Target>()) {
      return opt.value();
    } else if (auto str = obj.try_cast<ffi::String>()) {
      return Target(TargetInternal::FromString(str.value()));
    } else if (const auto* ptr = obj.as<ffi::MapObj>()) {
      for (const auto& kv : *ptr) {
        if (!kv.first.as<ffi::String>()) {
          TVM_FFI_THROW(TypeError)
              << "Target object requires key of dict to be str, but get: " << kv.first.GetTypeKey();
        }
      }
      ffi::Map<ffi::String, ffi::Any> config = ffi::GetRef<ffi::Map<ffi::String, ffi::Any>>(ptr);
      return Target(TargetInternal::FromConfig({config.begin(), config.end()}));
    }
    TVM_FFI_THROW(TypeError) << "Expect type 'dict' or 'str' to construct Target, but get: " +
                                    obj.GetTypeKey();
  } else if (info.type_index == ffi::ArrayObj::RuntimeTypeIndex()) {
    // Parsing array
    const auto* array = ObjTypeCheck<const ffi::ArrayObj*>(obj, "Array");
    std::vector<Any> result;
    for (const Any& e : *array) {
      try {
        result.push_back(TargetInternal::ParseType(e, *info.key));
      } catch (const Error& e) {
        std::string index = '[' + std::to_string(result.size()) + ']';
        throw Error(e.kind(), index + e.message(), e.backtrace());
      }
    }
    return ffi::Array<Any>(result);
  } else if (info.type_index == ffi::MapObj::RuntimeTypeIndex()) {
    // Parsing map
    const auto* map = ObjTypeCheck<const ffi::MapObj*>(obj, "Map");
    std::unordered_map<Any, Any, ffi::AnyHash, ffi::AnyEqual> result;
    for (const auto& kv : *map) {
      Any key, val;
      try {
        key = TargetInternal::ParseType(kv.first, *info.key);
      } catch (const Error& e) {
        throw Error(e.kind(), e.message() + ", during parse key of map", e.backtrace());
      }
      try {
        val = TargetInternal::ParseType(kv.second, *info.val);
      } catch (const Error& e) {
        std::ostringstream os;
        os << ", during parseing value of map[\"" << key << "\"]";
        throw Error(e.kind(), e.message() + os.str(), e.backtrace());
      }
      result[key] = val;
    }
    return ffi::Map<Any, Any>(result);
  }
  if (info.type_index != obj.type_index()) {
    TVM_FFI_THROW(TypeError) << "Parsing type \"" << info.type_key
                             << "\" is not supported for the given object of type \""
                             << obj.GetTypeKey() << "\". The object is: " << obj;
  }
  return obj;
}

const std::string& TargetNode::str() const {
  if (str_repr_.empty()) {
    str_repr_ = std::string(ffi::json::Stringify(Export()));
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
    target = TargetInternal::FromConfig({config.begin(), config.end()});
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

std::vector<std::string> TargetNode::GetKeys() const {
  std::vector<std::string> result;
  for (auto& expr : keys) {
    result.push_back(expr);
  }
  return result;
}

std::unordered_set<std::string> TargetNode::GetLibs() const {
  ffi::Optional<ffi::Array<ffi::String>> libs = this->GetAttr<ffi::Array<ffi::String>>("libs");
  if (!libs.defined()) {
    return {};
  }
  std::unordered_set<std::string> result;
  for (const auto& item : libs.value()) {
    result.insert(item);
  }
  return result;
}

ffi::Map<ffi::String, ffi::Any> TargetNode::Export() const {
  ffi::Map<ffi::String, ffi::Any> result = {
      {"kind", this->kind->name},
      {"tag", this->tag},
      {"keys", this->keys},
  };
  if (this->host.defined()) {
    result.Set("host", this->GetHost().value_or(Target())->Export());
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

ffi::String TargetNode::ToDebugString() const {
  std::ostringstream os;
  os << "Target(";
  os << "id=" << std::hex << reinterpret_cast<size_t>(this);
  os << ", kind='" << kind->name << "'";
  if (!tag.empty()) {
    os << ", tag='" << tag << "'";
  }
  if (!keys.empty()) {
    os << ", keys={";
    bool first = true;
    for (const auto& key : keys) {
      if (!first) {
        os << ", ";
      }
      os << "'" << key << "'";
      first = false;
    }
    os << "}";
  }
  if (!attrs.empty()) {
    os << ", attrs={";
    bool first = true;
    for (const auto& pair : attrs) {
      if (!first) {
        os << ", ";
      }
      os << "'" << pair.first << "': " << pair.second;
      first = false;
    }
    os << "}";
  }
  if (host.defined()) {
    os << ", host=" << GetHost().value()->ToDebugString();
  }
  os << ")";
  return os.str();
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
  ICHECK(!entry->context_stack.empty());
  ICHECK(entry->context_stack.top().same_as(*this));
  entry->context_stack.pop();
}

Target Target::Current(bool allow_not_defined) {
  TVMTargetThreadLocalEntry* entry = TVMTargetThreadLocalStoreGet();
  if (entry->context_stack.size() > 0) {
    return entry->context_stack.top();
  }
  ICHECK(allow_not_defined)
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
      LOG(FATAL) << "TypeError: Cannot create target with type: " << args[0].GetTypeKey();
    }
    return;
  } else if (args.size() == 2) {
    if (args[0].as<Target>().has_value() && args[1].as<Target>().has_value()) {
      Target target = args[0].cast<Target>();
      Target host = args[1].cast<Target>();
      *rv = Target(target, host);
    } else {
      LOG(FATAL) << "ValueError: Invalid type of arguments. Expect 2 Target arguments.";
    }
    return;
  }
  LOG(FATAL) << "ValueError: Invalid number of arguments. Expect 1 or 2, but gets: " << args.size();
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
  return TargetInternal::FromConfig({{"kind", ffi::String(s)}});
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
  return TargetInternal::FromConfig({config.value().begin(), config.value().end()});
}

ObjectPtr<TargetNode> TargetInternal::FromConfig(ffi::Map<ffi::String, ffi::Any> config) {
  const ffi::String kKind = "kind";
  const ffi::String kTag = "tag";
  const ffi::String kKeys = "keys";
  const ffi::String kDeviceName = "device";
  const ffi::String kHost = "host";
  const ffi::String kFeatures = "features";
  ObjectPtr<TargetNode> target = ffi::make_object<TargetNode>();

  ICHECK(!config.count(kFeatures)) << "Target Features should be generated by Target parser";

  // parse 'kind'
  if (config.count(kKind)) {
    if (auto kind = config[kKind].try_cast<ffi::String>()) {
      target->kind = GetTargetKind(kind.value());
      ICHECK(!(target->kind->preprocessor != nullptr && target->kind->target_parser != nullptr))
          << "Cannot use both set_attrs_preprocessor and set_target_parser";

      // Run JSON Parser over JSON input
      if (target->kind->target_parser != nullptr) {
        VLOG(9) << "TargetInternal::FromConfig - Running target_parser";
        config = target->kind->target_parser(config);
        if (config.count(kFeatures)) {
          target->features = Downcast<ffi::Map<ffi::String, ffi::Any>>(config[kFeatures]);
          config.erase(kFeatures);
        }
      }

      config.erase(kKind);
    } else {
      TVM_FFI_THROW(TypeError) << "Expect type of field \"kind\" is String, but get type: "
                               << config[kKind].GetTypeKey();
    }
  } else {
    TVM_FFI_THROW(ValueError) << "Field \"kind\" is not found";
  }
  // parse "tag"
  if (config.count(kTag)) {
    if (auto tag = config[kTag].try_cast<ffi::String>()) {
      target->tag = tag.value();
      config.erase(kTag);
    } else {
      TVM_FFI_THROW(TypeError) << "Expect type of field \"tag\" is String, but get type: "
                               << config[kTag].GetTypeKey();
    }
  } else {
    target->tag = "";
  }
  // parse "keys"
  {
    std::vector<ffi::String> keys;
    bool has_user_keys = config.count(kKeys);
    if (has_user_keys) {
      // user provided keys
      if (const auto* cfg_keys = config[kKeys].as<ffi::ArrayObj>()) {
        for (const Any& e : *cfg_keys) {
          if (auto key = e.try_cast<ffi::String>()) {
            keys.push_back(key.value());
          } else {
            TVM_FFI_THROW(TypeError) << "Expect 'keys' to be an array of strings, but it "
                                     << "contains an element of type: " << e.GetTypeKey();
          }
        }
      } else {
        TVM_FFI_THROW(TypeError) << "Expect type of field \"keys\" is Array, but get type: "
                                 << config[kKeys].GetTypeKey();
      }
    }
    // add device name
    if (config.count(kDeviceName)) {
      if (auto device = config.at(kDeviceName).try_cast<ffi::String>()) {
        keys.push_back(device.value());
      }
    }
    if (!has_user_keys) {
      // add default keys
      for (const auto& key : target->kind->default_keys) {
        keys.push_back(key);
      }
    }
    // de-duplicate keys
    target->keys = DeduplicateKeys(keys);
    config.erase(kKeys);
  }
  // parse host
  if (config.count(kHost)) {
    target->host = ffi::Function(ConstructorDispatcher)(config[kHost]).cast<Target>();
    config.erase(kHost);
  } else {
    target->host = std::nullopt;
  }
  // parse attrs
  std::unordered_map<ffi::String, ffi::Any> attrs;
  for (const auto& cfg_kv : config) {
    const ffi::String& key = cfg_kv.first;
    const ffi::Any& value = cfg_kv.second;
    try {
      const TargetKindNode::ValueTypeInfo& info = TargetInternal::FindTypeInfo(target->kind, key);
      attrs[key] = TargetInternal::ParseType(value, info);
    } catch (const Error& e) {
      throw Error(e.kind(), std::string(e.message()) + ", during parsing target[\"" + key + "\"]",
                  e.backtrace());
    }
  }

  // If requested, query attributes from the device.  User-specified
  // parameters take precedence over queried parameters.
  if (attrs.count("from_device")) {
    int device_id = attrs.at("from_device").cast<int64_t>();
    attrs.erase("from_device");
    auto device_params = QueryDevice(device_id, target.get());

    for (const auto& kv : device_params) {
      if (attrs.count(kv.first) == 0) {
        attrs[kv.first] = kv.second;
      }
    }
  }

  // set default attribute values if they do not exist
  for (const auto& kv : target->kind->key2default_) {
    if (!attrs.count(kv.first)) {
      attrs[kv.first] = kv.second;
    }
  }
  // do extra pre-processing
  if (target->kind->preprocessor != nullptr) {
    target->attrs = target->kind->preprocessor(ffi::Map<ffi::String, ffi::Any>(attrs))
                        .cast<ffi::Map<ffi::String, ffi::Any>>();
  } else {
    target->attrs = attrs;
  }

  return target;
}  // namespace tvm

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
    ICHECK(device_exists) << "Requested reading the parameters for " << target->kind->name
                          << " from device_id " << device_id << ", but device_id " << device_id
                          << " doesn't exist.  Using default target parameters.";
    return output;
  }

  for (const auto& kv : target->kind->key2vtype_) {
    const ffi::String& key = kv.first;

    ffi::Any ret;
    api->GetTargetProperty(device, key, &ret);
    output[key] = ret;
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
      .def("target.TargetExport", TargetInternal::Export)
      .def("target.WithHost", TargetInternal::WithHost)
      .def("target.TargetGetDeviceType",
           [](const Target& target) { return target->GetTargetDeviceType(); })
      .def("target.TargetGetFeature",
           [](const Target& target, const ffi::String& feature_key) -> Any {
             if (auto opt_any = target->GetFeature<Any>(feature_key)) {
               return opt_any.value();
             } else {
               return Any();
             }
           });
}

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TargetNode>([](const ObjectRef& obj, ReprPrinter* p) {
      p->stream << Downcast<Target>(obj)->str();
    });

}  // namespace tvm
