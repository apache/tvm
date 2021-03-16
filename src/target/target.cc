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
#include <tvm/runtime/registry.h>
#include <tvm/target/tag.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>
#include <tvm/tir/expr.h>

#include <algorithm>
#include <stack>

#include "../runtime/object_internal.h"

namespace tvm {

TVM_REGISTER_NODE_TYPE(TargetNode);

class TargetInternal {
 public:
  static void EnterScope(Target target) { target.EnterWithScope(); }
  static void ExitScope(Target target) { target.ExitWithScope(); }
  static Map<String, ObjectRef> Export(Target target) { return target->Export(); }
  static const TargetKindNode::ValueTypeInfo& FindTypeInfo(const TargetKind& kind,
                                                           const std::string& key);
  static Optional<String> StringifyAttrsToRaw(const Map<String, ObjectRef>& attrs);
  static ObjectRef ParseType(const std::string& str, const TargetKindNode::ValueTypeInfo& info);
  static ObjectRef ParseType(const ObjectRef& obj, const TargetKindNode::ValueTypeInfo& info);
  static ObjectPtr<Object> FromString(const String& tag_or_config_or_target_str);
  static ObjectPtr<Object> FromConfigString(const String& config_str);
  static ObjectPtr<Object> FromRawString(const String& target_str);
  static ObjectPtr<Object> FromConfig(std::unordered_map<String, ObjectRef> config);
  static void ConstructorDispatcher(TVMArgs args, TVMRetValue* rv);
};

/**********  Helper functions  **********/

static std::vector<String> DeduplicateKeys(const std::vector<String>& keys) {
  std::vector<String> new_keys;
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

template <class TObj>
static const TObj* ObjTypeCheck(const ObjectRef& obj, const std::string& expected_type) {
  const TObj* ptr = obj.as<TObj>();
  if (ptr == nullptr) {
    std::ostringstream os;
    os << ": Expects type \"" << expected_type << "\", but gets \"" << obj->GetTypeKey()
       << "\" for object: " << obj;
    throw Error(os.str());
  }
  return ptr;
}

static TargetKind GetTargetKind(const String& name) {
  Optional<TargetKind> kind = TargetKind::Get(name);
  if (!kind.defined()) {
    throw Error(": Target kind \"" + name + "\" is not defined");
  }
  return kind.value();
}

static std::string RemovePrefixDashes(const std::string& s) {
  int n_dashes = 0;
  int len = s.length();
  for (; n_dashes < len && s[n_dashes] == '-'; ++n_dashes) {
  }
  if (n_dashes == 0) {
    throw Error(": Attribute keys should start with '-', not an attribute key: " + s);
  }
  if (n_dashes >= len) {
    throw Error(": Not an attribute key: " + s);
  }
  return s.substr(n_dashes);
}

static int FindFirstSubstr(const std::string& str, const std::string& substr) {
  size_t pos = str.find_first_of(substr);
  return pos == std::string::npos ? -1 : pos;
}

static Optional<String> JoinString(const std::vector<String>& array, char separator) {
  if (array.empty()) {
    return NullOpt;
  }
  std::ostringstream os;
  os << array[0];
  for (size_t i = 1; i < array.size(); ++i) {
    os << separator << array[i];
  }
  return String(os.str());
}

static int ParseKVPair(const std::string& s, const std::string& s_next, std::string* key,
                       std::string* value) {
  int pos;
  std::string& result_k = *key;
  std::string& result_v = *value;
  if ((pos = FindFirstSubstr(s, "=")) != -1) {
    // case 1. --key=value
    result_k = s.substr(0, pos);
    result_v = s.substr(pos + 1);
    if (result_k.empty() || result_v.empty()) {
      throw Error(": Empty attribute key or value in \"" + s + "\"");
    }
    return 1;
  } else if (!s_next.empty() && s_next[0] != '-') {
    // case 2. --key value
    result_k = s;
    result_v = s_next;
    return 2;
  }
  // case 3. --boolean-key
  result_k = s;
  result_v = "1";
  return 1;
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
    throw Error(os.str());
  }
  return it->second;
}

/**********  Parsing  **********/

ObjectRef TargetInternal::ParseType(const std::string& str,
                                    const TargetKindNode::ValueTypeInfo& info) {
  std::istringstream is(str);
  if (info.type_index == Integer::ContainerType::_GetOrAllocRuntimeTypeIndex()) {
    // Parsing integer
    int v;
    if (!(is >> v)) {
      throw Error(": Cannot parse into type \"Integer\" from string: " + str);
    }
    return Integer(v);
  } else if (info.type_index == String::ContainerType::_GetOrAllocRuntimeTypeIndex()) {
    // Parsing string
    std::string v;
    if (!(is >> v)) {
      throw Error(": Cannot parse into type \"String\" from string: " + str);
    }
    return String(v);
  } else if (info.type_index == Target::ContainerType::_GetOrAllocRuntimeTypeIndex()) {
    // Parsing target
    return Target(TargetInternal::FromString(str));
  } else if (info.type_index == ArrayNode::_GetOrAllocRuntimeTypeIndex()) {
    // Parsing array
    std::vector<ObjectRef> result;
    for (std::string substr; std::getline(is, substr, ',');) {
      try {
        ObjectRef parsed = TargetInternal::ParseType(substr, *info.key);
        result.push_back(parsed);
      } catch (const Error& e) {
        std::string index = "[" + std::to_string(result.size()) + "]";
        throw Error(index + e.what());
      }
    }
    return Array<ObjectRef>(result);
  }
  throw Error(": Unsupported type \"" + info.type_key + "\" for parsing from string: " + str);
}

ObjectRef TargetInternal::ParseType(const ObjectRef& obj,
                                    const TargetKindNode::ValueTypeInfo& info) {
  if (info.type_index == Integer::ContainerType::_GetOrAllocRuntimeTypeIndex()) {
    // Parsing integer
    return GetRef<Integer>(ObjTypeCheck<IntImmNode>(obj, "Integer"));
  } else if (info.type_index == String::ContainerType::_GetOrAllocRuntimeTypeIndex()) {
    // Parsing string
    return GetRef<String>(ObjTypeCheck<StringObj>(obj, "String"));
  } else if (info.type_index == Target::ContainerType::_GetOrAllocRuntimeTypeIndex()) {
    // Parsing target
    if (const auto* ptr = obj.as<TargetNode>()) {
      return GetRef<Target>(ptr);
    } else if (const auto* ptr = obj.as<StringObj>()) {
      return Target(TargetInternal::FromString(GetRef<String>(ptr)));
    } else if (const auto* ptr = obj.as<MapNode>()) {
      for (const auto& kv : *ptr) {
        if (!kv.first->IsInstance<StringObj>()) {
          throw Error(": Target object requires key of dict to be str, but get: " +
                      kv.first->GetTypeKey());
        }
      }
      Map<String, ObjectRef> config = GetRef<Map<String, ObjectRef>>(ptr);
      return Target(TargetInternal::FromConfig({config.begin(), config.end()}));
    }
    throw Error(": Expect type 'dict' or 'str' to construct Target, but get: " + obj->GetTypeKey());
  } else if (info.type_index == ArrayNode::_GetOrAllocRuntimeTypeIndex()) {
    // Parsing array
    const auto* array = ObjTypeCheck<ArrayNode>(obj, "Array");
    std::vector<ObjectRef> result;
    for (const ObjectRef& e : *array) {
      try {
        result.push_back(TargetInternal::ParseType(e, *info.key));
      } catch (const Error& e) {
        std::string index = '[' + std::to_string(result.size()) + ']';
        throw Error(index + e.what());
      }
    }
    return Array<ObjectRef>(result);
  } else if (info.type_index == MapNode::_GetOrAllocRuntimeTypeIndex()) {
    // Parsing map
    const auto* map = ObjTypeCheck<MapNode>(obj, "Map");
    std::unordered_map<ObjectRef, ObjectRef, ObjectHash, ObjectEqual> result;
    for (const auto& kv : *map) {
      ObjectRef key, val;
      try {
        key = TargetInternal::ParseType(kv.first, *info.key);
      } catch (const Error& e) {
        std::ostringstream os;
        os << "'s key \"" << key << "\"" << e.what();
        throw Error(os.str());
      }
      try {
        val = TargetInternal::ParseType(kv.second, *info.val);
      } catch (const Error& e) {
        std::ostringstream os;
        os << "[\"" << key << "\"]" << e.what();
        throw Error(os.str());
      }
      result[key] = val;
    }
    return Map<ObjectRef, ObjectRef>(result);
  }
  if (info.type_index != obj->type_index()) {
    std::ostringstream os;
    os << ": Parsing type \"" << info.type_key
       << "\" is not supported for the given object of type \"" << obj->GetTypeKey()
       << "\". The object is: " << obj;
    throw Error(os.str());
  }
  return obj;
}

/**********  Stringifying  **********/

static inline Optional<String> StringifyAtomicType(const ObjectRef& obj) {
  if (const auto* p = obj.as<IntImmNode>()) {
    return String(std::to_string(p->value));
  }
  if (const auto* p = obj.as<StringObj>()) {
    return GetRef<String>(p);
  }
  return NullOpt;
}

Optional<String> TargetInternal::StringifyAttrsToRaw(const Map<String, ObjectRef>& attrs) {
  std::ostringstream os;
  std::vector<String> keys;
  for (const auto& kv : attrs) {
    keys.push_back(kv.first);
  }
  std::sort(keys.begin(), keys.end());
  std::vector<String> result;
  for (const auto& key : keys) {
    const ObjectRef& obj = attrs[key];
    Optional<String> value = NullOpt;
    if (const auto* array = obj.as<ArrayNode>()) {
      std::vector<String> items;
      for (const ObjectRef& item : *array) {
        Optional<String> str = StringifyAtomicType(item);
        if (str.defined()) {
          items.push_back(str.value());
        } else {
          items.clear();
          break;
        }
      }
      value = JoinString(items, ',');
    } else {
      value = StringifyAtomicType(obj);
    }
    if (value.defined()) {
      result.push_back("-" + key + "=" + value.value());
    }
  }
  return JoinString(result, ' ');
}

const std::string& TargetNode::str() const {
  if (str_repr_.empty()) {
    std::ostringstream os;
    os << kind->name;
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
    if (Optional<String> attrs_str = TargetInternal::StringifyAttrsToRaw(attrs)) {
      os << ' ' << attrs_str.value();
    }
    str_repr_ = os.str();
  }
  return str_repr_;
}

/**********  Small member methods  **********/

Target::Target(const String& tag_or_config_or_target_str) {
  ObjectPtr<Object> target;
  try {
    target = TargetInternal::FromString(tag_or_config_or_target_str);
  } catch (const Error& e) {
    LOG(FATAL) << "ValueError" << e.what()
               << ". Target creation from string failed: " << tag_or_config_or_target_str;
  }
  data_ = std::move(target);
}

Target::Target(const Map<String, ObjectRef>& config) {
  ObjectPtr<Object> target;
  try {
    target = TargetInternal::FromConfig({config.begin(), config.end()});
  } catch (const Error& e) {
    LOG(FATAL) << "ValueError" << e.what()
               << ". Target creation from config dict failed: " << config;
  }
  data_ = std::move(target);
}

Target::Target(Target target, Target host) {
  ObjectPtr<TargetNode> n = make_object<TargetNode>(*target.get());
  CHECK(!n->host.defined())
      << "ValueError: Adding a host to a target whose host field has been defined";
  // add target host into host field
  n->host = std::move(host);
  data_ = std::move(n);
}

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

Map<String, ObjectRef> TargetNode::Export() const {
  Map<String, ObjectRef> result = {
      {"kind", this->kind->name},
      {"tag", this->tag},
      {"keys", this->keys},
  };
  for (const auto& kv : attrs) {
    result.Set(kv.first, kv.second);
  }
  return result;
}

/*! \brief Entry to hold the Target context stack. */
struct TVMTargetThreadLocalEntry {
  /*! \brief The current target context */
  std::stack<Target> context_stack;
};

/*! \brief Thread local store to hold the Target context stack. */
using TVMTargetThreadLocalStore = dmlc::ThreadLocalStore<TVMTargetThreadLocalEntry>;

void Target::EnterWithScope() {
  TVMTargetThreadLocalEntry* entry = TVMTargetThreadLocalStore::Get();
  entry->context_stack.push(*this);
}

void Target::ExitWithScope() {
  TVMTargetThreadLocalEntry* entry = TVMTargetThreadLocalStore::Get();
  ICHECK(!entry->context_stack.empty());
  ICHECK(entry->context_stack.top().same_as(*this));
  entry->context_stack.pop();
}

Target Target::Current(bool allow_not_defined) {
  TVMTargetThreadLocalEntry* entry = TVMTargetThreadLocalStore::Get();
  if (entry->context_stack.size() > 0) {
    return entry->context_stack.top();
  }
  ICHECK(allow_not_defined)
      << "Target context required. Please set it by constructing a TargetContext";

  return Target();
}

/**********  Creation  **********/

void TargetInternal::ConstructorDispatcher(TVMArgs args, TVMRetValue* rv) {
  if (args.num_args == 1) {
    const auto& arg = args[0];
    if (arg.IsObjectRef<Target>()) {
      *rv = Target(arg.AsObjectRef<Target>());
    } else if (String::CanConvertFrom(arg)) {
      *rv = Target(arg.operator String());
    } else if (arg.IsObjectRef<Map<String, ObjectRef>>()) {
      *rv = Target(arg.operator Map<String, ObjectRef>());
    } else if (arg.type_code() == kTVMObjectHandle) {
      ObjectRef obj = arg;
      LOG(FATAL) << "TypeError: Cannot create target with type: " << obj->GetTypeKey();
    } else {
      LOG(FATAL) << "TypeError: Cannot create target with type: "
                 << runtime::ArgTypeCode2Str(arg.type_code());
    }
    return;
  } else if (args.num_args == 2) {
    if (args[0].IsObjectRef<Target>() && args[1].IsObjectRef<Target>()) {
      Target target = args[0];
      Target host = args[1];
      *rv = Target(target, host);
    } else {
      LOG(FATAL) << "ValueError: Invalid type of arguments. Expect 2 Target arguments.";
    }
    return;
  }
  LOG(FATAL) << "ValueError: Invalid number of arguments. Expect 1 or 2, but gets: "
             << args.num_args;
}

ObjectPtr<Object> TargetInternal::FromString(const String& tag_or_config_or_target_str) {
  if (Optional<Target> target = TargetTag::Get(tag_or_config_or_target_str)) {
    Target value = target.value();
    return runtime::ObjectInternal::MoveObjectPtr(&value);
  }
  if (!tag_or_config_or_target_str.empty() && tag_or_config_or_target_str.data()[0] == '{') {
    return TargetInternal::FromConfigString(tag_or_config_or_target_str);
  }
  return TargetInternal::FromRawString(tag_or_config_or_target_str);
}

ObjectPtr<Object> TargetInternal::FromConfigString(const String& config_str) {
  const auto* loader = tvm::runtime::Registry::Get("target._load_config_dict");
  ICHECK(loader) << "AttributeError: \"target._load_config_dict\" is not registered. Please check "
                    "if the python module is properly loaded";
  Optional<Map<String, ObjectRef>> config = (*loader)(config_str);
  if (!config.defined()) {
    throw Error(": Cannot load config dict with python JSON loader");
  }
  return TargetInternal::FromConfig({config.value().begin(), config.value().end()});
}

ObjectPtr<Object> TargetInternal::FromRawString(const String& target_str) {
  // Split the string by empty spaces
  std::string name;
  std::vector<std::string> options;
  std::string str;
  for (std::istringstream is(target_str); is >> str;) {
    if (name.empty()) {
      name = str;
    } else {
      options.push_back(str);
    }
  }
  if (name.empty()) {
    throw Error(": Cannot parse empty target string");
  }
  // Create the target config
  std::unordered_map<String, ObjectRef> config = {{"kind", String(name)}};
  TargetKind kind = GetTargetKind(name);
  for (size_t iter = 0, end = options.size(); iter < end;) {
    std::string key, value;
    try {
      // Parse key-value pair
      std::string s_next = (iter + 1 < options.size()) ? options[iter + 1] : "";
      iter += ParseKVPair(RemovePrefixDashes(options[iter]), s_next, &key, &value);
    } catch (const Error& e) {
      throw Error(": Error when parsing target" + std::string(e.what()));
    }
    try {
      // check if `key` has been used
      if (config.count(key)) {
        throw Error(": The key \"" + key + "\" appears more than once");
      }
      config[key] = TargetInternal::ParseType(value, TargetInternal::FindTypeInfo(kind, key));
    } catch (const Error& e) {
      throw Error(": Error when parsing target[\"" + key + "\"]" + e.what());
    }
  }
  return TargetInternal::FromConfig(config);
}

ObjectPtr<Object> TargetInternal::FromConfig(std::unordered_map<String, ObjectRef> config) {
  const String kKind = "kind";
  const String kTag = "tag";
  const String kKeys = "keys";
  const String kDeviceName = "device";
  const String kHost = "host";
  ObjectPtr<TargetNode> target = make_object<TargetNode>();
  // parse 'kind'
  if (config.count(kKind)) {
    if (const auto* kind = config[kKind].as<StringObj>()) {
      target->kind = GetTargetKind(GetRef<String>(kind));
      config.erase(kKind);
    } else {
      throw Error(": Expect type of field \"kind\" is String, but get type: " +
                  config[kKind]->GetTypeKey());
    }
  } else {
    throw Error(": Field \"kind\" is not found");
  }
  // parse "tag"
  if (config.count(kTag)) {
    if (const auto* tag = config[kTag].as<StringObj>()) {
      target->tag = GetRef<String>(tag);
      config.erase(kTag);
    } else {
      throw Error(": Expect type of field \"tag\" is String, but get type: " +
                  config[kTag]->GetTypeKey());
    }
  } else {
    target->tag = "";
  }
  // parse "keys"
  {
    std::vector<String> keys;
    if (config.count(kKeys)) {
      // user provided keys
      if (const auto* cfg_keys = config[kKeys].as<ArrayNode>()) {
        for (const ObjectRef& e : *cfg_keys) {
          if (const auto* key = e.as<StringObj>()) {
            keys.push_back(GetRef<String>(key));
          } else {
            throw Error(
                ": Expect 'keys' to be an array of strings, but it "
                "contains an element of type: " +
                e->GetTypeKey());
          }
        }
      } else {
        throw Error(": Expect type of field \"keys\" is Array, but get type: " +
                    config[kKeys]->GetTypeKey());
      }
    }
    // add device name
    if (config.count(kDeviceName)) {
      if (const auto* device = config.at(kDeviceName).as<StringObj>()) {
        keys.push_back(GetRef<String>(device));
      }
    }
    // add default keys
    for (const auto& key : target->kind->default_keys) {
      keys.push_back(key);
    }
    // de-duplicate keys
    target->keys = DeduplicateKeys(keys);
    config.erase(kKeys);
  }
  // parse attrs
  std::unordered_map<String, ObjectRef> attrs;
  for (const auto& cfg_kv : config) {
    const String& key = cfg_kv.first;
    const ObjectRef& value = cfg_kv.second;
    try {
      const TargetKindNode::ValueTypeInfo& info = TargetInternal::FindTypeInfo(target->kind, key);
      attrs[key] = TargetInternal::ParseType(value, info);
    } catch (const Error& e) {
      throw Error(": Error when parsing target[\"" + key + "\"]" + e.what());
    }
  }
  // parse host
  if (config.count(kHost)) {
    target->host = PackedFunc(ConstructorDispatcher)(config[kHost]).AsObjectRef<Target>();
    config.erase(kHost);
  } else {
    target->host = NullOpt;
  }
  // set default attribute values if they do not exist
  for (const auto& kv : target->kind->key2default_) {
    if (!attrs.count(kv.first)) {
      attrs[kv.first] = kv.second;
    }
  }
  // do extra pre-processing
  if (target->kind->preprocessor != nullptr) {
    target->attrs = target->kind->preprocessor(Map<String, ObjectRef>(attrs));
  } else {
    target->attrs = attrs;
  }
  return target;
}

/**********  Registry  **********/

TVM_REGISTER_GLOBAL("target.Target").set_body(TargetInternal::ConstructorDispatcher);
TVM_REGISTER_GLOBAL("target.TargetEnterScope").set_body_typed(TargetInternal::EnterScope);
TVM_REGISTER_GLOBAL("target.TargetExitScope").set_body_typed(TargetInternal::ExitScope);
TVM_REGISTER_GLOBAL("target.TargetCurrent").set_body_typed(Target::Current);
TVM_REGISTER_GLOBAL("target.TargetExport").set_body_typed(TargetInternal::Export);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TargetNode>([](const ObjectRef& obj, ReprPrinter* p) {
      p->stream << Downcast<Target>(obj)->str();
    });

}  // namespace tvm
