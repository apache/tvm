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
#include <tvm/target/target_kind.h>
#include <tvm/tir/expr.h>

#include <algorithm>
#include <stack>

#include "../runtime/object_internal.h"

namespace tvm {

using runtime::PackedFunc;
using runtime::TVMArgs;
using runtime::TVMRetValue;

TVM_REGISTER_NODE_TYPE(TargetNode);

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

static inline std::string RemovePrefixDashes(const std::string& s) {
  size_t n_dashes = 0;
  for (; n_dashes < s.length() && s[n_dashes] == '-'; ++n_dashes) {
  }
  CHECK(0 < n_dashes && n_dashes < s.size()) << "ValueError: Not an attribute key \"" << s << "\"";
  return s.substr(n_dashes);
}

static inline int FindUniqueSubstr(const std::string& str, const std::string& substr) {
  size_t pos = str.find_first_of(substr);
  if (pos == std::string::npos) {
    return -1;
  }
  size_t next_pos = pos + substr.size();
  CHECK(next_pos >= str.size() || str.find_first_of(substr, next_pos) == std::string::npos)
      << "ValueError: At most one \"" << substr << "\" is allowed in "
      << "the the given string \"" << str << "\"";
  return pos;
}

static inline ObjectRef ParseAtomicType(uint32_t type_index, const std::string& str) {
  std::istringstream is(str);
  if (type_index == Integer::ContainerType::_GetOrAllocRuntimeTypeIndex()) {
    int v;
    is >> v;
    return is.fail() ? ObjectRef(nullptr) : Integer(v);
  } else if (type_index == String::ContainerType::_GetOrAllocRuntimeTypeIndex()) {
    std::string v;
    is >> v;
    return is.fail() ? ObjectRef(nullptr) : String(v);
  }
  return ObjectRef(nullptr);
}

Map<String, ObjectRef> TargetNode::ParseAttrsFromRaw(
    const std::vector<std::string>& options) const {
  std::unordered_map<String, ObjectRef> attrs;
  for (size_t iter = 0, end = options.size(); iter < end;) {
    // remove the prefix dashes
    std::string s = RemovePrefixDashes(options[iter++]);
    // parse name-obj pair
    std::string name;
    std::string obj;
    int pos;
    if ((pos = FindUniqueSubstr(s, "=")) != -1) {
      // case 1. --key=value
      name = s.substr(0, pos);
      obj = s.substr(pos + 1);
      CHECK(!name.empty()) << "ValueError: Empty attribute key in \"" << options[iter - 1] << "\"";
      CHECK(!obj.empty()) << "ValueError: Empty attribute in \"" << options[iter - 1] << "\"";
    } else if (iter < end && options[iter][0] != '-') {
      // case 2. --key value
      name = s;
      obj = options[iter++];
    } else {
      // case 3. --boolean-key
      name = s;
      obj = "1";
    }
    // check if `name` is invalid
    auto it = this->kind->key2vtype_.find(name);
    if (it == this->kind->key2vtype_.end()) {
      std::ostringstream os;
      os << "AttributeError: Invalid config option, cannot recognize \'" << name
         << "\'. Candidates are:";
      for (const auto& kv : this->kind->key2vtype_) {
        os << "\n  " << kv.first;
      }
      LOG(FATAL) << os.str();
    }
    // check if `name` has been set once
    CHECK(!attrs.count(name)) << "AttributeError: key \"" << name
                              << "\" appears more than once in the target string";
    // then `name` is valid, let's parse them
    // only several types are supported when parsing raw string
    const auto& info = it->second;
    ObjectRef parsed_obj(nullptr);
    if (info.type_index != ArrayNode::_type_index) {
      parsed_obj = ParseAtomicType(info.type_index, obj);
    } else {
      Array<ObjectRef> array;
      std::string item;
      bool failed = false;
      uint32_t type_index = info.key->type_index;
      for (std::istringstream is(obj); std::getline(is, item, ',');) {
        ObjectRef parsed_obj = ParseAtomicType(type_index, item);
        if (parsed_obj.defined()) {
          array.push_back(parsed_obj);
        } else {
          failed = true;
          break;
        }
      }
      if (!failed) {
        parsed_obj = std::move(array);
      }
    }
    if (!parsed_obj.defined()) {
      LOG(FATAL) << "ValueError: Cannot parse type \"" << info.type_key << "\""
                 << ", where attribute key is \"" << name << "\""
                 << ", and attribute is \"" << obj << "\"";
    }
    attrs[name] = std::move(parsed_obj);
  }
  // set default attribute values if they do not exist
  for (const auto& kv : this->kind->key2default_) {
    if (!attrs.count(kv.first)) {
      attrs[kv.first] = kv.second;
    }
  }
  return attrs;
}

static inline Optional<String> StringifyAtomicType(const ObjectRef& obj) {
  if (const auto* p = obj.as<IntImmNode>()) {
    return String(std::to_string(p->value));
  }
  if (const auto* p = obj.as<StringObj>()) {
    return GetRef<String>(p);
  }
  return NullOpt;
}

static inline Optional<String> JoinString(const std::vector<String>& array, char separator) {
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

Optional<String> TargetNode::StringifyAttrsToRaw(const Map<String, ObjectRef>& attrs) const {
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

Target Target::CreateTarget(const std::string& name, const std::vector<std::string>& options) {
  TargetKind kind = TargetKind::Get(name);
  ObjectPtr<TargetNode> target = make_object<TargetNode>();
  target->kind = kind;
  // tag is always empty
  target->tag = "";
  // parse attrs
  target->attrs = target->ParseAttrsFromRaw(options);
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
    for (const auto& key : target->kind->default_keys) {
      keys.push_back(key);
    }
    // de-duplicate keys
    target->keys = DeduplicateKeys(keys);
  }
  return Target(target);
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
    if (Optional<String> attrs_str = this->StringifyAttrsToRaw(attrs)) {
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

ObjectRef TargetNode::ParseAttr(const ObjectRef& obj,
                                const TargetKindNode::ValueTypeInfo& info) const {
  if (info.type_index == Integer::ContainerType::_GetOrAllocRuntimeTypeIndex()) {
    const auto* v = obj.as<IntImmNode>();
    CHECK(v != nullptr) << "Expect type 'int', but get: " << obj->GetTypeKey();
    return GetRef<Integer>(v);
  }
  if (info.type_index == String::ContainerType::_GetOrAllocRuntimeTypeIndex()) {
    const auto* v = obj.as<StringObj>();
    CHECK(v != nullptr) << "Expect type 'str', but get: " << obj->GetTypeKey();
    return GetRef<String>(v);
  }
  if (info.type_index == Target::ContainerType::_GetOrAllocRuntimeTypeIndex()) {
    CHECK(obj->IsInstance<MapNode>())
        << "Expect type 'dict' to construct Target, but get: " << obj->GetTypeKey();
    return Target::FromConfig(Downcast<Map<String, ObjectRef>>(obj));
  }
  if (info.type_index == ArrayNode::_GetOrAllocRuntimeTypeIndex()) {
    CHECK(obj->IsInstance<ArrayNode>()) << "Expect type 'list', but get: " << obj->GetTypeKey();
    Array<ObjectRef> array = Downcast<Array<ObjectRef>>(obj);
    std::vector<ObjectRef> result;
    int i = 0;
    for (const ObjectRef& e : array) {
      ++i;
      try {
        result.push_back(TargetNode::ParseAttr(e, *info.key));
      } catch (const dmlc::Error& e) {
        LOG(FATAL) << "Error occurred when parsing element " << i << " of the array: " << array
                   << ". Details:\n"
                   << e.what();
      }
    }
    return Array<ObjectRef>(result);
  }
  if (info.type_index == MapNode::_GetOrAllocRuntimeTypeIndex()) {
    CHECK(obj->IsInstance<MapNode>()) << "Expect type 'dict', but get: " << obj->GetTypeKey();
    std::unordered_map<ObjectRef, ObjectRef, ObjectHash, ObjectEqual> result;
    for (const auto& kv : Downcast<Map<ObjectRef, ObjectRef>>(obj)) {
      ObjectRef key, val;
      try {
        key = TargetNode::ParseAttr(kv.first, *info.key);
      } catch (const tvm::Error& e) {
        LOG(FATAL) << "Error occurred when parsing a key of the dict: " << kv.first
                   << ". Details:\n"
                   << e.what();
      }
      try {
        val = TargetNode::ParseAttr(kv.second, *info.val);
      } catch (const tvm::Error& e) {
        LOG(FATAL) << "Error occurred when parsing a value of the dict: " << kv.second
                   << ". Details:\n"
                   << e.what();
      }
      result[key] = val;
    }
    return Map<ObjectRef, ObjectRef>(result);
  }
  LOG(FATAL) << "Unsupported type registered: \"" << info.type_key
             << "\", and the type given is: " << obj->GetTypeKey();
  throw;
}

Target Target::FromConfig(const Map<String, ObjectRef>& config_dict) {
  const String kKind = "kind";
  const String kTag = "tag";
  const String kKeys = "keys";
  const String kDeviceName = "device";
  std::unordered_map<std::string, ObjectRef> config(config_dict.begin(), config_dict.end());
  ObjectPtr<TargetNode> target = make_object<TargetNode>();
  // parse 'kind'
  if (config.count(kKind)) {
    const auto* kind = config[kKind].as<StringObj>();
    CHECK(kind != nullptr) << "AttributeError: Expect type of field 'kind' is string, but get: "
                           << config[kKind]->GetTypeKey();
    target->kind = TargetKind::Get(GetRef<String>(kind));
    config.erase(kKind);
  } else {
    LOG(FATAL) << "AttributeError: Field 'kind' is not found";
  }
  // parse "tag"
  if (config.count(kTag)) {
    const auto* tag = config[kTag].as<StringObj>();
    CHECK(tag != nullptr) << "AttributeError: Expect type of field 'tag' is string, but get: "
                          << config[kTag]->GetTypeKey();
    target->tag = GetRef<String>(tag);
    config.erase(kTag);
  } else {
    target->tag = "";
  }
  // parse "keys"
  if (config.count(kKeys)) {
    std::vector<String> keys;
    // user provided keys
    const auto* cfg_keys = config[kKeys].as<ArrayNode>();
    CHECK(cfg_keys != nullptr)
        << "AttributeError: Expect type of field 'keys' is an Array, but get: "
        << config[kKeys]->GetTypeKey();
    for (const ObjectRef& e : *cfg_keys) {
      const auto* key = e.as<StringObj>();
      CHECK(key != nullptr) << "AttributeError: Expect 'keys' to be an array of strings, but it "
                               "contains an element of type: "
                            << e->GetTypeKey();
      keys.push_back(GetRef<String>(key));
    }
    // add device name
    if (config_dict.count(kDeviceName)) {
      if (const auto* device = config_dict.at(kDeviceName).as<StringObj>()) {
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
  } else {
    target->keys = {};
  }
  // parse attrs
  std::unordered_map<String, ObjectRef> attrs;
  const auto& key2vtype = target->kind->key2vtype_;
  for (const auto& cfg_kv : config) {
    const String& name = cfg_kv.first;
    const ObjectRef& obj = cfg_kv.second;
    if (!key2vtype.count(name)) {
      std::ostringstream os;
      os << "AttributeError: Unrecognized config option: \"" << name << "\". Candidates are:";
      for (const auto& kv : key2vtype) {
        os << " " << kv.first;
      }
      LOG(FATAL) << os.str();
    }
    ObjectRef val;
    try {
      val = target->ParseAttr(obj, key2vtype.at(name));
    } catch (const dmlc::Error& e) {
      LOG(FATAL) << "AttributeError: Error occurred in parsing the config key \"" << name
                 << "\". Details:\n"
                 << e.what();
    }
    attrs[name] = val;
  }
  // set default attribute values if they do not exist
  for (const auto& kv : target->kind->key2default_) {
    if (!attrs.count(kv.first)) {
      attrs[kv.first] = kv.second;
    }
  }
  target->attrs = attrs;
  return Target(target);
}

/*! \brief Entry to hold the Target context stack. */
struct TVMTargetThreadLocalEntry {
  /*! \brief The current target context */
  std::stack<tvm::Target> context_stack;
};

/*! \brief Thread local store to hold the Target context stack. */
using TVMTargetThreadLocalStore = dmlc::ThreadLocalStore<TVMTargetThreadLocalEntry>;

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

class Target::Internal {
 public:
  static void EnterScope(Target target) { target.EnterWithScope(); }
  static void ExitScope(Target target) { target.ExitWithScope(); }
};

TVM_REGISTER_GLOBAL("target.TargetCreate").set_body([](TVMArgs args, TVMRetValue* ret) {
  std::string name = args[0];
  std::vector<std::string> options;
  for (int i = 1; i < args.num_args; ++i) {
    std::string arg = args[i];
    options.push_back(arg);
  }

  *ret = Target::CreateTarget(name, options);
});

TVM_REGISTER_GLOBAL("target.EnterTargetScope").set_body_typed(Target::Internal::EnterScope);

TVM_REGISTER_GLOBAL("target.ExitTargetScope").set_body_typed(Target::Internal::ExitScope);

TVM_REGISTER_GLOBAL("target.GetCurrentTarget").set_body_typed(Target::Current);

TVM_REGISTER_GLOBAL("target.TargetFromString").set_body_typed(Target::Create);

TVM_REGISTER_GLOBAL("target.TargetFromConfig").set_body_typed(Target::FromConfig);

TVM_REGISTER_GLOBAL("target.TargetExport")
    .set_body_typed([](Target target) -> Map<String, ObjectRef> { return target->Export(); });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TargetNode>([](const ObjectRef& node, ReprPrinter* p) {
      const auto* target = node.as<TargetNode>();
      CHECK(target);
      p->stream << target->str();
    });

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
