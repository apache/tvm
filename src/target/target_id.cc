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
 * \file src/target/target_id.cc
 * \brief Target id registry
 */
#include <tvm/target/target_id.h>

#include <algorithm>

#include "../node/attr_registry.h"
#include "../runtime/object_internal.h"

namespace tvm {

TVM_REGISTER_NODE_TYPE(TargetIdNode);

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TargetIdNode>([](const ObjectRef& node, ReprPrinter* p) {
      auto* op = static_cast<const TargetIdNode*>(node.get());
      p->stream << op->name;
    });

using TargetIdRegistry = AttrRegistry<TargetIdRegEntry, TargetId>;

TargetIdRegEntry& TargetIdRegEntry::RegisterOrGet(const String& target_id_name) {
  return TargetIdRegistry::Global()->RegisterOrGet(target_id_name);
}

void TargetIdRegEntry::UpdateAttr(const String& key, TVMRetValue value, int plevel) {
  TargetIdRegistry::Global()->UpdateAttr(key, id_, value, plevel);
}

const AttrRegistryMapContainerMap<TargetId>& TargetId::GetAttrMapContainer(
    const String& attr_name) {
  return TargetIdRegistry::Global()->GetAttrMap(attr_name);
}

const TargetId& TargetId::Get(const String& target_id_name) {
  const TargetIdRegEntry* reg = TargetIdRegistry::Global()->Get(target_id_name);
  CHECK(reg != nullptr) << "ValueError: TargetId \"" << target_id_name << "\" is not registered";
  return reg->id_;
}

void TargetIdNode::VerifyTypeInfo(const ObjectRef& obj,
                                  const TargetIdNode::ValueTypeInfo& info) const {
  CHECK(obj.defined()) << "Object is None";
  if (!runtime::ObjectInternal::DerivedFrom(obj.get(), info.type_index)) {
    LOG(FATAL) << "AttributeError: expect type \"" << info.type_key << "\" but get "
               << obj->GetTypeKey();
    throw;
  }
  if (info.type_index == ArrayNode::_type_index) {
    int i = 0;
    for (const auto& e : *obj.as<ArrayNode>()) {
      try {
        VerifyTypeInfo(e, *info.key);
      } catch (const tvm::Error& e) {
        LOG(FATAL) << "The i-th element of array failed type checking, where i = " << i
                   << ", and the error is:\n"
                   << e.what();
        throw;
      }
      ++i;
    }
  } else if (info.type_index == MapNode::_type_index) {
    for (const auto& kv : *obj.as<MapNode>()) {
      try {
        VerifyTypeInfo(kv.first, *info.key);
      } catch (const tvm::Error& e) {
        LOG(FATAL) << "The key of map failed type checking, where key = \"" << kv.first
                   << "\", value = \"" << kv.second << "\", and the error is:\n"
                   << e.what();
        throw;
      }
      try {
        VerifyTypeInfo(kv.second, *info.val);
      } catch (const tvm::Error& e) {
        LOG(FATAL) << "The value of map failed type checking, where key = \"" << kv.first
                   << "\", value = \"" << kv.second << "\", and the error is:\n"
                   << e.what();
        throw;
      }
    }
  }
}

void TargetIdNode::ValidateSchema(const Map<String, ObjectRef>& config) const {
  const String kTargetId = "id";
  for (const auto& kv : config) {
    const String& name = kv.first;
    const ObjectRef& obj = kv.second;
    if (name == kTargetId) {
      CHECK(obj->IsInstance<StringObj>())
          << "AttributeError: \"id\" is not a string, but its type is \"" << obj->GetTypeKey()
          << "\"";
      CHECK(Downcast<String>(obj) == this->name)
          << "AttributeError: \"id\" = \"" << obj << "\" is inconsistent with TargetId \""
          << this->name << "\"";
      continue;
    }
    auto it = key2vtype_.find(name);
    if (it == key2vtype_.end()) {
      std::ostringstream os;
      os << "AttributeError: Invalid config option, cannot recognize \"" << name
         << "\". Candidates are:";
      for (const auto& kv : key2vtype_) {
        os << "\n  " << kv.first;
      }
      LOG(FATAL) << os.str();
      throw;
    }
    const auto& info = it->second;
    try {
      VerifyTypeInfo(obj, info);
    } catch (const tvm::Error& e) {
      LOG(FATAL) << "AttributeError: Schema validation failed for TargetId \"" << this->name
                 << "\", details:\n"
                 << e.what() << "\n"
                 << "The config is:\n"
                 << config;
      throw;
    }
  }
}

inline String GetId(const Map<String, ObjectRef>& target, const char* name) {
  const String kTargetId = "id";
  CHECK(target.count(kTargetId)) << "AttributeError: \"id\" does not exist in \"" << name << "\"\n"
                                 << name << " = " << target;
  const ObjectRef& obj = target[kTargetId];
  CHECK(obj->IsInstance<StringObj>()) << "AttributeError: \"id\" is not a string in \"" << name
                                      << "\", but its type is \"" << obj->GetTypeKey() << "\"\n"
                                      << name << " = \"" << target << '"';
  return Downcast<String>(obj);
}

void TargetValidateSchema(const Map<String, ObjectRef>& config) {
  try {
    const String kTargetHost = "target_host";
    Map<String, ObjectRef> target = config;
    Map<String, ObjectRef> target_host;
    String target_id = GetId(target, "target");
    String target_host_id;
    if (config.count(kTargetHost)) {
      target.erase(kTargetHost);
      target_host = Downcast<Map<String, ObjectRef>>(config[kTargetHost]);
      target_host_id = GetId(target_host, "target_host");
    }
    TargetId::Get(target_id)->ValidateSchema(target);
    if (!target_host.empty()) {
      TargetId::Get(target_host_id)->ValidateSchema(target_host);
    }
  } catch (const tvm::Error& e) {
    LOG(FATAL) << "AttributeError: schedule validation fails:\n"
               << e.what() << "\nThe configuration is:\n"
               << config;
  }
}

static inline size_t CountNumPrefixDashes(const std::string& s) {
  size_t i = 0;
  for (; i < s.length() && s[i] == '-'; ++i) {
  }
  return i;
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

static inline ObjectRef ParseScalar(uint32_t type_index, const std::string& str) {
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

static inline Optional<String> StringifyScalar(const ObjectRef& obj) {
  if (const auto* p = obj.as<IntImmNode>()) {
    return String(std::to_string(p->value));
  }
  if (const auto* p = obj.as<StringObj>()) {
    return GetRef<String>(p);
  }
  return NullOpt;
}

static inline Optional<String> Join(const std::vector<String>& array, char separator) {
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

Map<String, ObjectRef> TargetIdNode::ParseAttrsFromRaw(
    const std::vector<std::string>& options) const {
  std::unordered_map<String, ObjectRef> attrs;
  for (size_t iter = 0, end = options.size(); iter < end;) {
    std::string s = options[iter++];
    // remove the prefix dashes
    size_t n_dashes = CountNumPrefixDashes(s);
    CHECK(0 < n_dashes && n_dashes < s.size())
        << "ValueError: Not an attribute key \"" << s << "\"";
    s = s.substr(n_dashes);
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
    auto it = key2vtype_.find(name);
    if (it == key2vtype_.end()) {
      std::ostringstream os;
      os << "AttributeError: Invalid config option, cannot recognize \'" << name
         << "\'. Candidates are:";
      for (const auto& kv : key2vtype_) {
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
      parsed_obj = ParseScalar(info.type_index, obj);
    } else {
      Array<ObjectRef> array;
      std::string item;
      bool failed = false;
      uint32_t type_index = info.key->type_index;
      for (std::istringstream is(obj); std::getline(is, item, ',');) {
        ObjectRef parsed_obj = ParseScalar(type_index, item);
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
  for (const auto& kv : key2default_) {
    if (!attrs.count(kv.first)) {
      attrs[kv.first] = kv.second;
    }
  }
  return attrs;
}

Optional<String> TargetIdNode::StringifyAttrsToRaw(const Map<String, ObjectRef>& attrs) const {
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
        Optional<String> str = StringifyScalar(item);
        if (str.defined()) {
          items.push_back(str.value());
        } else {
          items.clear();
          break;
        }
      }
      value = Join(items, ',');
    } else {
      value = StringifyScalar(obj);
    }
    if (value.defined()) {
      result.push_back("-" + key + "=" + value.value());
    }
  }
  return Join(result, ' ');
}

// TODO(@junrushao1994): remove some redundant attributes

TVM_REGISTER_TARGET_ID("llvm")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<String>("mcpu")
    .add_attr_option<Array<String>>("mattr")
    .add_attr_option<String>("mtriple")
    .set_default_keys({"cpu"})
    .set_device_type(kDLCPU);

TVM_REGISTER_TARGET_ID("c")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .set_default_keys({"cpu"})
    .set_device_type(kDLCPU);

TVM_REGISTER_TARGET_ID("micro_dev")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .set_default_keys({"micro_dev"})
    .set_device_type(kDLMicroDev);

TVM_REGISTER_TARGET_ID("cuda")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(1024))
    .add_attr_option<Integer>("thread_warp_size", Integer(32))
    .add_attr_option<String>("mcpu")
    .set_default_keys({"cuda", "gpu"})
    .set_device_type(kDLGPU);

TVM_REGISTER_TARGET_ID("nvptx")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(1024))
    .add_attr_option<Integer>("thread_warp_size", Integer(32))
    .add_attr_option<String>("mcpu")
    .set_default_keys({"cuda", "gpu"})
    .set_device_type(kDLGPU);

TVM_REGISTER_TARGET_ID("rocm")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(256))
    .add_attr_option<Integer>("thread_warp_size", Integer(64))
    .set_default_keys({"rocm", "gpu"})
    .set_device_type(kDLROCM);

TVM_REGISTER_TARGET_ID("opencl")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(256))
    .add_attr_option<Integer>("thread_warp_size")
    .set_default_keys({"opencl", "gpu"})
    .set_device_type(kDLOpenCL);

TVM_REGISTER_TARGET_ID("metal")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(256))
    .set_default_keys({"metal", "gpu"})
    .set_device_type(kDLMetal);

TVM_REGISTER_TARGET_ID("vulkan")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(256))
    .set_default_keys({"vulkan", "gpu"})
    .set_device_type(kDLVulkan);

TVM_REGISTER_TARGET_ID("webgpu")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .add_attr_option<Integer>("max_num_threads", Integer(256))
    .set_default_keys({"webgpu", "gpu"})
    .set_device_type(kDLWebGPU);

TVM_REGISTER_TARGET_ID("sdaccel")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .set_default_keys({"sdaccel", "hls"})
    .set_device_type(kDLOpenCL);

TVM_REGISTER_TARGET_ID("aocl")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .set_default_keys({"aocl", "hls"})
    .set_device_type(kDLAOCL);

TVM_REGISTER_TARGET_ID("aocl_sw_emu")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .set_default_keys({"aocl", "hls"})
    .set_device_type(kDLAOCL);

TVM_REGISTER_TARGET_ID("hexagon")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .set_default_keys({"hexagon"})
    .set_device_type(kDLHexagon);

TVM_REGISTER_TARGET_ID("stackvm")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .set_device_type(kDLCPU);

TVM_REGISTER_TARGET_ID("ext_dev")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .set_device_type(kDLExtDev);

TVM_REGISTER_TARGET_ID("hybrid")
    .add_attr_option<Array<String>>("keys")
    .add_attr_option<Array<String>>("libs")
    .add_attr_option<String>("device")
    .add_attr_option<String>("model")
    .add_attr_option<Bool>("system-lib")
    .set_device_type(kDLCPU);

}  // namespace tvm
