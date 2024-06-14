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
#include <tvm/ir/transform.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/registry.h>
#include <tvm/target/tag.h>
#include <tvm/target/target.h>
#include <tvm/target/target_kind.h>
#include <tvm/tir/expr.h>

#include <algorithm>
#include <cctype>
#include <ios>
#include <sstream>
#include <stack>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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
  static ObjectPtr<Object> FromConfig(Map<String, ObjectRef> config);
  static void ConstructorDispatcher(TVMArgs args, TVMRetValue* rv);
  static Target WithHost(const Target& target, const Target& target_host) {
    ObjectPtr<TargetNode> n = make_object<TargetNode>(*target.get());
    n->host = target_host;
    return (Target)n;
  }

 private:
  static std::unordered_map<String, ObjectRef> QueryDevice(int device_id, const TargetNode* target);
  static bool IsQuoted(const std::string& str);
  static std::string Quote(const std::string& str);
  static std::string JoinString(const std::vector<std::string>& array, char separator);
  static std::vector<std::string> SplitString(const std::string& str, char separator);
  static std::string Interpret(const std::string& str);
  static std::string Uninterpret(const std::string& str);
  static std::string StringifyAtomicType(const ObjectRef& obj);
  static std::string StringifyArray(const ArrayNode& array);

  static constexpr char quote = '\'';
  static constexpr char escape = '\\';
};

/**********  Helper functions  **********/
Target Target::WithHost(const Target& target, const Target& host) {
  return TargetInternal::WithHost(target, host);
}

void CheckAndUpdateHostConsistency(Target* target, Target* host) {
  *target = Target(*target, *host);
  *host = (*target)->GetHost().value_or(Target());
}

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

bool TargetInternal::IsQuoted(const std::string& str) {
  std::string::size_type start = 0, end = str.size();
  if (end < 2 || str[start] != quote || str[end - 1] != quote) {
    return false;
  }
  bool escaping = false;
  for (auto i = start + 1, e = end - 1; i < e; ++i) {
    if (escaping) {
      escaping = false;
    } else if (str[i] == escape) {
      escaping = true;
    } else if (str[i] == quote) {
      return false;
    }
  }
  // If the reduced string ends with \, then the terminating quote is escaped.
  return !escaping;
}

std::string TargetInternal::Quote(const std::string& str) {
  std::string result(1, quote);
  result.append(str);
  result.push_back(quote);
  return result;
}

std::string TargetInternal::JoinString(const std::vector<std::string>& array, char separator) {
  std::string result;
  ICHECK(separator != quote && separator != escape)
      << "string join separator cannot be " << quote << " or " << escape;

  bool is_first = true;
  for (const auto& s : array) {
    if (!is_first) {
      result.push_back(separator);
    }
    result.append(s);
    is_first = false;
  }

  return result;
}

std::vector<std::string> TargetInternal::SplitString(const std::string& str, char separator) {
  std::vector<std::string> output;

  const char* start = str.data();
  const char* end = start + str.size();
  const char* pos = start;

  std::stringstream current_word;

  auto finish_word = [&]() {
    std::string word = current_word.str();
    if (word.size()) {
      output.push_back(word);
      current_word.str("");
    }
  };

  bool pos_quoted = false;

  while (pos < end) {
    if ((*pos == separator) && !pos_quoted) {
      finish_word();
      pos++;
    } else if (*pos == escape && pos + 1 < end) {
      current_word << escape;
      current_word << pos[1];
      pos += 2;
    } else if (*pos == quote) {
      current_word << quote;
      pos_quoted = !pos_quoted;
      pos++;
    } else {
      current_word << *pos;
      pos++;
    }
  }

  ICHECK(!pos_quoted) << "Mismatched quotes '' in string";

  finish_word();

  return output;
}

std::string TargetInternal::Interpret(const std::string& str) {
  // String interpretation deals with quotes (') and escapes(\).
  // - An escape character must be followed by another character forming an
  //   "escape sequence". (Trailing escape is not allowed.) An escape prevents
  //   interpretation of the character that follows. This happens regardless of
  //   whether the escape sequence appears within quoted substring or not.
  // - A quote character, when interpreted, marks the beginning or the end of a
  //   quoted substring. (A quoted substring cannot contain unescaped quotes.)
  // - Any other character, when interpreted, represents itself.
  //
  // Interpretation happens in two steps:
  // 1. If the entire string is quoted, the quotes are removed first, and the
  //    resulting string is treated as unquoted.
  // 2. Each character or escape sequence is interpreted, and the result is copied
  //    to the result. When not inside a quoted substring, the interpretation of an
  //    escape sequence is the escaped character, otherwise it is the entire escape
  //    sequence.
  //
  // Examples:
  //    blah                -> blah         Nothing happened
  //    'blah'              -> blah         Enclosing quotes removed
  //    'bl'ah              -> 'bl'ah       Non-enclosing quotes remain
  //    '\'blah\''          -> 'blah'       Enclosing quotes removed, escaped quotes
  //                                        interpreted.
  //    '\'\\\'blah\\\'\''  -> '\'blah\''   Same as above.
  //
  // Note that
  //    '\'\\\'blah\\\'\'' -> '\'blah\'' -> 'blah'

  std::string result;
  if (str.empty()) {
    return result;
  }

  // Check if the entire string is enclosed in quotes ''. If so, strip the quotes
  // and treat the string as unquoted (so that escapes are interpreted). Doing that
  // will allow '\'foo\'' to become 'foo', instead of \'foo\'.
  std::string::size_type start = 0, end = str.size();
  if (IsQuoted(str)) {
    start++;
    end--;
  }

  bool inside_quote = false;
  bool escaping = false;

  for (auto i = start, e = end; i < e; ++i) {
    std::string::value_type c = str[i];
    if (escaping) {
      escaping = false;
    } else if (c == escape) {
      escaping = true;
      if (!inside_quote) {
        continue;
      }
    } else if (c == quote) {
      inside_quote = !inside_quote;
    }
    result.push_back(c);
  }

  return result;
}

std::string TargetInternal::Uninterpret(const std::string& str) {
  // Do the opposite to `Interpret`, so that Interpret(Uninterpret(str)) == str.
  std::string result;

  for (std::string::size_type i = 0, e = str.size(); i < e; ++i) {
    std::string::value_type c = str[i];
    if (c == escape || c == quote) {
      result.push_back(escape);
    }
    result.push_back(c);
  }

  return result;
}

static int ParseKVPair(const std::string& s, const std::string& s_next, std::string* key,
                       std::string* value) {
  std::string::size_type pos;
  std::string& result_k = *key;
  std::string& result_v = *value;
  if ((pos = s.find_first_of('=')) != std::string::npos) {
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
  std::string interp_str = Interpret(str);
  if (info.type_index == runtime::Int::ContainerType::_GetOrAllocRuntimeTypeIndex() ||
      info.type_index == runtime::Bool::ContainerType::_GetOrAllocRuntimeTypeIndex()) {
    // Parsing integer or boolean
    std::istringstream is(interp_str);
    int v;
    if (!(is >> v)) {
      std::string lower(interp_str.size(), '\x0');
      std::transform(interp_str.begin(), interp_str.end(), lower.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      // Mimic C++ automatic conversions, allowing bool to be used for
      // integer parameters.
      if (lower == "true") {
        v = 1;
      } else if (lower == "false") {
        v = 0;
      } else {
        throw Error(": Cannot parse integer from string: " + interp_str);
      }
    }

    if (info.type_index == runtime::Int::ContainerType::_GetOrAllocRuntimeTypeIndex()) {
      return runtime::Int(v);
    } else {
      return runtime::Bool(v);
    }
  } else if (info.type_index == String::ContainerType::_GetOrAllocRuntimeTypeIndex()) {
    // Parsing string, strip leading/trailing spaces, and enclosing quotes if any
    auto start = interp_str.find_first_not_of(' ');
    auto end = interp_str.find_last_not_of(' ');
    if (start == std::string::npos || end == std::string::npos) {
      // The whole string is made of spaces.
      return String();
    }
    return String(interp_str.substr(start, (end - start + 1)));

  } else if (info.type_index == Target::ContainerType::_GetOrAllocRuntimeTypeIndex()) {
    // Parsing target
    return Target(TargetInternal::FromString(interp_str));
  } else if (info.type_index == ArrayNode::_GetOrAllocRuntimeTypeIndex()) {
    // Parsing array
    std::vector<ObjectRef> result;
    for (const std::string& substr : SplitString(interp_str, ',')) {
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
  throw Error(": Unsupported type \"" + info.type_key +
              "\" for parsing from string: " + interp_str);
}

ObjectRef TargetInternal::ParseType(const ObjectRef& obj,
                                    const TargetKindNode::ValueTypeInfo& info) {
  if (info.type_index == runtime::Int::ContainerType::_GetOrAllocRuntimeTypeIndex()) {
    // Parsing integer
    return GetRef<runtime::Int>(ObjTypeCheck<runtime::Int::ContainerType>(obj, "runtime.BoxInt"));
  } else if (info.type_index == String::ContainerType::RuntimeTypeIndex()) {
    // Parsing string
    return GetRef<String>(ObjTypeCheck<StringObj>(obj, "String"));
  } else if (info.type_index == Target::ContainerType::RuntimeTypeIndex()) {
    // Parsing target
    if (auto opt = obj.as<Target>()) {
      return opt.value();
    } else if (auto str = obj.as<String>()) {
      return Target(TargetInternal::FromString(str.value()));
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

std::string TargetInternal::StringifyAtomicType(const ObjectRef& obj) {
  if (const auto* p = obj.as<runtime::Int::ContainerType>()) {
    return std::to_string(p->value);
  } else if (const auto* p = obj.as<runtime::Bool::ContainerType>()) {
    return std::to_string(p->value);
  } else if (const auto* p = obj.as<IntImmNode>()) {
    return std::to_string(p->value);
  }
  if (auto tvm_str = obj.as<String>()) {
    std::string s = tvm_str.value();
    auto u = Uninterpret(s);
    if (u.find_first_of(' ') != std::string::npos && !IsQuoted(u)) {
      u = Quote(u);
    }
    return u;
  }
  LOG(FATAL) << "Cannot stringify object of type " << obj->GetTypeKey();
}

std::string TargetInternal::StringifyArray(const ArrayNode& array) {
  std::vector<std::string> elements;

  for (const ObjectRef& item : array) {
    std::string s = StringifyAtomicType(item);
    std::string u = Uninterpret(s);
    if (u.find_first_of(',') != std::string::npos && !IsQuoted(u)) {
      u = Quote(u);
    }
    elements.push_back(u);
  }

  return JoinString(elements, ',');
}

Optional<String> TargetInternal::StringifyAttrsToRaw(const Map<String, ObjectRef>& attrs) {
  std::ostringstream os;
  std::vector<String> keys;
  for (const auto& kv : attrs) {
    keys.push_back(kv.first);
  }
  std::sort(keys.begin(), keys.end());
  std::vector<std::string> result;

  for (const auto& key : keys) {
    const ObjectRef& obj = attrs[key];
    std::string value;
    if (const auto* array = obj.as<ArrayNode>()) {
      value = String(StringifyArray(*array));
    } else {
      value = StringifyAtomicType(obj);
    }
    if (!value.empty()) {
      result.push_back("-" + key + "=" + value);
    }
  }
  return String(JoinString(result, ' '));
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
  n->host = std::move(host);
  data_ = std::move(n);
}

Target::Target(TargetKind kind, Optional<ObjectRef> host, String tag, Array<String> keys,
               Map<String, ObjectRef> attrs) {
  auto data = runtime::make_object<TargetNode>();
  data->kind = std::move(kind);
  data->host = std::move(host);
  data->tag = std::move(tag);
  data->keys = std::move(keys);
  data->attrs = std::move(attrs);
  data_ = std::move(data);
}

bool Target::IsExternalCodegen() const {
  TargetKindAttrMap<Bool> is_external_codegen_map =
      TargetKind::GetAttrMap<Bool>(tvm::attr::kIsExternalCodegen);
  TargetKindAttrMap<tvm::transform::Pass> relay_to_tir_map =
      TargetKind::GetAttrMap<tvm::transform::Pass>(tvm::attr::kRelayToTIR);
  return is_external_codegen_map.get(get()->kind, Bool(false)) ||
         relay_to_tir_map.count(get()->kind);
}

bool Target::IsExternalCodegenFor(const Target& that) const {
  return get()->GetTargetDeviceType() == that->GetTargetDeviceType() && IsExternalCodegen() &&
         !that.IsExternalCodegen();
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
  if (this->host.defined()) {
    result.Set("host", this->GetHost().value_or(Target())->Export());
  }
  for (const auto& kv : attrs) {
    result.Set(kv.first, kv.second);
  }
  return result;
}

Optional<Target> TargetNode::GetHost() const { return this->host.as<Target>(); }

Target Target::WithoutHost() const {
  if ((*this)->GetHost()) {
    auto output = make_object<TargetNode>(*get());
    output->host = NullOpt;
    return Target(output);
  } else {
    return *this;
  }
}

int TargetNode::GetTargetDeviceType() const {
  if (Optional<Integer> device_type = GetAttr<Integer>("target_device_type")) {
    return Downcast<Integer>(device_type)->value;
  }
  return kind->default_device_type;
}

bool TargetNode::HasKey(const std::string& query_key) const {
  return std::any_of(keys.begin(), keys.end(),
                     [&query_key](const auto& key) { return key == query_key; });
}

String TargetNode::ToDebugString() const {
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

bool TargetNode::SEqualReduce(const TargetNode* other, SEqualReducer equal) const {
  return equal(kind.get(), other->kind.get()) && equal(host, other->host) &&
         equal(tag, other->tag) && equal(keys, other->keys) && equal(attrs, other->attrs);
}

void TargetNode::SHashReduce(SHashReducer hash_reduce) const {
  hash_reduce(kind.get());
  hash_reduce(host);
  hash_reduce(tag);
  hash_reduce(keys);
  hash_reduce(attrs);
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
  ICHECK_GT(target_str.length(), 0) << "Cannot parse empty target string";
  // Split the string by empty spaces
  std::vector<std::string> options = SplitString(std::string(target_str), ' ');
  std::string name = options[0];
  // Create the target config
  std::unordered_map<String, ObjectRef> config = {{"kind", String(name)}};
  TargetKind kind = GetTargetKind(name);
  for (size_t iter = 1, end = options.size(); iter < end;) {
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

ObjectPtr<Object> TargetInternal::FromConfig(Map<String, ObjectRef> config) {
  const String kKind = "kind";
  const String kTag = "tag";
  const String kKeys = "keys";
  const String kDeviceName = "device";
  const String kHost = "host";
  const String kFeatures = "features";
  ObjectPtr<TargetNode> target = make_object<TargetNode>();

  ICHECK(!config.count(kFeatures)) << "Target Features should be generated by Target parser";

  // parse 'kind'
  if (config.count(kKind)) {
    if (auto kind = config[kKind].as<String>()) {
      target->kind = GetTargetKind(kind.value());
      ICHECK(!(target->kind->preprocessor != nullptr && target->kind->target_parser != nullptr))
          << "Cannot use both set_attrs_preprocessor and set_target_parser";

      // Run JSON Parser over JSON input
      if (target->kind->target_parser != nullptr) {
        VLOG(9) << "TargetInternal::FromConfig - Running target_parser";
        config = target->kind->target_parser(config);
        if (config.count(kFeatures)) {
          target->features = Downcast<Map<String, ObjectRef>>(config[kFeatures]);
          config.erase(kFeatures);
        }
      }

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
    if (auto tag = config[kTag].as<String>()) {
      target->tag = tag.value();
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
    bool has_user_keys = config.count(kKeys);
    if (has_user_keys) {
      // user provided keys
      if (const auto* cfg_keys = config[kKeys].as<ArrayNode>()) {
        for (const ObjectRef& e : *cfg_keys) {
          if (auto key = e.as<String>()) {
            keys.push_back(key.value());
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
      if (auto device = config.at(kDeviceName).as<String>()) {
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
    target->host = PackedFunc(ConstructorDispatcher)(config[kHost]).AsObjectRef<Target>();
    config.erase(kHost);
  } else {
    target->host = NullOpt;
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

  // If requested, query attributes from the device.  User-specified
  // parameters take precedence over queried parameters.
  if (attrs.count("from_device")) {
    int device_id = Downcast<runtime::Int>(attrs.at("from_device"))->value;
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
    target->attrs = target->kind->preprocessor(Map<String, ObjectRef>(attrs));
  } else {
    target->attrs = attrs;
  }

  return target;
}  // namespace tvm

std::unordered_map<String, ObjectRef> TargetInternal::QueryDevice(int device_id,
                                                                  const TargetNode* target) {
  std::unordered_map<String, ObjectRef> output;

  Device device{static_cast<DLDeviceType>(target->GetTargetDeviceType()), device_id};

  auto api = runtime::DeviceAPI::Get(device, true);
  if (!api) {
    LOG(INFO) << "Requested reading the parameters for " << target->kind->name << " from device_id "
              << device_id << ", but support for this runtime wasn't enabled at compile-time.  "
              << "Using default target parameters.";
    return output;
  }

  TVMRetValue ret;
  api->GetAttr(device, runtime::kExist, &ret);
  bool device_exists = ret;
  if (!device_exists) {
    ICHECK(device_exists) << "Requested reading the parameters for " << target->kind->name
                          << " from device_id " << device_id << ", but device_id " << device_id
                          << " doesn't exist.  Using default target parameters.";
    return output;
  }

  for (const auto& kv : target->kind->key2vtype_) {
    const String& key = kv.first;

    TVMRetValue ret;
    api->GetTargetProperty(device, key, &ret);

    // Delegate conversion from TVMRetValue to the FFI's default conversions.
    if (Optional<ObjectRef> opt = ret) {
      output[key] = opt.value();
    }
  }

  return output;
}

/**********  Registry  **********/

TVM_REGISTER_GLOBAL("target.Target").set_body(TargetInternal::ConstructorDispatcher);
TVM_REGISTER_GLOBAL("target.TargetEnterScope").set_body_typed(TargetInternal::EnterScope);
TVM_REGISTER_GLOBAL("target.TargetExitScope").set_body_typed(TargetInternal::ExitScope);
TVM_REGISTER_GLOBAL("target.TargetCurrent").set_body_typed(Target::Current);
TVM_REGISTER_GLOBAL("target.TargetExport").set_body_typed(TargetInternal::Export);
TVM_REGISTER_GLOBAL("target.WithHost").set_body_typed(TargetInternal::WithHost);
TVM_REGISTER_GLOBAL("target.TargetGetDeviceType").set_body_typed([](const Target& target) {
  return target->GetTargetDeviceType();
});
TVM_REGISTER_GLOBAL("target.TargetGetFeature")
    .set_body_typed([](const Target& target, const String& feature_key) {
      return target->GetFeature<ObjectRef>(feature_key);
    });

TVM_STATIC_IR_FUNCTOR(ReprPrinter, vtable)
    .set_dispatch<TargetNode>([](const ObjectRef& obj, ReprPrinter* p) {
      p->stream << Downcast<Target>(obj)->str();
    });

}  // namespace tvm
