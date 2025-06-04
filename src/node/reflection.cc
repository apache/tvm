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
 * Reflection utilities.
 * \file node/reflection.cc
 */
#include <tvm/ffi/function.h>
#include <tvm/ir/attrs.h>
#include <tvm/node/node.h>
#include <tvm/node/reflection.h>

namespace tvm {

using ffi::Any;
using ffi::Function;
using ffi::PackedArgs;

// Attr getter.
class AttrGetter : public AttrVisitor {
 public:
  const String& skey;
  ffi::Any* ret;

  AttrGetter(const String& skey, ffi::Any* ret) : skey(skey), ret(ret) {}

  bool found_ref_object{false};

  void Visit(const char* key, double* value) final {
    if (skey == key) *ret = value[0];
  }
  void Visit(const char* key, int64_t* value) final {
    if (skey == key) *ret = value[0];
  }
  void Visit(const char* key, uint64_t* value) final {
    ICHECK_LE(value[0], static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
        << "cannot return too big constant";
    if (skey == key) *ret = static_cast<int64_t>(value[0]);
  }
  void Visit(const char* key, int* value) final {
    if (skey == key) *ret = static_cast<int64_t>(value[0]);
  }
  void Visit(const char* key, bool* value) final {
    if (skey == key) *ret = static_cast<int64_t>(value[0]);
  }
  void Visit(const char* key, void** value) final {
    if (skey == key) *ret = static_cast<void*>(value[0]);
  }
  void Visit(const char* key, DataType* value) final {
    if (skey == key) *ret = value[0];
  }
  void Visit(const char* key, std::string* value) final {
    if (skey == key) *ret = value[0];
  }
  void Visit(const char* key, Optional<double>* value) final {
    if (skey == key) {
      *ret = value[0];
      found_ref_object = true;
    }
  }
  void Visit(const char* key, Optional<int64_t>* value) final {
    if (skey == key) {
      *ret = value[0];
      found_ref_object = true;
    }
  }

  void Visit(const char* key, runtime::NDArray* value) final {
    if (skey == key) {
      *ret = value[0];
      found_ref_object = true;
    }
  }
  void Visit(const char* key, runtime::ObjectRef* value) final {
    if (skey == key) {
      *ret = value[0];
      found_ref_object = true;
    }
  }
};

ffi::Any ReflectionVTable::GetAttr(Object* self, const String& field_name) const {
  ffi::Any ret;
  AttrGetter getter(field_name, &ret);

  bool success;
  if (getter.skey == "type_key") {
    ret = self->GetTypeKey();
    success = true;
  } else if (!self->IsInstance<DictAttrsNode>()) {
    VisitAttrs(self, &getter);
    success = getter.found_ref_object || ret != nullptr;
  } else {
    // specially handle dict attr
    DictAttrsNode* dnode = static_cast<DictAttrsNode*>(self);
    auto it = dnode->dict.find(getter.skey);
    if (it != dnode->dict.end()) {
      success = true;
      ret = (*it).second;
    } else {
      success = false;
    }
  }
  if (!success) {
    LOG(FATAL) << "AttributeError: " << self->GetTypeKey() << " object has no attributed "
               << getter.skey;
  }
  return ret;
}

// List names;
class AttrDir : public AttrVisitor {
 public:
  std::vector<std::string>* names;

  void Visit(const char* key, double* value) final { names->push_back(key); }
  void Visit(const char* key, int64_t* value) final { names->push_back(key); }
  void Visit(const char* key, uint64_t* value) final { names->push_back(key); }
  void Visit(const char* key, bool* value) final { names->push_back(key); }
  void Visit(const char* key, int* value) final { names->push_back(key); }
  void Visit(const char* key, void** value) final { names->push_back(key); }
  void Visit(const char* key, DataType* value) final { names->push_back(key); }
  void Visit(const char* key, std::string* value) final { names->push_back(key); }
  void Visit(const char* key, runtime::NDArray* value) final { names->push_back(key); }
  void Visit(const char* key, runtime::ObjectRef* value) final { names->push_back(key); }
  void Visit(const char* key, Optional<double>* value) final { names->push_back(key); }
  void Visit(const char* key, Optional<int64_t>* value) final { names->push_back(key); }
};

std::vector<std::string> ReflectionVTable::ListAttrNames(Object* self) const {
  std::vector<std::string> names;
  AttrDir dir;
  dir.names = &names;

  if (!self->IsInstance<DictAttrsNode>()) {
    VisitAttrs(self, &dir);
  } else {
    // specially handle dict attr
    DictAttrsNode* dnode = static_cast<DictAttrsNode*>(self);
    for (const auto& kv : dnode->dict) {
      names.push_back(kv.first);
    }
  }
  return names;
}

ReflectionVTable* ReflectionVTable::Global() {
  static ReflectionVTable inst;
  return &inst;
}

ObjectPtr<Object> ReflectionVTable::CreateInitObject(const std::string& type_key,
                                                     const std::string& repr_bytes) const {
  int32_t tindex;
  TVMFFIByteArray type_key_arr{type_key.data(), type_key.length()};
  TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_arr, &tindex));
  if (static_cast<size_t>(tindex) >= fcreate_.size() || fcreate_[tindex] == nullptr) {
    LOG(FATAL) << "TypeError: " << type_key << " is not registered via TVM_REGISTER_NODE_TYPE";
  }
  return fcreate_[tindex](repr_bytes);
}

class NodeAttrSetter : public AttrVisitor {
 public:
  std::string type_key;
  std::unordered_map<std::string, ffi::AnyView> attrs;

  void Visit(const char* key, double* value) final { *value = GetAttr(key).cast<double>(); }
  void Visit(const char* key, int64_t* value) final { *value = GetAttr(key).cast<int64_t>(); }
  void Visit(const char* key, uint64_t* value) final { *value = GetAttr(key).cast<uint64_t>(); }
  void Visit(const char* key, int* value) final { *value = GetAttr(key).cast<int>(); }
  void Visit(const char* key, bool* value) final { *value = GetAttr(key).cast<bool>(); }
  void Visit(const char* key, std::string* value) final {
    *value = GetAttr(key).cast<std::string>();
  }
  void Visit(const char* key, void** value) final { *value = GetAttr(key).cast<void*>(); }
  void Visit(const char* key, DataType* value) final { *value = GetAttr(key).cast<DataType>(); }
  void Visit(const char* key, runtime::NDArray* value) final {
    *value = GetAttr(key).cast<runtime::NDArray>();
  }
  void Visit(const char* key, ObjectRef* value) final { *value = GetAttr(key).cast<ObjectRef>(); }

  void Visit(const char* key, Optional<double>* value) final {
    *value = GetAttr(key).cast<Optional<double>>();
  }
  void Visit(const char* key, Optional<int64_t>* value) final {
    *value = GetAttr(key).cast<Optional<int64_t>>();
  }

 private:
  ffi::AnyView GetAttr(const char* key) {
    auto it = attrs.find(key);
    if (it == attrs.end()) {
      LOG(FATAL) << type_key << ": require field " << key;
    }
    ffi::AnyView v = it->second;
    attrs.erase(it);
    return v;
  }
};

void InitNodeByPackedArgs(ReflectionVTable* reflection, Object* n, const ffi::PackedArgs& args) {
  NodeAttrSetter setter;
  setter.type_key = n->GetTypeKey();
  ICHECK_EQ(args.size() % 2, 0);
  for (int i = 0; i < args.size(); i += 2) {
    setter.attrs.emplace(args[i].cast<std::string>(), args[i + 1]);
  }
  reflection->VisitAttrs(n, &setter);

  if (setter.attrs.size() != 0) {
    std::ostringstream os;
    os << setter.type_key << " does not contain field ";
    for (const auto& kv : setter.attrs) {
      os << " " << kv.first;
    }
    LOG(FATAL) << os.str();
  }
}

ObjectRef ReflectionVTable::CreateObject(const std::string& type_key,
                                         const ffi::PackedArgs& kwargs) {
  ObjectPtr<Object> n = this->CreateInitObject(type_key);
  if (n->IsInstance<BaseAttrsNode>()) {
    static_cast<BaseAttrsNode*>(n.get())->InitByPackedArgs(kwargs);
  } else {
    InitNodeByPackedArgs(this, n.get(), kwargs);
  }
  return ObjectRef(n);
}

ObjectRef ReflectionVTable::CreateObject(const std::string& type_key,
                                         const Map<String, Any>& kwargs) {
  // Redirect to the ffi::PackedArgs version
  // It is not the most efficient way, but CreateObject is not meant to be used
  // in a fast code-path and is mainly reserved as a flexible API for frontends.
  std::vector<AnyView> packed_args(kwargs.size() * 2);
  int index = 0;

  for (const auto& kv : *static_cast<const ffi::MapObj*>(kwargs.get())) {
    packed_args[index] = kv.first.cast<String>().c_str();
    packed_args[index + 1] = kv.second;
    index += 2;
  }

  return CreateObject(type_key, ffi::PackedArgs(packed_args.data(), packed_args.size()));
}

// Expose to FFI APIs.
void NodeGetAttr(ffi::PackedArgs args, ffi::Any* ret) {
  Object* self = const_cast<Object*>(args[0].cast<const Object*>());
  *ret = ReflectionVTable::Global()->GetAttr(self, args[1].cast<std::string>());
}

void NodeListAttrNames(ffi::PackedArgs args, ffi::Any* ret) {
  Object* self = const_cast<Object*>(args[0].cast<const Object*>());

  auto names =
      std::make_shared<std::vector<std::string>>(ReflectionVTable::Global()->ListAttrNames(self));

  *ret = ffi::Function([names](ffi::PackedArgs args, ffi::Any* rv) {
    int64_t i = args[0].cast<int64_t>();
    if (i == -1) {
      *rv = static_cast<int64_t>(names->size());
    } else {
      *rv = (*names)[i];
    }
  });
}

// API function to make node.
// args format:
//   key1, value1, ..., key_n, value_n
void MakeNode(const ffi::PackedArgs& args, ffi::Any* rv) {
  auto type_key = args[0].cast<std::string>();
  *rv = ReflectionVTable::Global()->CreateObject(type_key, args.Slice(1));
}

TVM_FFI_REGISTER_GLOBAL("node.NodeGetAttr").set_body_packed(NodeGetAttr);

TVM_FFI_REGISTER_GLOBAL("node.NodeListAttrNames").set_body_packed(NodeListAttrNames);

TVM_FFI_REGISTER_GLOBAL("node.MakeNode").set_body_packed(MakeNode);

namespace {
// Attribute visitor class for finding the attribute key by its address
class GetAttrKeyByAddressVisitor : public AttrVisitor {
 public:
  explicit GetAttrKeyByAddressVisitor(const void* attr_address)
      : attr_address_(attr_address), key_(nullptr) {}

  void Visit(const char* key, double* value) final { DoVisit(key, value); }
  void Visit(const char* key, int64_t* value) final { DoVisit(key, value); }
  void Visit(const char* key, uint64_t* value) final { DoVisit(key, value); }
  void Visit(const char* key, int* value) final { DoVisit(key, value); }
  void Visit(const char* key, bool* value) final { DoVisit(key, value); }
  void Visit(const char* key, std::string* value) final { DoVisit(key, value); }
  void Visit(const char* key, void** value) final { DoVisit(key, value); }
  void Visit(const char* key, DataType* value) final { DoVisit(key, value); }
  void Visit(const char* key, runtime::NDArray* value) final { DoVisit(key, value); }
  void Visit(const char* key, runtime::ObjectRef* value) final { DoVisit(key, value); }
  void Visit(const char* key, Optional<double>* value) final { DoVisit(key, value); }
  void Visit(const char* key, Optional<int64_t>* value) final { DoVisit(key, value); }
  const char* GetKey() const { return key_; }

 private:
  const void* attr_address_;
  const char* key_;

  void DoVisit(const char* key, const void* candidate) {
    if (attr_address_ == candidate) {
      key_ = key;
    }
  }
};
}  // anonymous namespace

Optional<String> GetAttrKeyByAddress(const Object* object, const void* attr_address) {
  GetAttrKeyByAddressVisitor visitor(attr_address);
  ReflectionVTable::Global()->VisitAttrs(const_cast<Object*>(object), &visitor);
  const char* key = visitor.GetKey();
  if (key == nullptr) {
    return std::nullopt;
  } else {
    return String(key);
  }
}

}  // namespace tvm
