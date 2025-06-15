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
#ifndef TVM_RUNTIME_DISCO_PROTOCOL_H_
#define TVM_RUNTIME_DISCO_PROTOCOL_H_

#include <dmlc/io.h>
#include <dmlc/memory_io.h>
#include <tvm/ffi/function.h>
#include <tvm/runtime/base.h>
#include <tvm/runtime/disco/session.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "../../support/arena.h"
#include "../../support/base64.h"
#include "../minrpc/rpc_reference.h"

namespace tvm {
namespace runtime {

/*!
 * \brief The communication protocol used by Disco message channel.
 * \tparam SubClassType The subclass type that inherits this protocol.
 */
template <class SubClassType>
struct DiscoProtocol {
 protected:
  /*! \brief Virtual destructor */
  virtual ~DiscoProtocol() = default;

  /*! \brief Recycle all the memory used in the arena */
  inline void RecycleAll() {
    this->object_arena_.clear();
    this->arena_.RecycleAll();
  }

  /*! \brief Get the length of the object being serialized. Used by RPCReference. */
  inline uint64_t GetObjectBytes(Object* obj);

  /*! \brief Write the object to stream. Used by RPCReference. */
  inline void WriteObject(Object* obj);

  /*! \brief Read the object from stream. Used by RPCReference. */
  inline void ReadObject(TVMFFIAny* out);

  /*! \brief Callback method used when starting a new message. Used by RPCReference. */
  void MessageStart(uint64_t packet_nbytes) {}

  /*! \brief Callback method used when a new message is complete. Used by RPCReference. */
  void MessageDone() {}

  /*! \brief Callback method when an error occurs in (de)-serialization. Used by RPCReference. */
  void ThrowError(RPCServerStatus status) {
    LOG(FATAL) << "InternalError: Unexpected error in RPC: " << RPCServerStatusToString(status);
  }

  /*!\ brief Arena used by RPCReference to allocate POD memory */
  template <typename T>
  T* ArenaAlloc(int count) {
    static_assert(std::is_pod<T>::value, "need to be trival");
    return arena_.template allocate_<T>(count);
  }

  support::Arena arena_;
  std::vector<ObjectRef> object_arena_;
  friend struct RPCReference;
};

/*!
 * \brief The debug extension of the communication protocol that allows serialization and
 * deserialization of NDArrays and reflection-capable TVM objects.
 */
struct DiscoDebugObject : public Object {
 public:
  /*! \brief The data to be serialized */
  ffi::Any data;

  /*! \brief Wrap an NDArray or reflection-capable TVM object into the debug extension. */
  static ObjectRef Wrap(const ffi::Any& data) {
    ObjectPtr<DiscoDebugObject> n = make_object<DiscoDebugObject>();
    n->data = data;
    return ObjectRef(n);
  }

  /*! \brief Wrap an NDArray or reflection-capable TVM object into the debug extension. */
  static ObjectRef Wrap(const ffi::AnyView& data) {
    ffi::Any rv;
    rv = data;
    return Wrap(std::move(rv));
  }

  /*! \brief Serialize the debug object to string */
  inline std::string SaveToStr() const;
  /*! \brief Deserialize the debug object from string */
  static inline ObjectPtr<DiscoDebugObject> LoadFromStr(std::string json_str);
  /*! \brief Get the size of the debug object in bytes */
  inline uint64_t GetObjectBytes() const { return sizeof(uint64_t) + this->SaveToStr().size(); }

  static constexpr const char* _type_key = "runtime.disco.DiscoDebugObject";
  TVM_DECLARE_FINAL_OBJECT_INFO(DiscoDebugObject, SessionObj);
};

template <class SubClassType>
inline uint64_t DiscoProtocol<SubClassType>::GetObjectBytes(Object* obj) {
  if (obj->IsInstance<DRefObj>()) {
    return sizeof(uint32_t) + sizeof(int64_t);
  } else if (obj->IsInstance<ffi::StringObj>()) {
    uint64_t size = static_cast<ffi::StringObj*>(obj)->size;
    return sizeof(uint32_t) + sizeof(uint64_t) + size * sizeof(char);
  } else if (obj->IsInstance<ffi::BytesObj>()) {
    uint64_t size = static_cast<ffi::BytesObj*>(obj)->size;
    return sizeof(uint32_t) + sizeof(uint64_t) + size * sizeof(char);
  } else if (obj->IsInstance<ffi::ShapeObj>()) {
    uint64_t ndim = static_cast<ffi::ShapeObj*>(obj)->size;
    return sizeof(uint32_t) + sizeof(uint64_t) + ndim * sizeof(ffi::ShapeObj::index_type);
  } else if (obj->IsInstance<DiscoDebugObject>()) {
    return sizeof(uint32_t) + static_cast<DiscoDebugObject*>(obj)->GetObjectBytes();
  } else {
    LOG(FATAL) << "ValueError: Object type is not supported in Disco calling convention: "
               << obj->GetTypeKey() << " (type_index = " << obj->type_index() << ")";
  }
}
template <class SubClassType>
inline void DiscoProtocol<SubClassType>::WriteObject(Object* obj) {
  SubClassType* self = static_cast<SubClassType*>(this);
  if (obj->IsInstance<DRefObj>()) {
    int64_t reg_id = static_cast<DRefObj*>(obj)->reg_id;
    self->template Write<uint32_t>(TypeIndex::kRuntimeDiscoDRef);
    self->template Write<int64_t>(reg_id);
  } else if (obj->IsInstance<ffi::StringObj>()) {
    ffi::StringObj* str = static_cast<ffi::StringObj*>(obj);
    self->template Write<uint32_t>(ffi::TypeIndex::kTVMFFIStr);
    self->template Write<uint64_t>(str->size);
    self->template WriteArray<char>(str->data, str->size);
  } else if (obj->IsInstance<ffi::BytesObj>()) {
    ffi::BytesObj* bytes = static_cast<ffi::BytesObj*>(obj);
    self->template Write<uint32_t>(ffi::TypeIndex::kTVMFFIBytes);
    self->template Write<uint64_t>(bytes->size);
    self->template WriteArray<char>(bytes->data, bytes->size);
  } else if (obj->IsInstance<ffi::ShapeObj>()) {
    ffi::ShapeObj* shape = static_cast<ffi::ShapeObj*>(obj);
    self->template Write<uint32_t>(ffi::TypeIndex::kTVMFFIShape);
    self->template Write<uint64_t>(shape->size);
    self->template WriteArray<ffi::ShapeObj::index_type>(shape->data, shape->size);
  } else if (obj->IsInstance<DiscoDebugObject>()) {
    self->template Write<uint32_t>(0);
    std::string str = static_cast<DiscoDebugObject*>(obj)->SaveToStr();
    self->template Write<uint64_t>(str.size());
    self->template WriteArray<char>(str.data(), str.size());
  } else {
    LOG(FATAL) << "ValueError: Object type is not supported in Disco calling convention: "
               << obj->GetTypeKey() << " (type_index = " << obj->type_index() << ")";
  }
}

template <class SubClassType>
inline void DiscoProtocol<SubClassType>::ReadObject(TVMFFIAny* out) {
  SubClassType* self = static_cast<SubClassType*>(this);
  ObjectRef result{nullptr};
  uint32_t type_index;
  self->template Read<uint32_t>(&type_index);
  if (type_index == TypeIndex::kRuntimeDiscoDRef) {
    ObjectPtr<DRefObj> dref = make_object<DRefObj>();
    self->template Read<int64_t>(&dref->reg_id);
    dref->session = Session{nullptr};
    result = ObjectRef(std::move(dref));
  } else if (type_index == ffi::TypeIndex::kTVMFFIStr) {
    uint64_t size = 0;
    self->template Read<uint64_t>(&size);
    std::string data(size, '\0');
    self->template ReadArray<char>(data.data(), size);
    result = String(std::move(data));
  } else if (type_index == ffi::TypeIndex::kTVMFFIBytes) {
    uint64_t size = 0;
    self->template Read<uint64_t>(&size);
    std::string data(size, '\0');
    self->template ReadArray<char>(data.data(), size);
    result = ffi::Bytes(std::move(data));
  } else if (type_index == ffi::TypeIndex::kTVMFFIShape) {
    uint64_t ndim = 0;
    self->template Read<uint64_t>(&ndim);
    std::vector<ffi::ShapeObj::index_type> data(ndim);
    self->template ReadArray<ffi::ShapeObj::index_type>(data.data(), ndim);
    result = ffi::Shape(std::move(data));
  } else if (type_index == 0) {
    uint64_t size = 0;
    self->template Read<uint64_t>(&size);
    std::string data(size, '\0');
    self->template ReadArray<char>(data.data(), size);
    result = DiscoDebugObject::LoadFromStr(std::move(data))->data.cast<ObjectRef>();
  } else {
    LOG(FATAL) << "ValueError: Object type is not supported in Disco calling convention: "
               << Object::TypeIndex2Key(type_index) << " (type_index = " << type_index << ")";
  }
  *reinterpret_cast<ffi::AnyView*>(out) = result;
  object_arena_.push_back(result);
}

inline std::string DiscoDebugObject::SaveToStr() const {
  if (auto opt_nd = this->data.as<NDArray>()) {
    NDArray array = opt_nd.value();
    std::string result;
    {
      dmlc::MemoryStringStream mstrm(&result);
      support::Base64OutStream b64strm(&mstrm);
      runtime::SaveDLTensor(&b64strm, array.operator->());
      b64strm.Finish();
    }
    result.push_back('1');
    return result;
  } else if (auto opt_obj = this->data.as<ObjectRef>()) {
    ObjectRef obj = opt_obj.value();
    const auto f = tvm::ffi::Function::GetGlobal("node.SaveJSON");
    CHECK(f.has_value()) << "ValueError: Cannot serialize object in non-debugging mode: "
                         << obj->GetTypeKey();
    std::string result = (*f)(obj).cast<std::string>();
    result.push_back('0');
    return result;
  }
  LOG(FATAL) << "ValueError: Cannot serialize the following type code in non-debugging mode: "
             << this->data.GetTypeKey();
}

inline ObjectPtr<DiscoDebugObject> DiscoDebugObject::LoadFromStr(std::string json_str) {
  ICHECK(!json_str.empty());
  char control_bit = json_str.back();
  json_str.pop_back();
  ObjectPtr<DiscoDebugObject> result = make_object<DiscoDebugObject>();
  if (control_bit == '0') {
    const auto f = tvm::ffi::Function::GetGlobal("node.LoadJSON");
    CHECK(f.has_value()) << "ValueError: Cannot deserialize object in non-debugging mode";
    result->data = (*f)(json_str);
  } else if (control_bit == '1') {
    dmlc::MemoryStringStream mstrm(&json_str);
    support::Base64InStream b64strm(&mstrm);
    b64strm.InitPosition();
    runtime::NDArray array;
    ICHECK(array.Load(&b64strm));
    result->data = std::move(array);
  } else {
    LOG(FATAL) << "ValueError: Unsupported control bit: " << control_bit
               << ". Full string: " << json_str;
  }
  return result;
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_DISCO_PROTOCOL_H_
