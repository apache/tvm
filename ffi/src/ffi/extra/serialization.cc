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
/*
 * \file src/ffi/extra/serialization.cc
 *
 * \brief Reflection-based serialization utilities.
 */
#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/base64.h>
#include <tvm/ffi/extra/serialization.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

namespace tvm {
namespace ffi {

class ObjectGraphSerializer {
 public:
  static json::Value Serialize(const Any& value, Any metadata) {
    ObjectGraphSerializer serializer;
    json::Object result;
    result.Set("root_index", serializer.GetOrCreateNodeIndex(value));
    result.Set("nodes", std::move(serializer.nodes_));
    if (metadata != nullptr) {
      result.Set("metadata", metadata);
    }
    return result;
  }

 private:
  ObjectGraphSerializer() = default;

  int64_t GetOrCreateNodeIndex(const Any& value) {
    // already mapped value, return the index
    auto it = node_index_map_.find(value);
    if (it != node_index_map_.end()) {
      return (*it).second;
    }
    json::Object node;
    switch (value.type_index()) {
      case TypeIndex::kTVMFFINone: {
        node.Set("type", ffi::StaticTypeKey::kTVMFFINone);
        break;
      }
      case TypeIndex::kTVMFFIBool: {
        node.Set("type", ffi::StaticTypeKey::kTVMFFIBool);
        node.Set("data", details::AnyUnsafe::CopyFromAnyViewAfterCheck<bool>(value));
        break;
      }
      case TypeIndex::kTVMFFIInt: {
        node.Set("type", ffi::StaticTypeKey::kTVMFFIInt);
        node.Set("data", details::AnyUnsafe::CopyFromAnyViewAfterCheck<int64_t>(value));
        break;
      }
      case TypeIndex::kTVMFFIFloat: {
        node.Set("type", ffi::StaticTypeKey::kTVMFFIFloat);
        node.Set("data", details::AnyUnsafe::CopyFromAnyViewAfterCheck<double>(value));
        break;
      }
      case TypeIndex::kTVMFFIDataType: {
        DLDataType dtype = details::AnyUnsafe::CopyFromAnyViewAfterCheck<DLDataType>(value);
        node.Set("type", ffi::StaticTypeKey::kTVMFFIDataType);
        node.Set("data", DLDataTypeToString(dtype));
        break;
      }
      case TypeIndex::kTVMFFIDevice: {
        DLDevice device = details::AnyUnsafe::CopyFromAnyViewAfterCheck<DLDevice>(value);
        node.Set("type", ffi::StaticTypeKey::kTVMFFIDevice);
        node.Set("data", json::Array{
                             static_cast<int64_t>(device.device_type),
                             static_cast<int64_t>(device.device_id),
                         });
        break;
      }
      case TypeIndex::kTVMFFISmallStr:
      case TypeIndex::kTVMFFIStr: {
        String str = details::AnyUnsafe::CopyFromAnyViewAfterCheck<String>(value);
        node.Set("type", ffi::StaticTypeKey::kTVMFFIStr);
        node.Set("data", str);
        break;
      }
      case TypeIndex::kTVMFFISmallBytes:
      case TypeIndex::kTVMFFIBytes: {
        Bytes bytes = details::AnyUnsafe::CopyFromAnyViewAfterCheck<Bytes>(value);
        node.Set("type", ffi::StaticTypeKey::kTVMFFIBytes);
        node.Set("data", Base64Encode(bytes));
        break;
      }
      case TypeIndex::kTVMFFIArray: {
        Array<Any> array = details::AnyUnsafe::CopyFromAnyViewAfterCheck<Array<Any>>(value);
        node.Set("type", ffi::StaticTypeKey::kTVMFFIArray);
        node.Set("data", CreateArrayData(array));
        break;
      }
      case TypeIndex::kTVMFFIMap: {
        Map<Any, Any> map = details::AnyUnsafe::CopyFromAnyViewAfterCheck<Map<Any, Any>>(value);
        node.Set("type", ffi::StaticTypeKey::kTVMFFIMap);
        node.Set("data", CreateMapData(map));
        break;
      }
      case TypeIndex::kTVMFFIShape: {
        ffi::Shape shape = details::AnyUnsafe::CopyFromAnyViewAfterCheck<ffi::Shape>(value);
        node.Set("type", ffi::StaticTypeKey::kTVMFFIShape);
        node.Set("data", Array<int64_t>(shape->data, shape->data + shape->size));
        break;
      }
      default: {
        if (value.type_index() >= TypeIndex::kTVMFFIStaticObjectBegin) {
          // serialize type key since type index is runtime dependent
          node.Set("type", value.GetTypeKey());
          node.Set("data", CreateObjectData(value));
        } else {
          TVM_FFI_THROW(RuntimeError) << "Cannot serialize type `" << value.GetTypeKey() << "`";
          TVM_FFI_UNREACHABLE();
        }
      }
    }
    int64_t node_index = nodes_.size();
    nodes_.push_back(node);
    node_index_map_.Set(value, node_index);
    return node_index;
  }

  json::Array CreateArrayData(const Array<Any>& value) {
    json::Array data;
    data.reserve(value.size());
    for (const Any& item : value) {
      data.push_back(GetOrCreateNodeIndex(item));
    }
    return data;
  }

  json::Array CreateMapData(const Map<Any, Any>& value) {
    json::Array data;
    data.reserve(value.size() * 2);
    for (const auto& [key, value] : value) {
      data.push_back(GetOrCreateNodeIndex(key));
      data.push_back(GetOrCreateNodeIndex(value));
    }
    return data;
  }

  // create the data for the object, if the type has a custom data to json function,
  // use it. otherwise, we go over the fields and create the data.
  json::Value CreateObjectData(const Any& value) {
    static reflection::TypeAttrColumn data_to_json = reflection::TypeAttrColumn("__data_to_json__");
    if (data_to_json[value.type_index()] != nullptr) {
      return data_to_json[value.type_index()].cast<Function>()(value);
    }
    // NOTE: invariant: lhs and rhs are already the same type
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(value.type_index());
    if (type_info->metadata == nullptr) {
      TVM_FFI_THROW(TypeError) << "Type metadata is not set for type `"
                               << String(type_info->type_key)
                               << "`, so ToJSONGraph is not supported for this type";
    }
    const Object* obj = value.cast<const Object*>();
    json::Object data;
    // go over the content and hash the fields
    reflection::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* field_info) {
      // get the field value from both side
      reflection::FieldGetter getter(field_info);
      Any field_value = getter(obj);
      int field_static_type_index = field_info->field_static_type_index;
      String field_name(field_info->name);
      // for static field index that are known, we can directly set the field value.
      switch (field_static_type_index) {
        case TypeIndex::kTVMFFINone: {
          data.Set(field_name, nullptr);
          break;
        }
        case TypeIndex::kTVMFFIBool: {
          data.Set(field_name, details::AnyUnsafe::CopyFromAnyViewAfterCheck<bool>(field_value));
          break;
        }
        case TypeIndex::kTVMFFIInt: {
          data.Set(field_name, details::AnyUnsafe::CopyFromAnyViewAfterCheck<int64_t>(field_value));
          break;
        }
        case TypeIndex::kTVMFFIFloat: {
          data.Set(field_name, details::AnyUnsafe::CopyFromAnyViewAfterCheck<double>(field_value));
          break;
        }
        case TypeIndex::kTVMFFIDataType: {
          DLDataType dtype = details::AnyUnsafe::CopyFromAnyViewAfterCheck<DLDataType>(field_value);
          data.Set(field_name, DLDataTypeToString(dtype));
          break;
        }
        default: {
          // for dynamic field index, we need need to put them onto nodes
          int64_t node_index = GetOrCreateNodeIndex(field_value);
          data.Set(field_name, node_index);
          break;
        }
      }
    });
    return data;
  }

  // maps the original value to the index of the node in the nodes_ array
  Map<Any, int64_t> node_index_map_;
  // records nodes that are serialized
  json::Array nodes_;
};

json::Value ToJSONGraph(const Any& value, const Any& metadata) {
  return ObjectGraphSerializer::Serialize(value, metadata);
}

class ObjectGraphDeserializer {
 public:
  static Any Deserialize(const json::Value& value) {
    ObjectGraphDeserializer deserializer(value);
    return deserializer.GetOrDecodeNode(deserializer.root_index_);
  }

  Any GetOrDecodeNode(int64_t node_index) {
    // already decoded null index
    if (node_index == decoded_null_index_) {
      return Any(nullptr);
    }
    // already decoded
    if (decoded_nodes_[node_index] != nullptr) {
      return decoded_nodes_[node_index];
    }
    // now decode the node
    Any value = DecodeNode(nodes_[node_index].cast<json::Object>());
    decoded_nodes_[node_index] = value;
    if (value == nullptr) {
      decoded_null_index_ = node_index;
    }
    return value;
  }

 private:
  Any DecodeNode(const json::Object& node) {
    String type_key = node["type"].cast<String>();
    TVMFFIByteArray type_key_arr{type_key.data(), type_key.length()};
    int32_t type_index;
    TVM_FFI_CHECK_SAFE_CALL(TVMFFITypeKeyToIndex(&type_key_arr, &type_index));

    switch (type_index) {
      case TypeIndex::kTVMFFINone: {
        return nullptr;
      }
      case TypeIndex::kTVMFFIBool: {
        return node["data"].cast<bool>();
      }
      case TypeIndex::kTVMFFIInt: {
        return node["data"].cast<int64_t>();
      }
      case TypeIndex::kTVMFFIFloat: {
        return node["data"].cast<double>();
      }
      case TypeIndex::kTVMFFIDataType: {
        return StringToDLDataType(node["data"].cast<String>());
      }
      case TypeIndex::kTVMFFIDevice: {
        Array<int32_t> data = node["data"].cast<Array<int32_t>>();
        return DLDevice{static_cast<DLDeviceType>(data[0]), data[1]};
      }
      case TypeIndex::kTVMFFIStr: {
        return node["data"].cast<String>();
      }
      case TypeIndex::kTVMFFIBytes: {
        return Base64Decode(node["data"].cast<String>());
      }
      case TypeIndex::kTVMFFIMap: {
        return DecodeMapData(node["data"].cast<json::Array>());
      }
      case TypeIndex::kTVMFFIArray: {
        return DecodeArrayData(node["data"].cast<json::Array>());
      }
      case TypeIndex::kTVMFFIShape: {
        Array<int64_t> data = node["data"].cast<Array<int64_t>>();
        return ffi::Shape(data);
      }
      default: {
        return DecodeObjectData(type_index, node["data"]);
      }
    }
  }

  Array<Any> DecodeArrayData(const json::Array& data) {
    Array<Any> array;
    array.reserve(data.size());
    for (size_t i = 0; i < data.size(); i++) {
      array.push_back(GetOrDecodeNode(data[i].cast<int64_t>()));
    }
    return array;
  }

  Map<Any, Any> DecodeMapData(const json::Array& data) {
    Map<Any, Any> map;
    for (size_t i = 0; i < data.size(); i += 2) {
      int64_t key_index = data[i].cast<int64_t>();
      int64_t value_index = data[i + 1].cast<int64_t>();
      map.Set(GetOrDecodeNode(key_index), GetOrDecodeNode(value_index));
    }
    return map;
  }

  Any DecodeObjectData(int32_t type_index, const json::Value& data) {
    static reflection::TypeAttrColumn data_from_json =
        reflection::TypeAttrColumn("__data_from_json__");
    if (data_from_json[type_index] != nullptr) {
      return data_from_json[type_index].cast<Function>()(data);
    }
    // otherwise, we go over the fields and create the data.
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(type_index);
    if (type_info->metadata == nullptr || type_info->metadata->creator == nullptr) {
      TVM_FFI_THROW(RuntimeError) << "Type `" << TypeIndexToTypeKey(type_index)
                                  << "` does not support default constructor"
                                  << ", so ToJSONGraph is not supported for this type";
    }
    TVMFFIObjectHandle handle;
    TVM_FFI_CHECK_SAFE_CALL(type_info->metadata->creator(&handle));
    ObjectPtr<Object> ptr =
        details::ObjectUnsafe::ObjectPtrFromOwned<Object>(static_cast<TVMFFIObject*>(handle));

    auto decode_field_value = [&](const TVMFFIFieldInfo* field_info, json::Value data) -> Any {
      switch (field_info->field_static_type_index) {
        case TypeIndex::kTVMFFINone: {
          return nullptr;
        }
        case TypeIndex::kTVMFFIBool: {
          return data.cast<bool>();
        }
        case TypeIndex::kTVMFFIInt: {
          return data.cast<int64_t>();
        }
        case TypeIndex::kTVMFFIFloat: {
          return data.cast<double>();
        }
        case TypeIndex::kTVMFFIDataType: {
          return StringToDLDataType(data.cast<String>());
        }
        default: {
          return GetOrDecodeNode(data.cast<int64_t>());
        }
      }
    };

    json::Object data_object = data.cast<json::Object>();
    reflection::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* field_info) {
      String field_name(field_info->name);
      void* field_addr = reinterpret_cast<char*>(ptr.get()) + field_info->offset;
      if (data_object.count(field_name) != 0) {
        Any field_value = decode_field_value(field_info, data_object[field_name]);
        field_info->setter(field_addr, reinterpret_cast<const TVMFFIAny*>(&field_value));
      } else if (field_info->flags & kTVMFFIFieldFlagBitMaskHasDefault) {
        field_info->setter(field_addr, &(field_info->default_value));
      } else {
        TVM_FFI_THROW(TypeError) << "Required field `"
                                 << String(field_info->name.data, field_info->name.size)
                                 << "` not set in type `" << TypeIndexToTypeKey(type_index) << "`";
      }
    });
    return ObjectRef(ptr);
  }

  explicit ObjectGraphDeserializer(json::Value serialized) {
    if (!serialized.as<json::Object>()) {
      TVM_FFI_THROW(ValueError) << "Invalid JSON Object Graph, expected an object";
    }
    json::Object encoded_object = serialized.cast<json::Object>();
    if (encoded_object.count("root_index") == 0 || !encoded_object["root_index"].as<int64_t>()) {
      TVM_FFI_THROW(ValueError) << "Invalid JSON Object Graph, expected `root_index` integer field";
    }
    if (encoded_object.count("nodes") == 0 || !encoded_object["nodes"].as<json::Array>()) {
      TVM_FFI_THROW(ValueError) << "Invalid JSON Object Graph, expected `nodes` array field";
    }
    root_index_ = encoded_object["root_index"].cast<int64_t>();
    nodes_ = encoded_object["nodes"].cast<json::Array>();
    decoded_nodes_.resize(nodes_.size(), Any(nullptr));
  }
  // nodes
  json::Array nodes_;
  // root index
  int64_t root_index_;
  // null index if already created
  int64_t decoded_null_index_{-1};
  // decoded nodes
  std::vector<Any> decoded_nodes_;
};

Any FromJSONGraph(const json::Value& value) { return ObjectGraphDeserializer::Deserialize(value); }

// string version of the api
Any FromJSONGraphString(const String& value) { return FromJSONGraph(json::Parse(value)); }

String ToJSONGraphString(const Any& value, const Any& metadata) {
  return json::Stringify(ToJSONGraph(value, metadata));
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("ffi.ToJSONGraph", ToJSONGraph)
      .def("ffi.ToJSONGraphString", ToJSONGraphString)
      .def("ffi.FromJSONGraph", FromJSONGraph)
      .def("ffi.FromJSONGraphString", FromJSONGraphString);
  refl::EnsureTypeAttrColumn("__data_to_json__");
  refl::EnsureTypeAttrColumn("__data_from_json__");
});

}  // namespace ffi
}  // namespace tvm
