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
 * \file src/ffi/reflection/structural_equal.cc
 *
 * \brief Structural equal implementation.
 */
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/container/shape.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/reflection/accessor.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include <cmath>
#include <limits>
#include <unordered_map>
#include <utility>

namespace tvm {
namespace ffi {
/**
 * \brief Internal Handler class for structural hash.
 */
class StructuralHashHandler {
 public:
  StructuralHashHandler() = default;

  uint64_t HashAny(ffi::Any src) {
    using ffi::details::AnyUnsafe;
    const TVMFFIAny* src_data = AnyUnsafe::TVMFFIAnyPtrFromAny(src);

    if (src_data->type_index < TypeIndex::kTVMFFIStaticObjectBegin) {
      // specially handle nan for float, as there can be multiple representations of nan
      // make sure they map to the same hash value
      if (src_data->type_index == TypeIndex::kTVMFFIFloat && std::isnan(src_data->v_float64)) {
        TVMFFIAny temp = *src_data;
        temp.v_float64 = std::numeric_limits<double>::quiet_NaN();
        return details::StableHashCombine(temp.type_index, temp.v_uint64);
      }
      if (src_data->type_index == TypeIndex::kTVMFFISmallStr) {
        // for small string, we use the same type key hash as normal string
        // so heap allocated string and on stack string will have the same hash
        return details::StableHashCombine(TypeIndex::kTVMFFIStr,
                                          details::StableHashSmallStrBytes(src_data));
      }
      // this is POD data, we can just hash the value
      return details::StableHashCombine(src_data->type_index, src_data->v_uint64);
    }

    switch (src_data->type_index) {
      case TypeIndex::kTVMFFIStr:
      case TypeIndex::kTVMFFIBytes: {
        // return same hash as AnyHash
        const details::BytesObjBase* src_str =
            AnyUnsafe::CopyFromAnyViewAfterCheck<const details::BytesObjBase*>(src);
        return details::StableHashCombine(src_data->type_index,
                                          details::StableHashBytes(src_str->data, src_str->size));
      }
      case TypeIndex::kTVMFFIArray: {
        return HashArray(AnyUnsafe::MoveFromAnyAfterCheck<Array<Any>>(std::move(src)));
      }
      case TypeIndex::kTVMFFIMap: {
        return HashMap(AnyUnsafe::MoveFromAnyAfterCheck<Map<Any, Any>>(std::move(src)));
      }
      case TypeIndex::kTVMFFIShape: {
        return HashShape(AnyUnsafe::MoveFromAnyAfterCheck<Shape>(std::move(src)));
      }
      case TypeIndex::kTVMFFITensor: {
        return HashTensor(AnyUnsafe::MoveFromAnyAfterCheck<Tensor>(std::move(src)));
      }
      default: {
        return HashObject(AnyUnsafe::MoveFromAnyAfterCheck<ObjectRef>(std::move(src)));
      }
    }
  }

  uint64_t HashObject(ObjectRef obj) {
    // NOTE: invariant: lhs and rhs are already the same type
    const TVMFFITypeInfo* type_info = TVMFFIGetTypeInfo(obj->type_index());
    if (type_info->metadata == nullptr) {
      TVM_FFI_THROW(TypeError) << "Type metadata is not set for type `"
                               << String(type_info->type_key)
                               << "`, so StructuralHash is not supported for this type";
    }
    if (type_info->metadata->structural_eq_hash_kind == kTVMFFISEqHashKindUnsupported) {
      TVM_FFI_THROW(TypeError) << "_type_s_eq_hash_kind is not set for type `"
                               << String(type_info->type_key)
                               << "`, so StructuralHash is not supported for this type";
    }

    auto structural_eq_hash_kind = type_info->metadata->structural_eq_hash_kind;
    if (structural_eq_hash_kind == kTVMFFISEqHashKindUnsupported) {
      // Fallback to pointer hash
      return std::hash<const Object*>()(obj.get());
    }
    // return recored hash value if it is already computed
    auto it = hash_memo_.find(obj);
    if (it != hash_memo_.end()) {
      return it->second;
    }

    static reflection::TypeAttrColumn custom_s_hash = reflection::TypeAttrColumn("__s_hash__");

    // compute the hash value
    uint64_t hash_value = obj->GetTypeKeyHash();
    if (custom_s_hash[type_info->type_index] == nullptr) {
      // go over the content and hash the fields
      reflection::ForEachFieldInfo(type_info, [&](const TVMFFIFieldInfo* field_info) {
        // skip fields that are marked as structural eq hash ignore
        if (!(field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashIgnore)) {
          // get the field value from both side
          reflection::FieldGetter getter(field_info);
          Any field_value = getter(obj);
          // field is in def region, enable free var mapping
          if (field_info->flags & kTVMFFIFieldFlagBitMaskSEqHashDef) {
            bool allow_free_var = true;
            std::swap(allow_free_var, map_free_vars_);
            hash_value = details::StableHashCombine(hash_value, HashAny(field_value));
            std::swap(allow_free_var, map_free_vars_);
          } else {
            hash_value = details::StableHashCombine(hash_value, HashAny(field_value));
          }
        }
      });
    } else {
      if (s_hash_callback_ == nullptr) {
        s_hash_callback_ =
            ffi::Function::FromTyped([this](AnyView val, uint64_t init_hash, bool def_region) {
              if (def_region) {
                bool allow_free_var = true;
                std::swap(allow_free_var, map_free_vars_);
                uint64_t hash_value = HashAny(val);
                std::swap(allow_free_var, map_free_vars_);
                return details::StableHashCombine(init_hash, hash_value);
              } else {
                return details::StableHashCombine(init_hash, HashAny(val));
              }
            });
      }
      hash_value = custom_s_hash[type_info->type_index]
                       .cast<ffi::Function>()(obj, hash_value, s_hash_callback_)
                       .cast<uint64_t>();
    }

    if (structural_eq_hash_kind == kTVMFFISEqHashKindFreeVar) {
      if (map_free_vars_) {
        // use lexical order of free var and its type
        hash_value = details::StableHashCombine(hash_value, free_var_counter_++);
      } else {
        // Fallback to pointer hash, we are not mapping free var.
        hash_value = std::hash<const Object*>()(obj.get());
      }
    }
    // if it is a DAG node, also record the lexical order of graph counter
    // this helps to distinguish DAG from trees.
    if (structural_eq_hash_kind == kTVMFFISEqHashKindDAGNode) {
      hash_value = details::StableHashCombine(hash_value, graph_node_counter_++);
    }
    // record the hash value for this object
    hash_memo_[obj] = hash_value;
    return hash_value;
  }

  uint64_t HashArray(Array<Any> arr) {
    uint64_t hash_value = details::StableHashCombine(arr->GetTypeKeyHash(), arr.size());
    for (size_t i = 0; i < arr.size(); ++i) {
      hash_value = details::StableHashCombine(hash_value, HashAny(arr[i]));
    }
    return hash_value;
  }

  // Find an order independent hash value for a given Any.
  // Order independent hash value means the hash value will remain stable independent
  // of the order we hash the content at the current context.
  // This property is needed to support stable hash for map.
  std::optional<uint64_t> FindOrderIndependentHash(Any src) {
    using ffi::details::AnyUnsafe;
    const TVMFFIAny* src_data = AnyUnsafe::TVMFFIAnyPtrFromAny(src);

    if (src_data->type_index < TypeIndex::kTVMFFIStaticObjectBegin) {
      if (src_data->type_index == TypeIndex::kTVMFFISmallStr) {
        // for small string, we use the same type key hash as normal string
        // so heap allocated string and on stack string will have the same hash
        return details::StableHashCombine(
            TypeIndex::kTVMFFIStr,
            details::StableHashBytes(src_data->v_bytes, src_data->small_str_len));
      }
      // this is POD data, we can just hash the value
      return details::StableHashCombine(src_data->type_index, src_data->v_uint64);
    } else {
      if (src_data->type_index == TypeIndex::kTVMFFIStr ||
          src_data->type_index == TypeIndex::kTVMFFIBytes) {
        const details::BytesObjBase* src_str =
            AnyUnsafe::CopyFromAnyViewAfterCheck<const details::BytesObjBase*>(src);
        // return same hash as AnyHash
        return details::StableHashCombine(src_data->type_index,
                                          details::StableHashBytes(src_str->data, src_str->size));
      } else {
        // if the hash of the object is already computed, return it
        auto it = hash_memo_.find(src.cast<ObjectRef>());
        if (it != hash_memo_.end()) {
          return it->second;
        }
        return std::nullopt;
      }
    }
  }

  uint64_t HashMap(Map<Any, Any> map) {
    // Compute a deterministic hash value for the map.
    uint64_t hash_value = details::StableHashCombine(map->GetTypeKeyHash(), map.size());
    std::vector<std::pair<uint64_t, Any>> items;
    for (auto [key, value] : map) {
      // if we cannot find order independent hash, we skip the key
      if (auto hash_key = FindOrderIndependentHash(key)) {
        items.emplace_back(*hash_key, value);
      }
    }
    // sort the items by the hash key, so the hash value is deterministic
    // and independent of the order of insertion
    std::sort(items.begin(), items.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    for (size_t i = 0; i < items.size();) {
      size_t k = i + 1;
      for (; k < items.size() && items[k].first == items[i].first; ++k) {
      }
      // detect ties, which are rare, but we need to skip value hash during ties
      // to make sure that the hash value is deterministic.
      if (k == i + 1) {
        // no ties, we just hash the key and value
        hash_value = details::StableHashCombine(hash_value, items[i].first);
        hash_value = details::StableHashCombine(hash_value, HashAny(items[i].second));
      } else {
        // ties occur, we skip the value hash to make sure that the hash value is deterministic.
        hash_value = details::StableHashCombine(hash_value, items[i].first);
      }
      i = k;
    }
    return hash_value;
  }

  uint64_t HashShape(Shape shape) {
    uint64_t hash_value = details::StableHashCombine(shape->GetTypeKeyHash(), shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
      hash_value = details::StableHashCombine(hash_value, shape[i]);
    }
    return hash_value;
  }

  uint64_t HashTensor(Tensor tensor) {
    uint64_t hash_value = details::StableHashCombine(tensor->GetTypeKeyHash(), tensor->ndim);
    for (int i = 0; i < tensor->ndim; ++i) {
      hash_value = details::StableHashCombine(hash_value, tensor->shape[i]);
    }
    TVMFFIAny temp;
    temp.v_uint64 = 0;
    temp.v_dtype = tensor->dtype;
    hash_value = details::StableHashCombine(hash_value, temp.v_int64);

    if (!skip_tensor_content_) {
      TVM_FFI_ICHECK_EQ(tensor->device.device_type, kDLCPU) << "can only hash CPU tensor";
      TVM_FFI_ICHECK(tensor.IsContiguous()) << "Can only hash contiguous tensor";
      size_t data_size = GetDataSize(*(tensor.operator->()));
      uint64_t data_hash =
          details::StableHashBytes(static_cast<const char*>(tensor->data), data_size);
      hash_value = details::StableHashCombine(hash_value, data_hash);
    }
    return hash_value;
  }

  bool map_free_vars_{false};
  bool skip_tensor_content_{false};
  // free var counter.
  uint32_t free_var_counter_{0};
  // graph node counter.
  uint32_t graph_node_counter_{0};
  // lazily initialize custom hash function
  ffi::Function s_hash_callback_ = nullptr;
  // map from lhs to rhs
  std::unordered_map<ObjectRef, uint64_t, ObjectPtrHash, ObjectPtrEqual> hash_memo_;
};

uint64_t StructuralHash::Hash(const Any& value, bool map_free_vars, bool skip_tensor_content) {
  StructuralHashHandler handler;
  handler.map_free_vars_ = map_free_vars;
  handler.skip_tensor_content_ = skip_tensor_content;
  return handler.HashAny(value);
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("ffi.StructuralHash", StructuralHash::Hash);
  refl::EnsureTypeAttrColumn("__s_hash__");
});

}  // namespace ffi
}  // namespace tvm
