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
 * \file src/runtime/relax_vm/attention_kv_cache.cc
 * \brief A simple implementation of inplace attention kv cache for runtime.
 *
 * This file provides a simple implementation of inplace attention
 * kv cache for relax runtime. The main goal here is to help us enable
 * auto-regressive decoding quickly in relax.
 *
 * This is not the only way to support attention kv-cache.
 * Our support of attention kv-cache can subject to future
 * changes as we build more LM verticals.
 *
 * We will keep the impact minimum by puting it as a private
 * runtime builtin provide as in this file.
 *
 * We can evolve this implementation as we build more LM verticals.
 */

#include <tvm/runtime/container/shape_tuple.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/relax_vm/vm.h>

namespace tvm {
namespace runtime {
namespace relax_vm {

//-------------------------------------------
// We keep the implementation private as
// they may subject to future changes.
//
// Users can interact with it through the
// runtime API function calls
//-------------------------------------------
/*!
 * \brief An object representing an attention kv cache.
 */
class AttentionKVCacheObj : public Object {
 public:
  /*!
   * \brief Underlying support data.
   */
  NDArray data;

  /*!
   * \brief number of slots already filled.
   */
  int64_t fill_count{0};

  /*!
   * \brief View all current cached values as one array.
   * \param shape The cached values.
   */
  NDArray View(const ShapeTuple& shape) {
    CHECK_EQ(shape[0], fill_count) << "Requested shape do not match the filled count";
    for (int i = 1; i < this->data->ndim; ++i) {
      CHECK_EQ(shape[i], data->shape[i]) << "Dimension " << i << " mismatch";
    }
    return data.CreateView(shape, data->dtype);
  }

  /*!
   * \brief Append value to the cache.
   * \param value The value to be appended.
   */
  void Append(NDArray value) {
    CHECK(data.DataType() == value.DataType()) << "dtype mismatch";
    // reallocate cache
    int64_t reserved_slots = data->shape[0];
    while (fill_count + value->shape[0] > reserved_slots) {
      reserved_slots *= 2;
    }
    if (reserved_slots != data->shape[0]) {
      std::vector<int64_t> new_shape(data->shape, data->shape + data->ndim);
      new_shape[0] = reserved_slots;
      NDArray new_data = NDArray::Empty(new_shape, data->dtype, data->device);
      new_data.CreateView(data.Shape(), data->dtype).CopyFrom(data);
      this->data = new_data;
    }
    // copy into the fill count position.
    ICHECK_LE(fill_count + value->shape[0], data->shape[0]);
    ICHECK(data.IsContiguous());

    int64_t num_filled_elements = fill_count;
    for (int i = 1; i < data->ndim; ++i) {
      CHECK_EQ(value->shape[i], data->shape[i]) << "Dimension " << i << " mismatch";
      num_filled_elements *= data->shape[i];
    }
    // create a view of copy dest to copy the value into it.
    DLTensor copy_dst = *(data.operator->());
    copy_dst.byte_offset = num_filled_elements * ((data->dtype.bits * data->dtype.lanes + 7) / 8);
    copy_dst.shape = value->shape;
    NDArray::CopyFromTo(value.operator->(), &copy_dst);
    this->fill_count += value->shape[0];
  }

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "relax.vm.AttentionKVCache";
  TVM_DECLARE_FINAL_OBJECT_INFO(AttentionKVCacheObj, Object);
};

/*! \brief reference to closure. */
class AttentionKVCache : public ObjectRef {
 public:
  /*!
   * \brief Create the attention kv cache.
   * \param init_reserve The initial reserved.
   */
  static AttentionKVCache Create(NDArray init_data) {
    auto n = make_object<AttentionKVCacheObj>();
    n->data = std::move(init_data);
    n->fill_count = 0;
    return AttentionKVCache(n);
  }

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(AttentionKVCache, ObjectRef, AttentionKVCacheObj);
};

TVM_REGISTER_OBJECT_TYPE(AttentionKVCacheObj);

//-------------------------------------------------
//  Register runtime functions
//-------------------------------------------------
TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_create")
    .set_body_typed(AttentionKVCache::Create);

AttentionKVCache AttentionKVCacheAppend(AttentionKVCache cache, NDArray value) {
  cache->Append(value);
  return cache;
}

TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_append").set_body_typed(AttentionKVCacheAppend);

NDArray AttentionKVCacheView(AttentionKVCache cache, ShapeTuple shape) {
  return cache->View(shape);
}

TVM_REGISTER_GLOBAL("vm.builtin.attention_kv_cache_view").set_body_typed(AttentionKVCacheView);

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
