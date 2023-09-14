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
 * \file src/runtime/relax_vm/ndarray_cache_support.cc
 * \brief Runtime to support ndarray cache file loading.
 *
 * This file provides a minimum support for ndarray cache file loading.
 *
 * The main focus of this implementation is to enable loading
 * with minimum set of intermediate files while also being
 * compatible to some of the multi-shard files that are more
 * friendly in some of the environments.
 *
 * NDArray cache also provides a way to do system-wide
 * parameter sharing across multiple VMs.
 *
 * There are likely other ways to load the parameters ndarray-ache.
 * We will keep the impact minimum by puting it as a private
 * runtime builtin provide as in this file.
 */
#define PICOJSON_USE_INT64
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include "./ndarray_cache_support.h"

#include <picojson.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <vector>

#include "../../support/utils.h"
#include "../file_utils.h"

namespace tvm {
namespace runtime {
namespace relax_vm {

NDArrayCacheMetadata::FileRecord::ParamRecord JSONAsParamRecord(const picojson::object& json) {
  std::vector<ShapeTuple::index_type> shape;
  {
    picojson::array shape_json = json.at("shape").get<picojson::array>();
    shape.reserve(shape_json.size());
    for (const picojson::value& d : shape_json) {
      shape.push_back(d.get<int64_t>());
    }
  }
  NDArrayCacheMetadata::FileRecord::ParamRecord result;
  result.name = json.at("name").get<std::string>();
  result.dtype = DataType(String2DLDataType(json.at("dtype").get<std::string>()));
  result.format = json.at("format").get<std::string>();
  result.nbytes = json.at("nbytes").get<int64_t>();
  result.byte_offset = json.at("byteOffset").get<int64_t>();
  result.shape = ShapeTuple(std::move(shape));
  return result;
}

NDArrayCacheMetadata::FileRecord JSONAsFileRecord(const picojson::object& json) {
  picojson::array records = json.at("records").get<picojson::array>();
  NDArrayCacheMetadata::FileRecord result;
  result.data_path = json.at("dataPath").get<std::string>();
  result.format = json.at("format").get<std::string>();
  result.nbytes = json.at("nbytes").get<int64_t>();
  result.records.reserve(records.size());
  for (const picojson::value& item : records) {
    result.records.push_back(JSONAsParamRecord(item.get<picojson::object>()));
  }
  return result;
}

NDArrayCacheMetadata JSONAsNDArrayCacheMetadata(const picojson::object& json) {
  picojson::array records = json.at("records").get<picojson::array>();
  NDArrayCacheMetadata result;
  result.records.reserve(records.size());
  for (const picojson::value& item : records) {
    result.records.push_back(JSONAsFileRecord(item.get<picojson::object>()));
  }
  return result;
}

NDArrayCacheMetadata NDArrayCacheMetadata::LoadFromStr(const std::string& json_str,
                                                       const std::string& path) {
  picojson::value json_info;
  picojson::parse(json_info, json_str);
  NDArrayCacheMetadata result = JSONAsNDArrayCacheMetadata(json_info.get<picojson::object>());
  result.path = path;
  return result;
}

std::unordered_map<std::string, int> LoadShardInfoFromStr(const std::string& json_str) {
  picojson::value json_info;
  picojson::parse(json_info, json_str);
  picojson::object json_obj = json_info.get<picojson::object>();
  std::unordered_map<std::string, int> result;
  for (const auto& kv : json_obj) {
    std::string name = kv.first;
    int64_t shard_dim = kv.second.get<int64_t>();
    result[name] = shard_dim;
  }
  return result;
}

NDArray NDArrayCacheMetadata::FileRecord::ParamRecord::Load(
    Device device, const std::string* raw_data,
    std::function<void(NDArray, const void*, int64_t)> f_load) const {
  NDArray arr = NDArray::Empty(shape, dtype, device);
  if (dtype == DataType::Float(32) && format == "f32-to-bf16") {
    // decode bf16 to f32
    std::vector<uint16_t> buffer(nbytes / 2);
    std::vector<uint32_t> decoded(nbytes / 2);
    std::memcpy(buffer.data(), raw_data->data() + byte_offset, nbytes);
    for (size_t i = 0; i < buffer.size(); ++i) {
      decoded[i] = static_cast<uint32_t>(buffer[i]) << 16;
    }
    f_load(arr, decoded.data(), decoded.size() * sizeof(uint32_t));
  } else {
    f_load(arr, raw_data->data() + byte_offset, nbytes);
  }
  return arr;
}

/*!
 * A NDArray cache to store pre-loaded arrays in the system.
 */
class NDArrayCache {
 public:
  static NDArrayCache* Global() {
    static NDArrayCache* inst = new NDArrayCache();
    return inst;
  }

  static void Update(String name, NDArray arr, bool override) {
    NDArrayCache* pool = Global();
    if (!override) {
      ICHECK_EQ(pool->pool_.count(name), 0) << "Name " << name << " already exists in the cache";
    }
    pool->pool_.Set(name, arr);
  }

  static Optional<NDArray> Get(String name) {
    NDArrayCache* pool = Global();
    auto it = pool->pool_.find(name);
    if (it != pool->pool_.end()) {
      return (*it).second;
    } else {
      return NullOpt;
    }
  }

  static void Remove(String name) {
    NDArrayCache* pool = Global();
    pool->pool_.erase(name);
  }

  static void Clear() { Global()->pool_.clear(); }

  /*!
   * \brief Load parameters from path and append them.
   *
   * \param cache_path The cache to path.
   * \param device_type The type of device to be loaded.
   * \param device_id The device id.
   */
  static void Load(const std::string& cache_path, int device_type, int device_id) {
    DLDevice device{static_cast<DLDeviceType>(device_type), device_id};
    std::string json_str;
    LoadBinaryFromFile(cache_path + "/ndarray-cache.json", &json_str);
    NDArrayCacheMetadata metadata = NDArrayCacheMetadata::LoadFromStr(json_str, cache_path);
    Optional<NDArray> staging_buffer;
    auto fcopy_param_from_bytes = [&](NDArray param, const void* data, size_t nbytes) {
      if (device_type != kDLOpenCL) {
        param.CopyFromBytes(data, nbytes);
        return;
      }
      // special handle OpenCL
      // OpenCL runtime can create a host side memory mirror
      // for every cl_mem that tries to copy data from host
      // which can cause memory issue.
      // We use a single staging buffer here
      // that get de-allocated later
      if (staging_buffer.defined()) {
        size_t curr_size = runtime::GetDataSize(*(staging_buffer.value().operator->()));
        if (curr_size < nbytes) {
          staging_buffer = NullOpt;
        }
      }
      if (!staging_buffer.defined()) {
        staging_buffer = NDArray::Empty(param.Shape(), param->dtype, param->device);
      }
      NDArray staging_view = staging_buffer.value().CreateView(param.Shape(), param->dtype);
      staging_view.CopyFromBytes(data, nbytes);
      param.CopyFrom(staging_view);
      TVMSynchronize(device_type, device_id, nullptr);
    };

    Map<String, NDArray> result;
    std::string raw_data;
    for (const auto& shard_rec : metadata.records) {
      LoadBinaryFromFile(cache_path + "/" + shard_rec.data_path, &raw_data);
      CHECK_EQ(shard_rec.format, "raw-shard") << "ValueError: Only `raw-shard` format is supported";
      CHECK_EQ(shard_rec.nbytes, raw_data.length())
          << "ValueError: Parameters are not loaded properly. Please check your parameter shards "
             "and git lfs installation";
      for (const auto& nd_rec : shard_rec.records) {
        Update(nd_rec.name, nd_rec.Load(device, &raw_data, fcopy_param_from_bytes), true);
      }
    }
  }

 private:
  Map<String, NDArray> pool_;
};

TVM_REGISTER_GLOBAL("vm.builtin.ndarray_cache.get").set_body_typed(NDArrayCache::Get);
TVM_REGISTER_GLOBAL("vm.builtin.ndarray_cache.update").set_body_typed(NDArrayCache::Update);
TVM_REGISTER_GLOBAL("vm.builtin.ndarray_cache.remove").set_body_typed(NDArrayCache::Remove);
TVM_REGISTER_GLOBAL("vm.builtin.ndarray_cache.clear").set_body_typed(NDArrayCache::Clear);
TVM_REGISTER_GLOBAL("vm.builtin.ndarray_cache.load").set_body_typed(NDArrayCache::Load);

// This param module node can be useful to get param dict in RPC mode
// when the remote already have loaded parameters from file.
class ParamModuleNode : public runtime::ModuleNode {
 public:
  const char* type_key() const final { return "param_module"; }

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) final {
    if (name == "get_params") {
      auto params = params_;
      return PackedFunc([params](TVMArgs args, TVMRetValue* rv) { *rv = params; });
    } else {
      return PackedFunc();
    }
  }

  static Array<NDArray> GetParams(const String& prefix, int num_params) {
    Array<NDArray> params;
    for (int i = 0; i < num_params || num_params == -1; ++i) {
      std::string name = prefix + "_" + std::to_string(i);
      auto opt = NDArrayCache::Get(name);
      if (opt) {
        params.push_back(opt.value());
      } else {
        if (num_params == -1) return params;
        LOG(FATAL) << "Cannot find " << name << " in cache";
      }
    }
    return params;
  }

  static Module Create(const std::string& prefix, int num_params) {
    auto n = make_object<ParamModuleNode>();
    n->params_ = GetParams(prefix, num_params);
    return Module(n);
  }

 private:
  Array<NDArray> params_;
};

TVM_REGISTER_GLOBAL("vm.builtin.param_module_from_cache").set_body_typed(ParamModuleNode::Create);
TVM_REGISTER_GLOBAL("vm.builtin.param_array_from_cache").set_body_typed(ParamModuleNode::GetParams);

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
