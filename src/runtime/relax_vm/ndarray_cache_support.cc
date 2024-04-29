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
#include <picojson.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/ndarray_cache_support.h>

#include <string>
#include <vector>

#include "../../support/utils.h"
#include "../file_utils.h"

namespace tvm {
namespace runtime {
namespace relax_vm {

template <typename ExpectedType>
inline ExpectedType AsType(const picojson::value& json) {
  ICHECK(json.is<ExpectedType>());
  return json.get<ExpectedType>();
}

template <typename ValueType>
inline ValueType GetValue(const picojson::object& json, const std::string& key) {
  return AsType<ValueType>(json.at(key));
}

NDArrayCacheMetadata::FileRecord::ParamRecord JSONAsParamRecord(const picojson::object& json) {
  std::vector<ShapeTuple::index_type> shape;
  {
    picojson::array shape_json = GetValue<picojson::array>(json, "shape");
    shape.reserve(shape_json.size());
    for (const picojson::value& d : shape_json) {
      shape.push_back(AsType<int64_t>(d));
    }
  }
  NDArrayCacheMetadata::FileRecord::ParamRecord result;
  std::string dtype = GetValue<std::string>(json, "dtype");
  result.name = GetValue<std::string>(json, "name");
  result.dtype = DataType(String2DLDataType(dtype));
  result.format = GetValue<std::string>(json, "format");
  result.nbytes = GetValue<int64_t>(json, "nbytes");
  result.byte_offset = GetValue<int64_t>(json, "byteOffset");
  result.shape = ShapeTuple(std::move(shape));
  return result;
}

NDArrayCacheMetadata::FileRecord JSONAsFileRecord(const picojson::object& json) {
  picojson::array records = GetValue<picojson::array>(json, "records");
  NDArrayCacheMetadata::FileRecord result;
  result.data_path = GetValue<std::string>(json, "dataPath");
  result.format = GetValue<std::string>(json, "format");
  result.nbytes = GetValue<int64_t>(json, "nbytes");
  result.records.reserve(records.size());
  for (const picojson::value& item : records) {
    result.records.push_back(JSONAsParamRecord(AsType<picojson::object>(item)));
  }
  return result;
}

NDArrayCacheMetadata JSONAsNDArrayCacheMetadata(const picojson::object& json) {
  picojson::array records = GetValue<picojson::array>(json, "records");
  NDArrayCacheMetadata result;
  result.records.reserve(records.size());
  for (const picojson::value& item : records) {
    result.records.push_back(JSONAsFileRecord(AsType<picojson::object>(item)));
  }
  return result;
}

NDArrayCacheMetadata NDArrayCacheMetadata::LoadFromStr(const std::string& json_str,
                                                       const std::string& path) {
  picojson::value json_info;
  {
    std::string err = picojson::parse(json_info, json_str);
    if (!err.empty()) {
      LOG(FATAL) << "Failed to parse JSON: err. The JSON string is:" << json_str;
    }
    CHECK(json_info.is<picojson::object>())
        << "ValueError: The given string is not a JSON object: " << json_str;
  }
  NDArrayCacheMetadata result = JSONAsNDArrayCacheMetadata(AsType<picojson::object>(json_info));
  result.path = path;
  return result;
}

TVM_DLL NDArrayCacheMetadata NDArrayCacheMetadata::Load(const std::string& path) {
  picojson::value json_info;
  {
    std::string json_str;
    LoadBinaryFromFile(path + "/ndarray-cache.json", &json_str);
    std::string err = picojson::parse(json_info, json_str);
    if (!err.empty()) {
      LOG(FATAL) << "Failed to parse JSON: err. The JSON string is:" << json_str;
    }
    CHECK(json_info.is<picojson::object>())
        << "ValueError: The given string is not a JSON object: " << json_str;
  }
  NDArrayCacheMetadata result = JSONAsNDArrayCacheMetadata(AsType<picojson::object>(json_info));
  result.path = path;
  return result;
}

void CopyNDArrayFromBytes(NDArray param, const void* data, size_t nbytes,
                          Optional<NDArray>* staging_buffer) {
  Device device = param->device;
  if (device.device_type != kDLOpenCL || staging_buffer == nullptr) {
    param.CopyFromBytes(data, nbytes);
    return;
  }
  // Special handle for OpenCL runtime.
  // It creates a host side memory mirror, for every cl_mem that tries to copy data from host
  // which can cause memory issue. Her we use a large staging buffer to postpone deallocation
  if (staging_buffer->defined()) {
    size_t curr_size = runtime::GetDataSize(*(staging_buffer->value().operator->()));
    if (curr_size < nbytes) {
      *staging_buffer = NullOpt;
    }
  }
  if (!staging_buffer->defined()) {
    *staging_buffer = NDArray::Empty(param.Shape(), param->dtype, param->device);
  }
  NDArray staging_view = staging_buffer->value().CreateView(param.Shape(), param->dtype);
  staging_view.CopyFromBytes(data, nbytes);
  param.CopyFrom(staging_view);
  TVMSynchronize(device.device_type, device.device_id, nullptr);
}

NDArray NDArrayCacheMetadata::FileRecord::ParamRecord::Load(
    Device device, const std::string* raw_data, Optional<NDArray>* staging_buffer) const {
  NDArray arr = NDArray::Empty(shape, dtype, device);
  if (dtype == DataType::Float(32) && format == "f32-to-bf16") {
    // decode bf16 to f32
    std::vector<uint16_t> buffer(nbytes / 2);
    std::vector<uint32_t> decoded(nbytes / 2);
    std::memcpy(buffer.data(), raw_data->data() + byte_offset, nbytes);
    for (size_t i = 0; i < buffer.size(); ++i) {
      decoded[i] = static_cast<uint32_t>(buffer[i]) << 16;
    }
    CopyNDArrayFromBytes(arr, decoded.data(), decoded.size() * sizeof(uint32_t), staging_buffer);
  } else {
    CopyNDArrayFromBytes(arr, raw_data->data() + byte_offset, nbytes, staging_buffer);
  }
  return arr;
}

TVM_DLL Array<NDArray> NDArrayCacheMetadata::FileRecord::Load(
    Device device,
    const std::string& path_prefix,  //
    std::string* raw_data_buffer,    //
    Optional<NDArray>* staging_buffer) const {
  LoadBinaryFromFile(path_prefix + "/" + this->data_path, raw_data_buffer);
  CHECK_EQ(this->format, "raw-shard") << "ValueError: Only `raw-shard` format is supported";
  CHECK_EQ(this->nbytes, raw_data_buffer->length())
      << "ValueError: Encountered an corrupted parameter shard. It means it is not downloaded "
         "completely or downloading is interrupted. Please try to download again.";
  Array<NDArray> result;
  result.reserve(this->records.size());
  for (const ParamRecord& nd_rec : this->records) {
    result.push_back(nd_rec.Load(device, raw_data_buffer, staging_buffer));
  }
  return result;
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
   * \param cache_path The cache to path.
   * \param device_type The type of device to be loaded.
   * \param device_id The device id.
   */
  static void Load(const std::string& cache_path, int device_type, int device_id) {
    DLDevice device{static_cast<DLDeviceType>(device_type), device_id};
    NDArrayCacheMetadata metadata = NDArrayCacheMetadata::Load(cache_path);
    Optional<NDArray> staging_buffer;
    std::string raw_data;
    Array<NDArray> params;
    for (const NDArrayCacheMetadata::FileRecord& shard_rec : metadata.records) {
      try {
        params = shard_rec.Load(device, cache_path, &raw_data, &staging_buffer);
      } catch (const dmlc::Error& e) {
        LOG(FATAL) << "ValueError: Error when loading parameters from " << shard_rec.data_path
                   << ": " << e.what();
      }
      int num_params = params.size();
      for (int i = 0; i < num_params; ++i) {
        Update(shard_rec.records[i].name, params[i], true);
      }
    }
  }

 private:
  Map<String, NDArray> pool_;
};

TVM_REGISTER_GLOBAL("vm.builtin.ndarray_cache.get").set_body_typed(NDArrayCache::Get);
TVM_REGISTER_GLOBAL("vm.builtin.ndarray_cache.update").set_body([](TVMArgs args, TVMRetValue* rv) {
  CHECK(args.size() == 2 || args.size() == 3);
  String name = args[0];
  bool is_override = args.size() == 2 ? false : args[2];

  NDArray arr;
  if (args[1].type_code() == kTVMNDArrayHandle) {
    arr = args[1];
  } else {
    // We support converting DLTensors to NDArrays as RPC references are always DLTensors
    DLTensor* tensor = args[1];
    std::vector<int64_t> shape;
    for (int64_t i = 0; i < tensor->ndim; i++) {
      shape.push_back(tensor->shape[i]);
    }
    arr = NDArray::Empty(shape, tensor->dtype, tensor->device);
    arr.CopyFrom(tensor);
    TVMSynchronize(arr->device.device_type, arr->device.device_id, nullptr);
  }

  NDArrayCache::Update(name, arr, is_override);
});
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

  static Array<NDArray> GetParamByName(const Array<String>& names) {
    Array<NDArray> result;
    result.reserve(names.size());
    for (const String& name : names) {
      if (Optional<NDArray> opt = NDArrayCache::Get(name)) {
        result.push_back(opt.value());
      } else {
        LOG(FATAL) << "ValueError: Cannot find parameter in cache: " << name;
      }
    }
    return result;
  }

  static Module Create(const std::string& prefix, int num_params) {
    auto n = make_object<ParamModuleNode>();
    n->params_ = GetParams(prefix, num_params);
    return Module(n);
  }

  static Module CreateByName(const Array<String>& names) {
    auto n = make_object<ParamModuleNode>();
    n->params_ = GetParamByName(names);
    return Module(n);
  }

 private:
  Array<NDArray> params_;
};

TVM_REGISTER_GLOBAL("vm.builtin.param_module_from_cache").set_body_typed(ParamModuleNode::Create);
TVM_REGISTER_GLOBAL("vm.builtin.param_module_from_cache_by_name")
    .set_body_typed(ParamModuleNode::CreateByName);
TVM_REGISTER_GLOBAL("vm.builtin.param_array_from_cache").set_body_typed(ParamModuleNode::GetParams);
TVM_REGISTER_GLOBAL("vm.builtin.param_array_from_cache_by_name")
    .set_body_typed(ParamModuleNode::GetParamByName);
TVM_REGISTER_GLOBAL("vm.builtin.param_array_from_cache_by_name_unpacked")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      Array<String> names;
      names.reserve(args.size());
      for (int i = 0; i < args.size(); ++i) {
        if (args[i].type_code() != kTVMStr) {
          LOG(FATAL) << "ValueError: Expect string as input, but get " << args[i].type_code()
                     << " at " << i;
        }
        names.push_back(args[i]);
      }
      *rv = ParamModuleNode::GetParamByName(names);
    });

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm
