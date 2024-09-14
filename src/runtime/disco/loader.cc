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
#define PICOJSON_USE_INT64
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif
#include <picojson.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/disco/builtin.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/ndarray_cache_support.h>

#include <functional>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "../file_utils.h"
#include "./utils.h"

namespace tvm {
namespace runtime {

using relax_vm::NDArrayCacheMetadata;
using FileRecord = NDArrayCacheMetadata::FileRecord;
using ParamRecord = NDArrayCacheMetadata::FileRecord::ParamRecord;

struct ShardInfo {
  struct TensorInfo {
    ShapeTuple shape;
    DataType dtype;
  };
  struct ShardFunc {
    std::string name;
    TensorInfo output_info;
    std::vector<int64_t> params;
  };
  std::vector<ShardFunc> funcs;
};

template <typename ExpectedType>
inline ExpectedType AsType(const picojson::value& json) {
  ICHECK(json.is<ExpectedType>());
  return json.get<ExpectedType>();
}

template <typename ValueType>
inline ValueType GetValue(const picojson::object& json, const std::string& key) {
  return AsType<ValueType>(json.at(key));
}

std::unordered_map<std::string, ShardInfo> LoadShardInfoFromStr(const std::string& json_str);
ShardInfo::TensorInfo LoadTensorInfoFromJSON(const picojson::array& json_tensor_info) {
  CHECK_EQ(json_tensor_info.size(), 2) << "ValueError: Invalid tensor info JSON";
  picojson::array shape_json = AsType<picojson::array>(json_tensor_info[0]);
  int ndim = shape_json.size();
  std::vector<int64_t> shape;
  shape.reserve(ndim);
  for (int i = 0; i < ndim; ++i) {
    shape.push_back(AsType<int64_t>(shape_json[i]));
  }
  std::string dtype = AsType<std::string>(json_tensor_info[1]);
  return ShardInfo::TensorInfo{ShapeTuple(std::move(shape)), DataType(String2DLDataType(dtype))};
}

ShardInfo::ShardFunc LoadShardFuncFromJSON(const picojson::array& json_shard_func) {
  int n = json_shard_func.size();
  ShardInfo::ShardFunc shard_info;
  shard_info.name = AsType<std::string>(json_shard_func[0]);
  shard_info.output_info = LoadTensorInfoFromJSON(AsType<picojson::array>(json_shard_func[1]));
  shard_info.params.reserve(n - 2);
  for (int i = 2; i < n; ++i) {
    shard_info.params.push_back(AsType<int64_t>(json_shard_func[i]));
  }
  return shard_info;
}

std::unordered_map<std::string, ShardInfo> LoadShardInfoFromStr(const std::string& json_str) {
  picojson::value json_info;
  picojson::parse(json_info, json_str);
  picojson::object json_obj = AsType<picojson::object>(json_info);
  std::unordered_map<std::string, ShardInfo> result;
  for (auto kv : json_obj) {
    std::string name = kv.first;
    picojson::array json_shard_funcs = AsType<picojson::array>(kv.second);
    ShardInfo info;
    std::vector<ShardInfo::ShardFunc>& shard_funcs = info.funcs;
    shard_funcs.reserve(json_shard_funcs.size());
    for (const picojson::value& json_shard_func : json_shard_funcs) {
      shard_funcs.push_back(LoadShardFuncFromJSON(AsType<picojson::array>(json_shard_func)));
    }
    result[name] = info;
  }
  return result;
}

/*! \brief An object that helps to load parameters in shards. */
class ShardLoaderObj : public Object {
 public:
  /*! \brief Create a shard loader. */
  static ObjectRef Create(const std::string& path_to_metadata, const std::string& metadata,
                          std::string shard_info, Module mod);
  /*! \brief Load the i-th parameter */
  NDArray Load(int weight_index) const;

  NDArray LoadParamOnWorker0(int weight_index) const;

  /*! \brief Load all the parameters */
  Array<NDArray> LoadAll() const;

  NDArray ApplyShardFunc(const ShardInfo::ShardFunc& shard_func, const NDArray& param) const;

  /*! \brief Load all the pre-sharded parameters */
  Array<NDArray> LoadAllPresharded() const;

  /*! \brief Load the i-th parameter from presharded binaries */
  NDArray LoadPresharded(int weight_index) const;

  /*! \brief Slice the given tensor at a specific dimension */
  NDArray Shard(NDArray source, int dim, int num_slices) const;

  static constexpr const char* _type_key = "runtime.disco.ShardLoader";
  TVM_DECLARE_FINAL_OBJECT_INFO(ShardLoaderObj, Object);

 public:
  /*! \brief Information of how each weight is stored and sharded */
  struct ParamInfo {
    const FileRecord* file;
    const ParamRecord* param;
    ShardInfo shard_info;
  };
  /*! \brief The PackedFuncs being used during sharding */
  std::unordered_map<std::string, PackedFunc> shard_funcs_;
  /*! \brief The metadata loaded from `ndarray-cache.json` */
  NDArrayCacheMetadata metadata_;
  /*! \brief Sharding information for each weight */
  std::vector<ParamInfo> param_info_;
  /*! \brief Maps the name of a shard to its index */
  std::unordered_map<std::string, int> param_name_to_index_;
  /*! \brief The current file opened to load weights in it */
  mutable const FileRecord* current_file_;
  /*! \brief The context of the current file to be loaded from */
  mutable std::string current_file_stream_;

 private:
  /*! \brief Load the i-th parameter without post-processing
   *
   * This function should not be called externally, as it does not
   * check for post-processing that may be required.  Instead, the
   * public function `Load` or `LoadPresharded` should be called.
   *
   * \param weight_index The index of NDArray tensor to load
   *
   * \returns The full tensor at the specified index
   */
  NDArray LoadDirect(int weight_index) const;
};

TVM_REGISTER_OBJECT_TYPE(ShardLoaderObj);

ObjectRef ShardLoaderObj::Create(const std::string& path_to_metadata, const std::string& metadata,
                                 std::string shard_info, Module mod) {
  if (shard_info.empty() && mod.defined()) {
    if (PackedFunc get_shard_info = mod->GetFunction("get_shard_info"); get_shard_info != nullptr) {
      shard_info = get_shard_info().operator String();
    }
  }
  ObjectPtr<ShardLoaderObj> n = make_object<ShardLoaderObj>();
  n->metadata_ = NDArrayCacheMetadata::LoadFromStr(metadata, path_to_metadata);
  n->current_file_ = nullptr;
  n->param_info_.clear();
  std::unordered_map<std::string, ShardInfo> shards = LoadShardInfoFromStr(shard_info);
  for (const FileRecord& file_record : n->metadata_.records) {
    for (const ParamRecord& param_record : file_record.records) {
      const std::string& name = param_record.name;
      int index = n->param_info_.size();
      n->param_name_to_index_[name] = index;
      ShardInfo& shard_info = shards[name];
      for (const ShardInfo::ShardFunc& shard_func : shard_info.funcs) {
        const std::string& name = shard_func.name;
        if (PackedFunc f = mod.defined() ? mod->GetFunction(name, true) : nullptr; f != nullptr) {
          n->shard_funcs_[name] = f;
        } else if (const PackedFunc* f = runtime::Registry::Get(name)) {
          n->shard_funcs_[name] = *f;
        } else {
          LOG(FATAL) << "ValueError: Undefined function: " << name;
        }
      }
      n->param_info_.emplace_back(ParamInfo{&file_record, &param_record, shard_info});
    }
  }
  return ObjectRef(std::move(n));
}

NDArray ShardLoaderObj::ApplyShardFunc(const ShardInfo::ShardFunc& shard_func,
                                       const NDArray& param) const {
  Device device = param->device;
  NDArray o = NDArray::Empty(shard_func.output_info.shape, shard_func.output_info.dtype, device);
  PackedFunc f = this->shard_funcs_.at(shard_func.name);
  int n = static_cast<int>(shard_func.params.size());
  std::vector<TVMValue> tvm_args(n + 2);
  std::vector<int> type_codes(n + 2);
  TVMArgsSetter setter(tvm_args.data(), type_codes.data());
  const DLTensor* w_in = param.operator->();
  const DLTensor* w_out = o.operator->();
  setter(0, const_cast<DLTensor*>(w_in));
  for (int i = 0; i < n; ++i) {
    setter(i + 1, shard_func.params[i]);
  }
  setter(n + 1, const_cast<DLTensor*>(w_out));
  TVMRetValue rv;
  f.CallPacked(TVMArgs(tvm_args.data(), type_codes.data(), n + 2), &rv);
  return o;
}

std::string GetSiblingPath(const std::string& path, const std::string& filename) {
  size_t found = path.find_last_of("/\\");
  if (found != std::string::npos) {
    return path.substr(0, found + 1) + filename;
  }
  LOG(FATAL) << "ValueError: Cannot find the parent directory: " << path;
}

NDArray ShardLoaderObj::LoadParamOnWorker0(int weight_index) const {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  int worker_id = worker->worker_id;
  Device device = worker->default_device;
  int param_index = param_name_to_index_.at("param_" + std::to_string(weight_index));
  const ParamInfo& param_info = param_info_.at(param_index);
  const ParamRecord* param = param_info.param;
  const FileRecord* file = param_info.file;

  auto load = [this, param, device, file]() {
    if (file != current_file_) {
      current_file_ = file;
      std::string file_name = GetSiblingPath(this->metadata_.path, file->data_path);
      LoadBinaryFromFile(file_name, &this->current_file_stream_);
    }
    return param->Load(device, &this->current_file_stream_);
  };

  if (worker_id == 0) {
    NDArray w = load();
    return w;
  } else {
    NDArray w = NDArray::Empty(param->shape, param->dtype, device);
    return w;
  }
}

std::tuple<int, int> ParseParamShardingInfo(const ParamRecord* param) {
  // Given a name "param_shard-X-of-Y", return the integer values
  // rank=(X-1) and world_size=Y.

  std::string name = param->name;
  size_t pos1 = name.rfind("-of-");
  CHECK(pos1 != std::string::npos)
      << "Attempt to read num_shards from unexpected param name: " << name;
  size_t pos2 = name.rfind("_shard-", pos1 - 1);
  CHECK(pos2 != std::string::npos)
      << "Attempt to read sharded worker_id from unexpected param name: " << name;

  int num_shards = std::stoi(name.substr(pos1 + 4));
  int worker_id = std::stoi(name.substr(pos2 + 7, pos1 - pos2 - 7)) - 1;

  CHECK_GT(num_shards, 1);
  CHECK_GE(worker_id, 0);
  CHECK_LT(worker_id, num_shards);

  return {num_shards, worker_id};
}

NDArray ShardLoaderObj::LoadDirect(int weight_index) const {
  const ParamInfo& param_info = param_info_.at(weight_index);
  const ParamRecord* param = param_info.param;
  const FileRecord* file = param_info.file;

  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  Device device = worker->default_device;

  if (file != current_file_) {
    current_file_ = file;
    std::string file_name = GetSiblingPath(this->metadata_.path, file->data_path);
    LoadBinaryFromFile(file_name, &this->current_file_stream_);
  }
  return param->Load(device, &this->current_file_stream_);
}

NDArray ShardLoaderObj::Load(int weight_index) const {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  int worker_id = worker->worker_id;
  int num_shards = worker->num_workers;
  Device device = worker->default_device;
  const ParamInfo& param_info = param_info_.at(weight_index);
  const ParamRecord* param = param_info.param;

  bool needs_sharding = !param_info.shard_info.funcs.empty();
  if (needs_sharding) {
    ShapeTuple shape = param_info.shard_info.funcs.back().output_info.shape;
    DataType dtype = param_info.shard_info.funcs.back().output_info.dtype;
    ICHECK(shape.size() >= 1 && shape[0] == num_shards)
        << "ValueError: The first dimension of the "
        << "output shape must be equal to the "
        << "number of shards, but got: " << shape << " and num_shards = " << num_shards;
    NDArray recv = NDArray::Empty(ShapeTuple(shape.begin() + 1, shape.end()), dtype, device);
    if (worker_id == 0) {
      NDArray w = LoadDirect(weight_index);
      for (const ShardInfo::ShardFunc& shard_func : param_info.shard_info.funcs) {
        w = this->ApplyShardFunc(shard_func, w);
      }
      ScatterFromWorker0(w, /*in_group=*/false, recv);
    } else {
      ScatterFromWorker0(NullOpt, /*in_group=*/false, recv);
    }
    return recv;
  } else {
    if (worker_id == 0) {
      NDArray w = LoadDirect(weight_index);
      BroadcastFromWorker0(w, /*in_group=*/false, w);
      return w;
    } else {
      NDArray w = NDArray::Empty(param->shape, param->dtype, device);
      BroadcastFromWorker0(w, /*in_group=*/false, w);
      return w;
    }
  }
}

Array<NDArray> ShardLoaderObj::LoadAll() const {
  int n = static_cast<int>(param_info_.size());
  Array<NDArray> shards;
  shards.reserve(n);
  for (int i = 0; i < n; ++i) {
    std::string param_name = "param_" + std::to_string(i);
    ICHECK(this->param_name_to_index_.count(param_name));
    int shard_id = this->param_name_to_index_.at(param_name);
    shards.push_back(this->Load(shard_id));
  }
  return shards;
}

NDArray ShardLoaderObj::LoadPresharded(int weight_index) const {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  int worker_id = worker->worker_id;
  int num_shards = worker->num_workers;
  size_t num_weights = param_info_.size() / num_shards;
  size_t index = worker_id * num_weights + weight_index;
  CHECK(index < param_info_.size())
      << "Loading param " << weight_index << " for shard " << worker_id << " at position " << index
      << " is out of bounds for the provided ndarray chace.";

  const auto& shard_info = param_info_[index];
  const ParamRecord* param = shard_info.param;
  const FileRecord* file = shard_info.file;

  auto [p_num_shards, p_worker_id] = ParseParamShardingInfo(param);
  CHECK_EQ(num_shards, p_num_shards)
      << "Runtime number of shards (" << num_shards
      << ") does not match number of compiled shards (" << p_num_shards << "): " << param->name
      << " loaded from " << file->data_path;
  CHECK_EQ(worker_id, p_worker_id)
      << "Runtime worker_id (" << worker_id << ") does not match worker_id of compiled shard ("
      << p_worker_id << "): " << param->name << " loaded from " << file->data_path;

  return LoadDirect(index);
}

Array<NDArray> ShardLoaderObj::LoadAllPresharded() const {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  size_t worker_id = static_cast<size_t>(worker->worker_id);
  size_t num_workers = static_cast<size_t>(worker->num_workers);
  size_t num_params = param_info_.size() / num_workers;

  Array<NDArray> params;
  params.reserve(num_params);
  for (size_t i_param = 0; i_param < num_params; ++i_param) {
    std::string param_name = static_cast<const std::stringstream&>(
                                 std::stringstream() << "param_" << i_param << "_shard-"
                                                     << (worker_id + 1) << "-of-" << num_workers)
                                 .str();

    auto it = param_name_to_index_.find(param_name);
    CHECK(it != param_name_to_index_.end())
        << "Parameter " << param_name << " was not found in the parameter set";
    int param_id = this->param_name_to_index_.at(param_name);
    params.push_back(this->LoadDirect(param_id));
  }
  return params;
}

TVM_REGISTER_GLOBAL("runtime.disco.ShardLoader").set_body_typed(ShardLoaderObj::Create);
TVM_REGISTER_GLOBAL("runtime.disco.ShardLoaderLoad")
    .set_body_typed([](ObjectRef loader_obj, ShapeTuple weight_index) {
      const auto* loader = loader_obj.as<ShardLoaderObj>();
      CHECK(loader != nullptr) << "TypeError: Expected ShardLoaderObj, but gets: "
                               << loader_obj->GetTypeKey();
      return loader->Load(IntegerFromShapeTuple(weight_index));
    });
TVM_REGISTER_GLOBAL("runtime.disco.ShardLoaderLoadPresharded")
    .set_body_typed([](ObjectRef loader_obj, ShapeTuple weight_index) {
      const auto* loader = loader_obj.as<ShardLoaderObj>();
      CHECK(loader != nullptr) << "TypeError: Expected ShardLoaderObj, but gets: "
                               << loader_obj->GetTypeKey();
      return loader->LoadPresharded(IntegerFromShapeTuple(weight_index));
    });

TVM_REGISTER_GLOBAL("runtime.disco.ShardLoaderLoadAll").set_body_typed([](ObjectRef loader_obj) {
  const auto* loader = loader_obj.as<ShardLoaderObj>();
  CHECK(loader != nullptr) << "TypeError: Expected ShardLoaderObj, but gets: "
                           << loader_obj->GetTypeKey();
  return loader->LoadAll();
});

TVM_REGISTER_GLOBAL("runtime.disco.ShardLoaderLoadAllPresharded")
    .set_body_typed([](ObjectRef loader_obj) {
      const auto* loader = loader_obj.as<ShardLoaderObj>();
      CHECK(loader != nullptr) << "TypeError: Expected ShardLoaderObj, but gets: "
                               << loader_obj->GetTypeKey();
      return loader->LoadAllPresharded();
    });

TVM_REGISTER_GLOBAL("runtime.disco.ShardLoaderLoadParamOnWorker0")
    .set_body_typed([](ObjectRef loader_obj, int param_index) {
      const auto* loader = loader_obj.as<ShardLoaderObj>();
      CHECK(loader != nullptr) << "TypeError: Expected ShardLoaderObj, but gets: "
                               << loader_obj->GetTypeKey();
      return loader->LoadParamOnWorker0(param_index);
    });

}  // namespace runtime
}  // namespace tvm
