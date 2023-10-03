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
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <functional>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include "../file_utils.h"
#include "../relax_vm/ndarray_cache_support.h"
#include "./builtin.h"
#include "./utils.h"

namespace tvm {
namespace runtime {

using relax_vm::NDArrayCacheMetadata;
using FileRecord = NDArrayCacheMetadata::FileRecord;
using ParamRecord = NDArrayCacheMetadata::FileRecord::ParamRecord;
using relax_vm::ShardInfo;

/*! \brief An object that helps to load parameters in shards. */
class ShardLoaderObj : public Object {
 public:
  /*! \brief Create a shard loader. */
  static ObjectRef Create(const std::string& path_to_metadata, const std::string& metadata,
                          std::string shard_info, Module mod);
  /*! \brief Load the i-th parameter */
  NDArray Load(int weight_index) const;
  /*! \brief Load all the parameters */
  Array<NDArray> LoadAll() const;

  NDArray ApplyShardFunc(const ShardInfo::ShardFunc& shard_func, const NDArray& param) const;

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
  std::unordered_map<std::string, ShardInfo> shards = relax_vm::LoadShardInfoFromStr(shard_info);
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

NDArray ShardLoaderObj::Load(int weight_index) const {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  int worker_id = worker->worker_id;
  int num_shards = worker->num_workers;
  Device device = worker->default_device;
  const ParamInfo& param_info = param_info_.at(weight_index);
  const ParamRecord* param = param_info.param;
  const FileRecord* file = param_info.file;

  auto load = [this, param, device, file]() {
    if (file != current_file_) {
      current_file_ = file;
      std::string file_name = GetSiblingPath(this->metadata_.path, file->data_path);
      LoadBinaryFromFile(file_name, &this->current_file_stream_);
    }
    return param->Load(
        device, &this->current_file_stream_,
        [](NDArray param, const void* data, size_t nbytes) { param.CopyFromBytes(data, nbytes); });
  };

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
      NDArray w = load();
      for (const ShardInfo::ShardFunc& shard_func : param_info.shard_info.funcs) {
        w = this->ApplyShardFunc(shard_func, w);
      }
      ScatterFromWorker0(w, recv);
    } else {
      ScatterFromWorker0(NullOpt, recv);
    }
    return recv;
  } else {
    if (worker_id == 0) {
      NDArray w = load();
      BroadcastFromWorker0(w, w);
      return w;
    } else {
      NDArray w = NDArray::Empty(param->shape, param->dtype, device);
      BroadcastFromWorker0(w, w);
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

TVM_REGISTER_GLOBAL("runtime.disco.ShardLoader").set_body_typed(ShardLoaderObj::Create);
TVM_REGISTER_GLOBAL("runtime.disco.ShardLoaderLoad")
    .set_body_typed([](ObjectRef loader_obj, ShapeTuple weight_index) {
      const auto* loader = loader_obj.as<ShardLoaderObj>();
      CHECK(loader != nullptr) << "TypeError: Expected ShardLoaderObj, but gets: "
                               << loader_obj->GetTypeKey();
      return loader->Load(IntegerFromShapeTuple(weight_index));
    });

TVM_REGISTER_GLOBAL("runtime.disco.ShardLoaderLoadAll").set_body_typed([](ObjectRef loader_obj) {
  const auto* loader = loader_obj.as<ShardLoaderObj>();
  CHECK(loader != nullptr) << "TypeError: Expected ShardLoaderObj, but gets: "
                           << loader_obj->GetTypeKey();
  return loader->LoadAll();
});

}  // namespace runtime
}  // namespace tvm
