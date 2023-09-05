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
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

#include <functional>
#include <numeric>
#include <string>
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
using relax_vm::LoadShardInfoFromStr;

/*! \brief An object that helps to load parameters in shards. */
class ShardLoaderObj : public Object {
 public:
  /*! \brief Create a shard loader. */
  static ObjectRef Create(const std::string& path_to_metadata, const std::string& metadata,
                          const std::string& shard_info,
                          TypedPackedFunc<void(DLTensor*, int, DLTensor*)> f_shard);
  /*! \brief Load the i-th parameter */
  NDArray Load(int weight_index) const;
  /*! \brief Load the i-th parameter from presharded binaries */
  NDArray LoadPresharded(int weight_index) const;
  /*! \brief Slice the given tensor at a specific dimension */
  NDArray Shard(NDArray source, int dim, int num_slices) const;

  static constexpr const char* _type_key = "runtime.disco.ShardLoader";
  TVM_DECLARE_FINAL_OBJECT_INFO(ShardLoaderObj, Object);

 public:
  /*! \brief Information of how each weight is stored and sharded */
  struct ShardInfo {
    const FileRecord* file;
    const ParamRecord* param;
    int shard_dim;
  };
  /*! \brief The metadata loaded from `ndarray-cache.json` */
  NDArrayCacheMetadata metadata_;
  /*! \brief Sharding information for each weight */
  std::vector<ShardInfo> shard_info_;
  /*! \brief A method to slice a 3-D tensor */
  TypedPackedFunc<void(DLTensor*, int, DLTensor*)> f_shard_;
  /*! \brief The current file opened to load weights in it */
  mutable const FileRecord* current_file_;
  /*! \brief The context of the current file to be loaded from */
  mutable std::string current_file_stream_;
};

TVM_REGISTER_OBJECT_TYPE(ShardLoaderObj);

/*!
 * \brief Get the shape of a result tensor if it is scattered along a given axis.
 * \param shape The shape of the input tensor.
 * \param dim The axis along which the tensor is scattered.
 * \param num_shards The number of shards.
 * \return The shape of the result tensor.
 */
inline std::vector<ShapeTuple::index_type> ShardShape(const ShapeTuple& shape, int dim,
                                                      int num_shards) {
  CHECK(0 <= dim && dim < static_cast<int>(shape.size()))
      << "ValueError: Cannot scatter at dim " << dim << ", because "
      << "shape is " << shape << ".";
  CHECK_EQ(shape[dim] % num_shards, 0)
      << "ValueError: The shape " << shape << " cannot be scattered at dim " << dim << " into "
      << num_shards << " shards.";
  std::vector<ShapeTupleObj::index_type> result{shape.begin(), shape.end()};
  result[dim] /= num_shards;
  return result;
}

ObjectRef ShardLoaderObj::Create(const std::string& path_to_metadata, const std::string& metadata,
                                 const std::string& shard_info,
                                 TypedPackedFunc<void(DLTensor*, int, DLTensor*)> f_shard) {
  ObjectPtr<ShardLoaderObj> n = make_object<ShardLoaderObj>();
  n->f_shard_ = f_shard;
  n->metadata_ = NDArrayCacheMetadata::LoadFromStr(metadata, path_to_metadata);
  n->current_file_ = nullptr;
  n->shard_info_.clear();
  std::unordered_map<std::string, int> shards = LoadShardInfoFromStr(shard_info);
  for (const FileRecord& file_record : n->metadata_.records) {
    for (const ParamRecord& param_record : file_record.records) {
      const std::string& name = param_record.name;
      int shard_id = shards.count(name) ? shards[name] : -1;
      n->shard_info_.push_back(ShardInfo{&file_record, &param_record, shard_id});
    }
  }
  return ObjectRef(std::move(n));
}

std::string GetSiblingPath(const std::string& path, const std::string& filename) {
  size_t found = path.find_last_of("/\\");
  if (found != std::string::npos) {
    return path.substr(0, found + 1) + filename;
  }
  LOG(FATAL) << "ValueError: Cannot find the parent directory: " << path;
}

std::tuple<int, int> ParseParamShardingInfo(const ParamRecord* param) {
  std::string name = param->name;
  size_t pos1 = name.rfind('_');
  if (pos1 == std::string::npos) {
    LOG(FATAL) << "Attempt to read num_shards from unexpected param name: " << name;
  }
  size_t pos2 = name.rfind('_', pos1 - 1);
  if (pos2 == std::string::npos) {
    LOG(FATAL) << "Attempt to read sharded worker_id from unexpected param name: " << name;
  }

  int num_shards = std::stoi(name.substr(pos1 + 1));
  int worker_id = std::stoi(name.substr(pos2 + 1, pos1 - pos2 - 1)) - 1;

  return {num_shards, worker_id};
}

NDArray ShardLoaderObj::Load(int weight_index) const {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  int shard_idx = worker->worker_id;
  Device device = worker->default_device;
  const auto& shard_info = shard_info_[weight_index];
  const ParamRecord* param = shard_info.param;
  const FileRecord* file = shard_info.file;
  int shard_dim = shard_info.shard_dim;
  int num_shards = worker->num_workers;
  Optional<NDArray> send = NullOpt;
  if (shard_idx == 0) {
    if (file != current_file_) {
      current_file_ = file;
      std::string file_name = GetSiblingPath(this->metadata_.path, file->data_path);
      LoadBinaryFromFile(file_name, &this->current_file_stream_);
    }
    auto f_load = [](NDArray param, const void* data, size_t nbytes) {
      param.CopyFromBytes(data, nbytes);
    };
    send = this->Shard(param->Load(device, &this->current_file_stream_, f_load), shard_dim,
                       num_shards);
  }
  NDArray recv =
      NDArray::Empty(ShardShape(param->shape, shard_dim, num_shards), param->dtype, device);
  ScatterFromWorker0(send, recv);
  return recv;
}

NDArray ShardLoaderObj::Shard(NDArray source, int dim, int num_slices) const {
  ICHECK(dim >= 0 && dim < source->ndim);
  // Assemble a flattened 3d src tensor
  int64_t src_flat[3] = {1, 1, 1};
  {
    const int64_t* s = source.Shape().data();
    int ndim = source->ndim;
    src_flat[0] = std::accumulate(&s[0], &s[dim], 1, std::multiplies<int64_t>());
    src_flat[1] = s[dim];
    src_flat[2] = std::accumulate(&s[dim + 1], &s[ndim], 1, std::multiplies<int64_t>());
  }
  DLTensor src_tensor = *source.operator->();
  src_tensor.ndim = 3;
  src_tensor.shape = src_flat;
  // Assmeble a flattened 4d dst tensor
  int64_t dst_flat[4] = {num_slices, src_flat[0], src_flat[1] / num_slices, src_flat[2]};
  NDArray destination{nullptr};
  {
    std::vector<ShapeTuple::index_type> dst_shape = ShardShape(source.Shape(), dim, num_slices);
    dst_shape.insert(dst_shape.begin(), static_cast<ShapeTuple::index_type>(num_slices));
    destination = NDArray::Empty(dst_shape, source->dtype, source->device);
  }
  DLTensor dst_tensor = *destination.operator->();
  dst_tensor.ndim = 4;
  dst_tensor.shape = dst_flat;
  // Copy slices using the API
  this->f_shard_(&src_tensor, num_slices, &dst_tensor);
  return destination;
}

NDArray ShardLoaderObj::LoadPresharded(int weight_index) const {
  DiscoWorker* worker = DiscoWorker::ThreadLocal();
  int worker_id = worker->worker_id;
  int num_shards = worker->num_workers;
  Device device = worker->default_device;
  size_t index = weight_index * num_shards + worker_id;
  CHECK(index < shard_info_.size())
      << "Loading param " << weight_index << " for shard " << worker_id << " at position " << index
      << " is out of bounds for the provided ndarray chace.";

  const auto& shard_info = shard_info_[index];
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

  std::string file_name = GetSiblingPath(this->metadata_.path, file->data_path);
  LoadBinaryFromFile(file_name, &this->current_file_stream_);

  auto f_load = [](NDArray param, const void* data, size_t nbytes) {
    param.CopyFromBytes(data, nbytes);
  };
  return param->Load(device, &this->current_file_stream_, f_load);
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

}  // namespace runtime
}  // namespace tvm
