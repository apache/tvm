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

#include <filesystem>
#include <functional>
#include <numeric>
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

/*! \brief An object that helps to load parameters in shards. */
class ShardLoaderObj : public Object {
 public:
  /*!
   * \brief Create a shard loader
   * \param path_metadata The path to `ndarray-cache.json`
   * \param path_shard_info The path to `shard-info.json`
   * \param slice A method to slice a 3-D tensor at its middle dimension
   */
  static ObjectRef Create(const std::string& path_metadata, const std::string& path_shard_info,
                          TypedPackedFunc<void(DLTensor*, int, int, DLTensor*)> slice);
  /*! \brief Load the i-th parameter */
  NDArray Load(int weight_index) const;
  /*! \brief Slice the given tensor at a specific dimension */
  std::vector<NDArray> Shard(NDArray source, int dim, int num_slices) const;

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
  /*! \brief A method to slice a 3-D tensor at its middle dimension */
  TypedPackedFunc<void(DLTensor*, int, int, DLTensor*)> slice_;
  /*! \brief The current file opened to load weights in it */
  mutable const FileRecord* current_file_;
  /*! \brief The context of the current file to be loaded from */
  mutable std::string current_file_stream_;
};

TVM_REGISTER_OBJECT_TYPE(ShardLoaderObj);

ObjectRef ShardLoaderObj::Create(const std::string& path_metadata,
                                 const std::string& path_shard_info,
                                 TypedPackedFunc<void(DLTensor*, int, int, DLTensor*)> slice) {
  ObjectPtr<ShardLoaderObj> n = make_object<ShardLoaderObj>();
  n->slice_ = slice;
  n->metadata_ = NDArrayCacheMetadata::LoadFromFile(path_metadata);
  std::unordered_map<std::string, int> shard_info =
      relax_vm::LoadShardInfoFromFile(path_shard_info);
  for (const FileRecord& file_record : n->metadata_.records) {
    for (const ParamRecord& param_record : file_record.records) {
      int shard_id = -1;
      if (auto it = shard_info.find(param_record.name); it != shard_info.end()) {
        shard_id = it->second;
      }
      n->shard_info_.push_back(ShardInfo{&file_record, &param_record, shard_id});
    }
  }
  n->current_file_ = nullptr;
  return ObjectRef(std::move(n));
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
  NDArray weight{nullptr};
  if (shard_idx == 0) {
    if (file != current_file_) {
      current_file_ = file;
      LoadBinaryFromFile(file->data_path, &this->current_file_stream_);
    }
    Array<NDArray> weights =
        this->Shard(param->Load(device, &this->current_file_stream_,
                                [](NDArray param, const void* data, size_t nbytes) {
                                  param.CopyFromBytes(data, nbytes);
                                }),
                    shard_dim, num_shards);
    weight = weights[0];
    ScatterFromWorker0(Array<NDArray>{weights.begin() + 1, weights.end()});
  } else {
    weight = NDArray::Empty(ShardShape(param->shape, shard_dim, num_shards), param->dtype, device);
    RecvFromWorker0(weight);
  }
  return weight;
}

std::vector<NDArray> ShardLoaderObj::Shard(NDArray source, int dim, int num_slices) const {
  ShapeTuple src_shape = source.Shape();
  ShapeTuple dst_shape = ShardShape(src_shape, dim, num_slices);
  Device device = source->device;
  DLDataType dtype = source->dtype;
  int64_t src_flat[3] = {1, 1, 1};
  {
    const int64_t* s = src_shape.data();
    int ndim = source->ndim;
    src_flat[0] = std::accumulate(&s[0], &s[dim], 1, std::multiplies<int64_t>());
    src_flat[1] = s[dim];
    src_flat[2] = std::accumulate(&s[dim + 1], &s[ndim], 1, std::multiplies<int64_t>());
  }
  int64_t dst_flat[3] = {src_flat[0], src_flat[1] / num_slices, src_flat[2]};
  DLTensor src_tensor = *source.operator->();
  DLTensor dst_tensor = src_tensor;
  src_tensor.ndim = 3;
  src_tensor.shape = src_flat;
  dst_tensor.ndim = 3;
  dst_tensor.shape = dst_flat;
  std::vector<NDArray> results;
  results.reserve(num_slices);
  for (int i = 0; i < num_slices; ++i) {
    NDArray destination = NDArray::Empty(dst_shape, dtype, device);
    dst_tensor.data = destination->data;
    this->slice_(&src_tensor, dst_flat[1] * i, dst_flat[1], &dst_tensor);
    results.push_back(destination);
  }
  return results;
}

TVM_REGISTER_GLOBAL("runtime.disco.ShardLoader").set_body_typed(ShardLoaderObj::Create);
TVM_REGISTER_GLOBAL("runtime.disco.ShardLoaderLoad")
    .set_body_typed([](ObjectRef loader_obj, int weight_index) {
      const auto* loader = loader_obj.as<ShardLoaderObj>();
      CHECK(loader != nullptr) << "TypeError: Expected ShardLoaderObj, but gets: "
                               << loader_obj->GetTypeKey();
      return loader->Load(weight_index);
    });

}  // namespace runtime
}  // namespace tvm
