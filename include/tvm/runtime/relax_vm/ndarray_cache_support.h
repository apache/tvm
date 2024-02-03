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
#ifndef TVM_RUNTIME_RELAX_VM_NDARRAY_CACHE_SUPPORT_H_
#define TVM_RUNTIME_RELAX_VM_NDARRAY_CACHE_SUPPORT_H_

#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <string>
#include <unordered_map>
#include <vector>

namespace tvm {
namespace runtime {
namespace relax_vm {

/*!
 * \brief Metadata for NDArray cache, which by default, is named as "ndarray-cache.json".
 */
struct NDArrayCacheMetadata {
  /*! \brief Each shard of NDArray cache, which by default, is named as "params_shard_x.bin". */
  struct FileRecord {
    /*! \brief Metadata of each parameter */
    struct ParamRecord {
      /*!
       * \brief Load the parameter from raw data.
       * \param device The device to load the parameter onto.
       * \param raw_data The raw data stream
       * \param staging_buffer The buffer to be used to avoid extra OpenCL copies. Pass in a nullptr
       * in other cases
       */
      NDArray Load(Device device, const std::string* raw_data,
                   Optional<NDArray>* staging_buffer = nullptr) const;

      /*! \brief Name of the parameter */
      std::string name;
      /*! \brief Shape of the parameter */
      ShapeTuple shape;
      /*! \brief Data type of the parameter */
      DataType dtype;
      /*! \brief Format of the parameter */
      std::string format;
      /*! \brief Number of bytes */
      int64_t nbytes;
      /*! \brief Offset from the raw stream */
      int64_t byte_offset;
    };

    /*! \brief Load a FileRecord into memory */
    TVM_DLL Array<NDArray> Load(Device device,                   //
                                const std::string& path_prefix,  //
                                std::string* raw_data_buffer,    //
                                Optional<NDArray>* staging_buffer = nullptr) const;

    /*! \brief Relative path to the bin file */
    std::string data_path;
    /*! \brief Format of the file */
    std::string format;
    /*! \brief Size of the file */
    int64_t nbytes;
    /*! \brief The parameters in the file */
    std::vector<ParamRecord> records;
  };
  /*! \brief The files in the NDArray cache */
  std::vector<FileRecord> records;
  /*! \brief The path to the `ndarray-cache.json` file */
  std::string path;

  /*! \brief Load the metadata from a specific directory */
  TVM_DLL static NDArrayCacheMetadata Load(const std::string& path);
  /*! \brief Load the metadata from a given JSON string */
  static NDArrayCacheMetadata LoadFromStr(const std::string& json_str, const std::string& path);
};

}  // namespace relax_vm
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_RELAX_VM_NDARRAY_CACHE_SUPPORT_H_
