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
 * \file meta_data.h
 * \brief Meta data related utilities
 */
#ifndef TVM_RUNTIME_META_DATA_H_
#define TVM_RUNTIME_META_DATA_H_

#include <dmlc/io.h>
#include <dmlc/json.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "runtime_base.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Structure used by the AOT to fill the tvm_module_t structure
 */
class AOTMetadataNode : public Object {
 public:
  /*! \brief number of inputs of the main function */
  int num_inputs = 1;
  /*! \brief number of outputs of the main function */
  int num_outputs = 1;

  static constexpr const uint32_t _type_index = TypeIndex::kDynamic;
  static constexpr const char* _type_key = "AOTMetadataObj";
  TVM_DECLARE_FINAL_OBJECT_INFO(AOTMetadataNode, Object);
};

/*!
 * \brief Managed reference to AOTMetadataNode.
 */
class AOTMetadata : public ObjectRef {
 public:
  TVM_DLL AOTMetadata(int num_inputs, int num_outputs) {
    auto n = make_object<AOTMetadataNode>();
    n->num_inputs = num_inputs;
    n->num_outputs = num_outputs;
    data_ = std::move(n);
  }

  TVM_DEFINE_OBJECT_REF_METHODS(AOTMetadata, ObjectRef, AOTMetadataNode);
  TVM_DEFINE_OBJECT_REF_COW_METHOD(AOTMetadataNode);
};

/*!
 * \brief Create a metadata module object.
 *
 * \param metadata The variable name to ndarray mapping.
 * \param sym_vars The symbol to the list of required constant variables
 * mapping.
 *
 * \return The created metadata module.
 */
Module MetadataModuleCreate(
    const std::unordered_map<std::string, NDArray>& metadata,
    const std::unordered_map<std::string, std::vector<std::string>>& sym_vars);

/*! \brief function information needed by device */
struct FunctionInfo {
  std::string name;
  std::vector<DLDataType> arg_types;
  std::vector<std::string> thread_axis_tags;

  void Save(dmlc::JSONWriter* writer) const;
  void Load(dmlc::JSONReader* reader);
  void Save(dmlc::Stream* writer) const;
  bool Load(dmlc::Stream* reader);
};
}  // namespace runtime
}  // namespace tvm

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, ::tvm::runtime::FunctionInfo, true);
}  // namespace dmlc
#endif  // TVM_RUNTIME_META_DATA_H_
