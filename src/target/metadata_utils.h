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
 * \file tvm/target/metadata_utils.h
 * \brief Declares utilty functions and classes for emitting metadata.
 */
#ifndef TVM_TARGET_METADATA_UTILS_H_
#define TVM_TARGET_METADATA_UTILS_H_

#include <tvm/runtime/data_type.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/object.h>

#include <string>
#include <tuple>
#include <vector>

#include "metadata.h"

namespace tvm {
namespace codegen {

std::string address_from_parts(const std::vector<std::string>& parts);
static constexpr const char* kMetadataGlobalSymbol = "kTvmgenMetadata";

class MetadataQueuer : public AttrVisitor {
 public:
  using QueueItem = std::tuple<std::string, runtime::metadata::MetadataBase>;
  explicit MetadataQueuer(std::vector<QueueItem>* queue);

  void Visit(const char* key, double* value) final;
  void Visit(const char* key, int64_t* value) final;
  void Visit(const char* key, uint64_t* value) final;
  void Visit(const char* key, int* value) final;
  void Visit(const char* key, bool* value) final;
  void Visit(const char* key, std::string* value) final;
  void Visit(const char* key, DataType* value) final;
  void Visit(const char* key, runtime::NDArray* value) final;
  void Visit(const char* key, void** value) final;

  void Visit(const char* key, ObjectRef* value) final;

 private:
  std::vector<QueueItem>* queue_;
  std::vector<std::string> address_parts_;
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_METADATA_UTILS_H_
