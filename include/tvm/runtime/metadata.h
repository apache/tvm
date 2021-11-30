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
 * \file tvm/runtime/metadata.h
 * \brief Defines types which can be used in Metadata.
 */
#ifndef TVM_RUNTIME_METADATA_H_
#define TVM_RUNTIME_METADATA_H_

#include <inttypes.h>
#ifdef __cplusplus
#include <memory>
#include <string>
#include <vector>
#endif
#include <tvm/runtime/c_runtime_api.h>
#ifdef __cplusplus
#include <tvm/runtime/metadata_base.h>
#endif
#include <tvm/support/span.h>

// Version number recorded in emitted artifacts for runtime checking.
#define TVM_METADATA_VERSION 1
static const constexpr int64_t kMetadataVersion = TVM_METADATA_VERSION;
#ifdef __cplusplus
extern "C" {
#endif

struct TVMMetadata {
  int64_t version;
  const struct TVMTensorInfo* inputs;
  int64_t num_inputs;
  const struct TVMTensorInfo* outputs;
  int64_t num_outputs;
  const char* mod_name;
};

struct TVMTensorInfo {
  const char* name;
  const int64_t* shape;
  int64_t num_shape;
  DLDataType dtype;
};
#ifdef __cplusplus
}  // extern "C"
#include <tvm/runtime/object.h>
namespace tvm {
namespace runtime {
namespace metadata {

class Metadata;
class TensorInfo;

class MetadataNode : public MetadataBaseNode {
 public:
  explicit MetadataNode(const struct ::TVMMetadata* data) : data_{data} {}
  static constexpr const char* _type_key = "metadata.MetadataNode";
  std::string get_name() override;
  inline int64_t version() const { return int64_t(data_->version); }
  inline int64_t num_inputs() const { return data_->num_inputs; }
  ArrayAccessor<struct TVMTensorInfo, TensorInfo> inputs();
  inline int64_t num_outputs() const { return data_->num_outputs; }
  ArrayAccessor<struct TVMTensorInfo, TensorInfo> outputs();
  inline ::tvm::runtime::String mod_name() const { return ::tvm::runtime::String(data_->mod_name); }
  const struct ::TVMMetadata* data() const { return data_; }
  TVM_DECLARE_FINAL_OBJECT_INFO(MetadataNode, MetadataBaseNode);

 private:
  const struct ::TVMMetadata* data_;
};

class Metadata : public MetadataBase {
 public:
  explicit Metadata(const struct ::TVMMetadata* data);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(Metadata, MetadataBase, MetadataNode);
};

class TensorInfoNode : public MetadataBaseNode {
 public:
  explicit TensorInfoNode(const struct ::TVMTensorInfo* data) : data_{data} {}
  static constexpr const char* _type_key = "metadata.TensorInfoNode";
  std::string get_name() override;
  inline ::tvm::runtime::String name() const { return ::tvm::runtime::String(data_->name); }
  inline int64_t num_shape() const { return data_->num_shape; }
  inline ::tvm::support::Span<const int64_t, int64_t> shape() const {
    return ::tvm::support::Span<const int64_t, int64_t>(data_->shape,
                                                        data_->shape + data_->num_shape);
  }
  inline ::tvm::runtime::DataType dtype() const { return ::tvm::runtime::DataType(data_->dtype); }
  const struct ::TVMTensorInfo* data() const { return data_; }
  TVM_DECLARE_FINAL_OBJECT_INFO(TensorInfoNode, MetadataBaseNode);

 private:
  const struct ::TVMTensorInfo* data_;
};

class TensorInfo : public MetadataBase {
 public:
  explicit TensorInfo(const struct ::TVMTensorInfo* data);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(TensorInfo, MetadataBase, TensorInfoNode);
};

}  // namespace metadata
}  // namespace runtime
}  // namespace tvm
#endif  // defined(__cplusplus)

#endif  // TVM_RUNTIME_METADATA_H_
