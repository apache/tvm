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

#include <dmlc/memory_io.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/metadata_base.h>
#include <tvm/runtime/metadata_types.h>
#include <tvm/runtime/object.h>
#include <tvm/support/span.h>

#include <memory>
#include <string>
#include <vector>

// Version number recorded in emitted artifacts for runtime checking.
#define TVM_METADATA_VERSION 1

namespace tvm {
namespace runtime {
namespace metadata {
/*!
 * \brief Version of metadata emitted and understood by this compiler/runtime.
 * Should be populated into the `version` field of all TVMMetadata.
 */
static const constexpr int64_t kMetadataVersion = TVM_METADATA_VERSION;

class Metadata;
class TensorInfo;
class ConstantInfoMetadata;

class MetadataNode : public MetadataBaseNode {
 public:
  explicit MetadataNode(const struct ::TVMMetadata* data) : data_{data} {}
  static constexpr const char* _type_key = "metadata.MetadataNode";
  const char* get_c_struct_name() const override;
  inline int64_t version() const { return int64_t(data_->version); }
  inline int64_t num_inputs() const { return data_->num_inputs; }
  ArrayAccessor<struct TVMTensorInfo, TensorInfo> inputs();
  inline int64_t num_outputs() const { return data_->num_outputs; }
  ArrayAccessor<struct TVMTensorInfo, TensorInfo> outputs();
  inline int64_t num_workspace_pools() const { return data_->num_workspace_pools; }
  ArrayAccessor<struct TVMTensorInfo, TensorInfo> workspace_pools();
  inline ::tvm::runtime::String mod_name() const { return ::tvm::runtime::String(data_->mod_name); }
  const struct ::TVMMetadata* data() const { return data_; }
  ArrayAccessor<struct TVMConstantInfo, ConstantInfoMetadata> constant_pools();
  inline int64_t num_constant_pools() const { return data_->num_constant_pools; }
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
  const char* get_c_struct_name() const override;
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

class ConstantInfoMetadataNode : public MetadataBaseNode {
 public:
  explicit ConstantInfoMetadataNode(const struct ::TVMConstantInfo* data) : data_{data} {}
  // This name should match TVMConstantInfo after processing
  static constexpr const char* _type_key = "metadata.ConstantInfoNode";
  const char* get_c_struct_name() const override;
  inline ::tvm::runtime::String name_hint() const {
    return ::tvm::runtime::String(data_->name_hint);
  }
  inline size_t byte_offset() const { return data_->byte_offset; }
  inline ::tvm::runtime::NDArray data() const {
    ::tvm::runtime::NDArray ndarray;
    if (data_->data_len) {
      dmlc::MemoryFixedSizeStream bytes(const_cast<void*>(data_->data_bytes), data_->data_len);
      ndarray.Load(&bytes);
    }
    return ndarray;
  }
  TVM_DECLARE_FINAL_OBJECT_INFO(ConstantInfoMetadataNode, MetadataBaseNode);

 protected:
  const struct ::TVMConstantInfo* data_;
};

class ConstantInfoMetadata : public MetadataBase {
 public:
  explicit ConstantInfoMetadata(const struct ::TVMConstantInfo* data);
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ConstantInfoMetadata, MetadataBase,
                                        ConstantInfoMetadataNode);
};

}  // namespace metadata
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_METADATA_H_
