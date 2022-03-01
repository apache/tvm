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
#include <tvm/support/span.h>
#endif

// Version number recorded in emitted artifacts for runtime checking.
#define TVM_METADATA_VERSION 1

#ifdef __cplusplus
namespace tvm {
namespace runtime {
namespace metadata {
/*!
 * \brief Version of metadata emitted and understood by this compiler/runtime.
 * Should be populated into the `version` field of all TVMMetadata.
 */
static const constexpr int64_t kMetadataVersion = TVM_METADATA_VERSION;
}  // namespace metadata
}  // namespace runtime
}  // namespace tvm

extern "C" {
#endif

/*!
 * \brief Top-level metadata structure. Holds all other metadata types.
 */
struct TVMMetadata {
  /*! \brief Version identifier for this metadata. */
  int64_t version;
  /*! \brief Inputs to the AOT run_model function.
   * The order of the elements is the same as in the arguments to run_model. That is to say,
   * this array specifies the first `num_inputs` arguments to run_model.
   */
  const struct TVMTensorInfo* inputs;
  /*! \brief Number of elements in `inputs` array. */
  int64_t num_inputs;
  /*! \brief Outputs of the AOT run_model function.
   * The order of the elements is the same as in the arguments to run_model. That is to say,
   * this array specifies the last `num_outputs` arguments to run_model.
   */
  const struct TVMTensorInfo* outputs;
  /*! \brief Number of elements in `outputs` array. */
  int64_t num_outputs;
  /*! \brief Memory Pools needed by the AOT main function.
   * The order of the elements is the same as in the arguments to run_model. That is to say,
   * this array specifies the last `num_pools` arguments to run_model.
   */
  const struct TVMTensorInfo* pools;
  /*! \brief Number of elements in `pools` array. */
  int64_t num_pools;
  /*! \brief Name of the model, as passed to tvm.relay.build. */
  const char* mod_name;
};

/*!
 * \brief Describes one tensor argument to `run_model`.
 * NOTE: while TIR allows for other types of arguments, such as scalars, the AOT run_model
 * function does not currently accept these. Therefore it's not possible to express those
 * in this metadata. A future patch may modify this.
 */
struct TVMTensorInfo {
  /*! \brief Name of the tensor, as specified in the Relay program. */
  const char* name;
  /*! \brief Shape of the tensor. */
  const int64_t* shape;
  /*! \brief Rank of this tensor. */
  int64_t num_shape;
  /*! \brief Data type of one element of this tensor. */
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
  inline int64_t version() const { return int64_t(data_->version); }
  inline int64_t num_inputs() const { return data_->num_inputs; }
  ArrayAccessor<struct TVMTensorInfo, TensorInfo> inputs();
  inline int64_t num_outputs() const { return data_->num_outputs; }
  ArrayAccessor<struct TVMTensorInfo, TensorInfo> outputs();
  inline int64_t num_pools() const { return data_->num_pools; }
  ArrayAccessor<struct TVMTensorInfo, TensorInfo> pools();
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
