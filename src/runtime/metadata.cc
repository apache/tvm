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
 * \file metadata.cc
 * \brief Implementations of the runtime component of Metadata.
 */

#include <tvm/runtime/metadata.h>

namespace tvm {
namespace runtime {
namespace metadata {

ArrayAccessor<struct TVMTensorInfo, TensorInfo> MetadataNode::inputs() {
  if (inputs_refs_.get() == nullptr) { inputs_refs_.reset(new ::std::vector<TensorInfo>()); }
  return ArrayAccessor<struct TVMTensorInfo, TensorInfo>(data_->inputs, data_->num_inputs, inputs_refs_);
}
ArrayAccessor<struct TVMTensorInfo, TensorInfo> MetadataNode::outputs() {
  if (outputs_refs_.get() == nullptr) { outputs_refs_.reset(new ::std::vector<TensorInfo>()); }
  return ArrayAccessor<struct TVMTensorInfo, TensorInfo>(data_->outputs, data_->num_outputs, outputs_refs_);
}
ArrayAccessor<const char*, ::tvm::runtime::String> MetadataNode::devices() {
  if (devices_refs_.get() == nullptr) { devices_refs_.reset(new ::std::vector<::tvm::runtime::String>()); }
  return ArrayAccessor<const char*, ::tvm::runtime::String>(data_->devices, data_->num_devices, devices_refs_);
}
Metadata::Metadata(const struct ::TVMMetadata* data) :
    MetadataBase{make_object<MetadataNode>(data)} {}
std::string MetadataNode::get_name() { return std::string{"Metadata"}; }
TVM_REGISTER_OBJECT_TYPE(MetadataNode);
TensorInfo::TensorInfo(const struct ::TVMTensorInfo* data) :
    MetadataBase{make_object<TensorInfoNode>(data)} {}
std::string TensorInfoNode::get_name() { return std::string{"TensorInfo"}; }
TVM_REGISTER_OBJECT_TYPE(TensorInfoNode);

}  // namespace metadata
}  // namespace runtime
}  // namespace tvm
