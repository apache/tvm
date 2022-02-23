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
  return ArrayAccessor<struct TVMTensorInfo, TensorInfo>(data_->inputs, data_->num_inputs);
}
ArrayAccessor<struct TVMTensorInfo, TensorInfo> MetadataNode::outputs() {
  return ArrayAccessor<struct TVMTensorInfo, TensorInfo>(data_->outputs, data_->num_outputs);
}
ArrayAccessor<struct TVMTensorInfo, TensorInfo> MetadataNode::pools() {
  return ArrayAccessor<struct TVMTensorInfo, TensorInfo>(data_->pools, data_->num_pools);
}

TVM_REGISTER_OBJECT_TYPE(MetadataBaseNode);

MetadataArray::MetadataArray(Array<ObjectRef> array, MetadataTypeIndex type_index,
                             const char* struct_name)
    : MetadataBase{make_object<MetadataArrayNode>(array, type_index, struct_name)} {}

TVM_REGISTER_OBJECT_TYPE(MetadataArrayNode);

Metadata::Metadata(const struct ::TVMMetadata* data)
    : MetadataBase{make_object<MetadataNode>(data)} {}
TVM_REGISTER_OBJECT_TYPE(MetadataNode);

TensorInfo::TensorInfo(const struct ::TVMTensorInfo* data)
    : MetadataBase{make_object<TensorInfoNode>(data)} {}
TVM_REGISTER_OBJECT_TYPE(TensorInfoNode);

}  // namespace metadata
}  // namespace runtime
}  // namespace tvm
