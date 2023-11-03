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
 * \file src/runtime/metadata.cc
 * \brief Defines implementations of TVM metadata which can exist in the runtime.
 */

#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/metadata.h>
#include <tvm/runtime/registry.h>

#include <string>

namespace tvm {
namespace runtime {
namespace metadata {

TVM_REGISTER_OBJECT_TYPE(MetadataBaseNode);

ArrayAccessor<struct TVMTensorInfo, TensorInfo> MetadataNode::inputs() {
  return ArrayAccessor<struct TVMTensorInfo, TensorInfo>(data_->inputs, data_->num_inputs);
}
ArrayAccessor<struct TVMTensorInfo, TensorInfo> MetadataNode::outputs() {
  return ArrayAccessor<struct TVMTensorInfo, TensorInfo>(data_->outputs, data_->num_outputs);
}
ArrayAccessor<struct TVMTensorInfo, TensorInfo> MetadataNode::workspace_pools() {
  return ArrayAccessor<struct TVMTensorInfo, TensorInfo>(data_->workspace_pools,
                                                         data_->num_workspace_pools);
}
ArrayAccessor<struct TVMConstantInfo, ConstantInfoMetadata> MetadataNode::constant_pools() {
  return ArrayAccessor<struct TVMConstantInfo, ConstantInfoMetadata>(data_->constant_pools,
                                                                     data_->num_constant_pools);
}

TVM_REGISTER_OBJECT_TYPE(MetadataBaseNode);

MetadataArray::MetadataArray(Array<ObjectRef> array, MetadataKind kind, const char* struct_name)
    : MetadataBase{make_object<MetadataArrayNode>(array, kind, struct_name)} {}

const char* MetadataArrayNode::get_c_struct_name() const {
  ICHECK(false) << "MetadataArrayNode get_c_struct_name is unimplemented";
  return nullptr;
}
TVM_REGISTER_OBJECT_TYPE(MetadataArrayNode);

Metadata::Metadata(const struct ::TVMMetadata* data)
    : MetadataBase{make_object<MetadataNode>(data)} {}
TVM_REGISTER_OBJECT_TYPE(MetadataNode);

const char* MetadataNode::get_c_struct_name() const { return "TVMMetadata"; }

TensorInfo::TensorInfo(const struct ::TVMTensorInfo* data)
    : MetadataBase{make_object<TensorInfoNode>(data)} {}
TVM_REGISTER_OBJECT_TYPE(TensorInfoNode);

const char* TensorInfoNode::get_c_struct_name() const { return "TVMTensorInfo"; }

ConstantInfoMetadata::ConstantInfoMetadata(const struct ::TVMConstantInfo* data)
    : MetadataBase{make_object<ConstantInfoMetadataNode>(data)} {}
TVM_REGISTER_OBJECT_TYPE(ConstantInfoMetadataNode);

const char* ConstantInfoMetadataNode::get_c_struct_name() const { return "TVMConstantInfo"; }

}  // namespace metadata

class MetadataModuleNode : public ::tvm::runtime::ModuleNode {
 public:
  explicit MetadataModuleNode(runtime::metadata::Metadata metadata)
      : metadata_{::std::move(metadata)} {}

  const char* type_key() const final { return "metadata_module"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final { return ModulePropertyMask::kBinarySerializable; }

  static Module LoadFromBinary() {
    return Module(make_object<MetadataModuleNode>(runtime::metadata::Metadata()));
  }

  void SaveToBinary(dmlc::Stream* stream) final {}

  PackedFunc GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) {
    if (name == "get_metadata") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        if (!metadata_.defined()) {
          TVMFunctionHandle f_handle;
          int32_t ret_code = TVMBackendGetFuncFromEnv(this, symbol::tvm_get_c_metadata, &f_handle);
          ICHECK_EQ(ret_code, 0) << "Unable to locate " << symbol::tvm_get_c_metadata
                                 << " PackedFunc";

          TVMValue ret_value;
          int ret_type_code;
          ret_code = TVMFuncCall(f_handle, nullptr, nullptr, 0, &ret_value, &ret_type_code);
          ICHECK_EQ(ret_code, 0) << "Invoking " << symbol::tvm_get_c_metadata
                                 << ": TVMFuncCall returned " << ret_code;

          ICHECK_EQ(ret_type_code, kTVMOpaqueHandle)
              << "Expected kOpaqueHandle returned; got " << ret_type_code;
          ICHECK(ret_value.v_handle != nullptr)
              << symbol::tvm_get_c_metadata << " returned nullptr";

          metadata_ = runtime::metadata::Metadata(
              static_cast<const struct ::TVMMetadata*>(ret_value.v_handle));
        }

        *rv = metadata_;
        return;
      });
    }

    return PackedFunc();
  }

 private:
  runtime::metadata::Metadata metadata_;
};

Module MetadataModuleCreate(metadata::Metadata metadata) {
  return Module(make_object<MetadataModuleNode>(metadata));
}

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_metadata_module")
    .set_body([](TVMArgs args, TVMRetValue* rv) { *rv = MetadataModuleNode::LoadFromBinary(); });

}  // namespace runtime
}  // namespace tvm
