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

MetadataArray::MetadataArray(Array<ObjectRef> array, const char* c_type)
    : MetadataBase{make_object<MetadataArrayNode>(array, c_type)} {}

std::string MetadataArrayNode::get_name() { return "MetadataArray"; }

TVM_REGISTER_OBJECT_TYPE(MetadataBaseNode);
TVM_REGISTER_OBJECT_TYPE(MetadataArrayNode);

ArrayAccessor<struct TVMTensorInfo, TensorInfo> MetadataNode::inputs() {
  if (inputs_refs_.get() == nullptr) {
    inputs_refs_.reset(new ::std::vector<TensorInfo>());
  }
  return ArrayAccessor<struct TVMTensorInfo, TensorInfo>(data_->inputs, data_->num_inputs,
                                                         inputs_refs_);
}
ArrayAccessor<struct TVMTensorInfo, TensorInfo> MetadataNode::outputs() {
  if (outputs_refs_.get() == nullptr) {
    outputs_refs_.reset(new ::std::vector<TensorInfo>());
  }
  return ArrayAccessor<struct TVMTensorInfo, TensorInfo>(data_->outputs, data_->num_outputs,
                                                         outputs_refs_);
}
ArrayAccessor<const char*, ::tvm::runtime::String> MetadataNode::devices() {
  if (devices_refs_.get() == nullptr) {
    devices_refs_.reset(new ::std::vector<::tvm::runtime::String>());
  }
  return ArrayAccessor<const char*, ::tvm::runtime::String>(data_->devices, data_->num_devices,
                                                            devices_refs_);
}
Metadata::Metadata(const struct ::TVMMetadata* data)
    : MetadataBase{make_object<MetadataNode>(data)} {}
std::string MetadataNode::get_name() { return std::string{"Metadata"}; }
TVM_REGISTER_OBJECT_TYPE(MetadataNode);
TensorInfo::TensorInfo(const struct ::TVMTensorInfo* data)
    : MetadataBase{make_object<TensorInfoNode>(data)} {}
std::string TensorInfoNode::get_name() { return std::string{"TensorInfo"}; }

}  // namespace metadata

class MetadataModuleNode : public ::tvm::runtime::ModuleNode {
 public:
  explicit MetadataModuleNode(runtime::metadata::Metadata metadata) {
    // CHECK((metadata.defined() && code.size() > 0) || (!metadata.defined() && code.size() == 0))
    //   << "metadata and code must both be either defined (when passed from compiler) or undefined
    //   "
    //   << "(when passed from runtime)";
    metadata_ = metadata;
    //    code_ = code;
  }

  const char* type_key() const { return "metadata_module"; }

  static Module LoadFromBinary() {
    return Module(make_object<MetadataModuleNode>(runtime::metadata::Metadata()));
  }

  void SaveToBinary(dmlc::Stream* stream) final {}

  PackedFunc GetFunction(const std::string& name, const ObjectPtr<Object>& sptr_to_self) {
    if (name == "get_metadata") {
      return PackedFunc([this, sptr_to_self](TVMArgs args, TVMRetValue* rv) {
        if (!metadata_.defined()) {
          TVMFunctionHandle f_handle;
          int32_t ret_code = TVMBackendGetFuncFromEnv(this, "get_c_metadata", &f_handle);
          CHECK_EQ(ret_code, 0) << "Unable to locate get_c_metadata PackedFunc";

          TVMValue ret_value;
          int ret_type_code;
          ret_code = TVMFuncCall(f_handle, nullptr, nullptr, 0, &ret_value, &ret_type_code);
          CHECK_EQ(ret_code, 0) << "Invoking get_c_metadata: TVMFuncCall returned " << ret_code;

          CHECK_EQ(ret_type_code, kTVMOpaqueHandle)
              << "Expected kOpaqueHandle returned; got " << ret_type_code;
          CHECK(ret_value.v_handle != nullptr) << "get_c_metadata returned nullptr";

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
