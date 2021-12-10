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
 * \file tvm/target/metadata.h
 * \brief Extends Metadata for use in the compiler.
 */
#ifndef TVM_TARGET_METADATA_H_
#define TVM_TARGET_METADATA_H_

#include <tvm/runtime/metadata.h>

#include <memory>
#include <string>
#include <vector>

namespace tvm {
namespace target {
namespace metadata {

class VisitableMetadataNode : public ::tvm::runtime::metadata::MetadataNode {
 public:
  explicit VisitableMetadataNode(const struct ::TVMMetadata* data) : MetadataNode{data} {}
  VisitableMetadataNode() : MetadataNode{nullptr} {}

  void VisitAttrs(AttrVisitor* v) {
    int64_t version_cpp{version()};
    v->Visit("version", &version_cpp);
    auto inputs_array = Array<ObjectRef>();
    auto inputs_accessor = inputs();
    inputs_array.reserve(num_inputs());
    for (int64_t i = 0; i < num_inputs(); ++i) {
      inputs_array.push_back(::tvm::runtime::metadata::TensorInfo{inputs_accessor[i]});
    }
    ::tvm::runtime::metadata::MetadataArray inputs_metadata_array{inputs_array,
                                                                  "struct TVMTensorInfo"};
    v->Visit("inputs", &inputs_metadata_array);
    auto outputs_array = Array<ObjectRef>();
    auto outputs_accessor = outputs();
    outputs_array.reserve(num_outputs());
    for (int64_t i = 0; i < num_outputs(); ++i) {
      outputs_array.push_back(::tvm::runtime::metadata::TensorInfo{outputs_accessor[i]});
    }
    ::tvm::runtime::metadata::MetadataArray outputs_metadata_array{outputs_array,
                                                                   "struct TVMTensorInfo"};
    v->Visit("outputs", &outputs_metadata_array);
    auto devices_array = Array<ObjectRef>();
    auto devices_accessor = devices();
    devices_array.reserve(num_devices());
    for (int64_t i = 0; i < num_devices(); ++i) {
      devices_array.push_back(::tvm::runtime::String{devices_accessor[i]});
    }
    ::tvm::runtime::metadata::MetadataArray devices_metadata_array{devices_array, "const char*"};
    v->Visit("devices", &devices_metadata_array);
    ::std::string executor_cpp{data()->executor};
    v->Visit("executor", &executor_cpp);
    ::std::string mod_name_cpp{data()->mod_name};
    v->Visit("mod_name", &mod_name_cpp);
    ::std::string interface_api_cpp{data()->interface_api};
    v->Visit("interface_api", &interface_api_cpp);
    bool use_unpacked_api_cpp{use_unpacked_api()};
    v->Visit("use_unpacked_api", &use_unpacked_api_cpp);
  }
};

class InMemoryMetadataNode : public ::tvm::target::metadata::VisitableMetadataNode {
 public:
  InMemoryMetadataNode()
      : InMemoryMetadataNode(0 /* version */, {} /* inputs */, {} /* outputs */, {} /* devices */,
                             "" /* executor */, "" /* mod_name */, "" /* interface_api */,
                             false /* use_unpacked_api */
        ) {}
  InMemoryMetadataNode(int64_t version,
                       const ::std::vector<::tvm::runtime::metadata::TensorInfo>& inputs,
                       const ::std::vector<::tvm::runtime::metadata::TensorInfo>& outputs,
                       const ::std::vector<::std::string>& devices,
                       const ::tvm::runtime::String executor, const ::tvm::runtime::String mod_name,
                       const ::tvm::runtime::String interface_api, bool use_unpacked_api)
      : VisitableMetadataNode{&storage_},
        inputs_{new struct TVMTensorInfo[inputs.size()]()},
        inputs_objs_{inputs},
        outputs_{new struct TVMTensorInfo[outputs.size()]()},
        outputs_objs_{outputs},
        devices_{new const char*[devices.size()]()},
        executor_{executor},
        mod_name_{mod_name},
        interface_api_{interface_api},
        storage_{version,
                 nullptr,
                 0,
                 nullptr,
                 0,
                 nullptr,
                 0,
                 executor_.c_str(),
                 mod_name_.c_str(),
                 interface_api_.c_str(),
                 use_unpacked_api} {
    storage_.inputs = inputs_.get();
    storage_.num_inputs = inputs.size();
    for (unsigned int i = 0; i < inputs.size(); ++i) {
      inputs_.get()[i] = *inputs[i]->data();
    }
    storage_.outputs = outputs_.get();
    storage_.num_outputs = outputs.size();
    for (unsigned int i = 0; i < outputs.size(); ++i) {
      outputs_.get()[i] = *outputs[i]->data();
    }
    storage_.devices = devices_.get();
    storage_.num_devices = devices.size();
    for (unsigned int i = 0; i < devices.size(); ++i) {
      devices_.get()[i] = devices[i].c_str();
    }
  }

 private:
  ::std::unique_ptr<struct TVMTensorInfo> inputs_;
  std::vector<::tvm::runtime::metadata::TensorInfo> inputs_objs_;
  ::std::unique_ptr<struct TVMTensorInfo> outputs_;
  std::vector<::tvm::runtime::metadata::TensorInfo> outputs_objs_;
  ::std::unique_ptr<const char*> devices_;
  ::std::string executor_;
  ::std::string mod_name_;
  ::std::string interface_api_;
  struct ::TVMMetadata storage_;
};

class VisitableTensorInfoNode : public ::tvm::runtime::metadata::TensorInfoNode {
 public:
  explicit VisitableTensorInfoNode(const struct ::TVMTensorInfo* data) : TensorInfoNode{data} {}
  VisitableTensorInfoNode() : TensorInfoNode{nullptr} {}

  void VisitAttrs(AttrVisitor* v) {
    ::std::string name_cpp{data()->name};
    v->Visit("name", &name_cpp);
    auto shape_array = Array<ObjectRef>();
    auto shape_accessor = shape();
    shape_array.reserve(num_shape());
    for (int64_t i = 0; i < num_shape(); ++i) {
      shape_array.push_back(::tvm::Integer{static_cast<int>(shape_accessor[i])});
    }
    ::tvm::runtime::metadata::MetadataArray shape_metadata_array{shape_array, "int64_t"};
    v->Visit("shape", &shape_metadata_array);
    ::tvm::runtime::DataType dtype_cpp{dtype()};
    v->Visit("dtype", &dtype_cpp);
  }
};

class InMemoryTensorInfoNode : public ::tvm::target::metadata::VisitableTensorInfoNode {
 public:
  InMemoryTensorInfoNode() : InMemoryTensorInfoNode("", {}, ::tvm::runtime::DataType(0, 0, 0)) {}
  InMemoryTensorInfoNode(const ::tvm::runtime::String& name, const ::std::vector<int64_t>& shape,
                         ::tvm::runtime::DataType dtype)
      : VisitableTensorInfoNode{&storage_},
        name_{name},
        shape_{new int64_t[shape.size()]()},
        storage_{name_.c_str(), nullptr, 0, dtype} {
    storage_.shape = shape_.get();
    storage_.num_shape = shape.size();
    for (unsigned int i = 0; i < shape.size(); ++i) {
      shape_.get()[i] = shape[i];
    }
  }

 private:
  ::std::string name_;
  ::std::unique_ptr<int64_t> shape_;
  struct ::TVMTensorInfo storage_;
};

}  // namespace metadata
}  // namespace target
}  // namespace tvm

#endif  // TVM_TARGET_METADATA_H_
