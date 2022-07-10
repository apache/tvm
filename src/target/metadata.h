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
#include <tvm/ir/memory_pools.h>
#include <tvm/runtime/metadata.h>

#include <memory>
#include <string>
#include <vector>

namespace tvm {
namespace target {
namespace metadata {

/*!
 * \brief Subclass of MetadataNode that implements the VisitAttrs reflection method.
 *
 * This implementation (and other such Visitable subclasses) is compiled into libtvm.so, but not
 * libtvm_runtime.so, because reflection is not supported in libtvm_runtime.so over code size
 * concerns. It is used during compilation by the generic metadata code-generators.
 */
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
    ::tvm::runtime::metadata::MetadataArray inputs_metadata_array{
        inputs_array, ::tvm::runtime::metadata::MetadataKind::kMetadata,
        ::tvm::runtime::metadata::TensorInfoNode::_type_key};
    v->Visit("inputs", &inputs_metadata_array);
    int64_t num_inputs_cpp = num_inputs();
    v->Visit("num_inputs", &num_inputs_cpp);
    auto outputs_array = Array<ObjectRef>();
    auto outputs_accessor = outputs();
    outputs_array.reserve(num_outputs());
    for (int64_t i = 0; i < num_outputs(); ++i) {
      outputs_array.push_back(::tvm::runtime::metadata::TensorInfo{outputs_accessor[i]});
    }
    ::tvm::runtime::metadata::MetadataArray outputs_metadata_array{
        outputs_array, ::tvm::runtime::metadata::MetadataKind::kMetadata,
        ::tvm::runtime::metadata::TensorInfoNode::_type_key};
    v->Visit("outputs", &outputs_metadata_array);
    int64_t num_outputs_cpp = num_outputs();
    v->Visit("num_outputs", &num_outputs_cpp);
    auto pools_array = Array<ObjectRef>();
    auto pools_accessor = workspace_pools();
    pools_array.reserve(num_workspace_pools());
    for (int64_t i = 0; i < num_workspace_pools(); ++i) {
      pools_array.push_back(::tvm::runtime::metadata::TensorInfo{pools_accessor[i]});
    }
    ::tvm::runtime::metadata::MetadataArray workspace_pools_metadata_array{
        pools_array, ::tvm::runtime::metadata::MetadataKind::kMetadata,
        ::tvm::runtime::metadata::TensorInfoNode::_type_key};
    v->Visit("workspace_pools", &workspace_pools_metadata_array);
    int64_t num_workspace_pools_cpp = num_workspace_pools();
    v->Visit("num_workspace_pools", &num_workspace_pools_cpp);

    auto consts_array = Array<ObjectRef>();
    auto consts_accessor = constant_pools();
    consts_array.reserve(num_constant_pools());
    for (int64_t i = 0; i < num_constant_pools(); ++i) {
      consts_array.push_back(::tvm::runtime::metadata::ConstantInfoMetadata{consts_accessor[i]});
    }

    int64_t num_const_pools_cpp = num_constant_pools();
    ::tvm::runtime::metadata::MetadataArray constant_pools_metadata_array{
        consts_array, ::tvm::runtime::metadata::MetadataKind::kMetadata,
        ::tvm::runtime::metadata::ConstantInfoMetadataNode::_type_key};
    v->Visit("constant_pools", &constant_pools_metadata_array);
    v->Visit("num_constant_pools", &num_const_pools_cpp);
    ::std::string mod_name_cpp{data()->mod_name};
    v->Visit("mod_name", &mod_name_cpp);
  }
};

/*!
 * \brief Subclass of MetadataNode which also owns the backing C structures.
 *
 * This class (and other InMemory subclasses) are used during compilation to instantiate Metadata
 * instances whose storage lives outside of .rodata. This class exists because the Module returned
 * from tvm.relay.build must also be ready to run inference.
 */
class InMemoryMetadataNode : public ::tvm::target::metadata::VisitableMetadataNode {
 public:
  InMemoryMetadataNode()
      : InMemoryMetadataNode(0 /* version */, {} /* inputs */, {} /* outputs */,
                             {} /* workspace_pools */, {} /* constant_pools */, "" /* mod_name */) {
  }
  InMemoryMetadataNode(int64_t version,
                       const ::std::vector<::tvm::runtime::metadata::TensorInfo>& inputs,
                       const ::std::vector<::tvm::runtime::metadata::TensorInfo>& outputs,
                       const ::std::vector<::tvm::runtime::metadata::TensorInfo>& workspace_pools,
                       const ::std::vector<::tvm::ConstantInfo>& constant_pools,
                       const ::tvm::runtime::String mod_name)
      : VisitableMetadataNode{&storage_},
        inputs_{new struct TVMTensorInfo[inputs.size()]},
        inputs_objs_{inputs},
        outputs_{new struct TVMTensorInfo[outputs.size()]},
        outputs_objs_{outputs},
        workspace_pools_{new struct TVMTensorInfo[workspace_pools.size()]},
        workspace_pools_objs_{workspace_pools},
        constant_pools_{new struct TVMConstantInfo[constant_pools.size()]},
        constant_pools_objs_{constant_pools},
        mod_name_{mod_name},
        storage_{version, nullptr, 0ull,    nullptr, 0ull,
                 nullptr, 0ull,    nullptr, 0ull,    mod_name_.c_str()} {
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
    storage_.workspace_pools = workspace_pools_.get();
    storage_.num_workspace_pools = workspace_pools.size();
    for (unsigned int i = 0; i < workspace_pools.size(); ++i) {
      workspace_pools_.get()[i] = *workspace_pools[i]->data();
    }
    storage_.constant_pools = constant_pools_.get();
    storage_.num_constant_pools = constant_pools.size();
    for (size_t i = 0; i < constant_pools.size(); ++i) {
      constant_pools_.get()[i].name_hint = constant_pools[i]->name_hint.c_str();
      constant_pools_.get()[i].byte_offset = constant_pools[i]->byte_offset.IntValue();

      std::string bytes;
      dmlc::MemoryStringStream stream(&bytes);
      auto data = constant_pools[i]->data;
      data.Save(&stream);
      // Allocated mem freed in destructor
      constant_pools_.get()[i].data_len = bytes.size();
      char* a = reinterpret_cast<char*>(malloc(bytes.size()));
      constant_pools_.get()[i].data_bytes = a;
      memcpy(a, bytes.c_str(), bytes.size());
    }
  }

  ~InMemoryMetadataNode() {
    // frees allocated mem for const_objs_
    for (int i = 0; i < storage_.num_constant_pools; ++i) {
      free(const_cast<void*>(constant_pools_.get()[i].data_bytes));
    }
  }

 private:
  ::std::unique_ptr<struct TVMTensorInfo[]> inputs_;
  std::vector<::tvm::runtime::metadata::TensorInfo> inputs_objs_;
  ::std::unique_ptr<struct TVMTensorInfo[]> outputs_;
  std::vector<::tvm::runtime::metadata::TensorInfo> outputs_objs_;
  ::std::unique_ptr<struct TVMTensorInfo[]> workspace_pools_;
  std::vector<::tvm::runtime::metadata::TensorInfo> workspace_pools_objs_;
  ::std::unique_ptr<struct TVMConstantInfo[]> constant_pools_;
  std::vector<::tvm::ConstantInfo> constant_pools_objs_;
  ::std::string mod_name_;
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
    ::tvm::runtime::metadata::MetadataArray shape_metadata_array{
        shape_array, ::tvm::runtime::metadata::MetadataKind::kInt64, nullptr};
    v->Visit("shape", &shape_metadata_array);
    int64_t num_shape_cpp = num_shape();
    v->Visit("num_shape", &num_shape_cpp);
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
  ::std::unique_ptr<int64_t[]> shape_;
  struct ::TVMTensorInfo storage_;
};

class VisitableConstantInfoMetadataNode
    : public ::tvm::runtime::metadata::ConstantInfoMetadataNode {
 public:
  explicit VisitableConstantInfoMetadataNode(const struct ::TVMConstantInfo* data)
      : ConstantInfoMetadataNode{data} {}
  VisitableConstantInfoMetadataNode() : ConstantInfoMetadataNode{nullptr} {}

  void VisitAttrs(AttrVisitor* v) {
    ::std::string name_cpp{name_hint()};
    v->Visit("name_hint", &name_cpp);

    uint64_t byte_offset_cpp{byte_offset()};
    v->Visit("byte_offset", &byte_offset_cpp);

    ::tvm::runtime::NDArray data_cpp = data();
    v->Visit("data", &data_cpp);
  }
};

}  // namespace metadata
}  // namespace target
}  // namespace tvm

#endif  // TVM_TARGET_METADATA_H_
