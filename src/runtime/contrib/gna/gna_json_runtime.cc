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
 * \file src/runtime/contrib/gna/gna_json_runtime.cc
 * \brief A simple JSON runtime for GNA.
 */

#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/ndarray.h>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "../../../../gna/src/gna-api/gna2-api.h"
#include "../json/json_node.h"
#include "../json/json_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

static void CheckGnaStatus(Gna2Status status, const std::string& context) {
  if (status != Gna2StatusSuccess) {
    auto const size = Gna2StatusGetMaxMessageLength();
    auto msg = std::unique_ptr<char[]>(new char[size]());
    Gna2StatusGetMessage(status, msg.get(), size);
    LOG(FATAL) << "GNA Error in " << context << ": " << msg.get();
  }
}

class GNAJSONRuntime : public JSONRuntimeBase {
 public:
  GNAJSONRuntime(const std::string& symbol_name, const std::string& graph_json,
                 const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names),
        device_index_(0),
        model_id_(GNA2_DISABLED),
        request_config_id_(GNA2_DISABLED) {}

  ~GNAJSONRuntime() override {
    if (request_config_id_ != GNA2_DISABLED) {
      Gna2RequestConfigRelease(request_config_id_);
    }
    if (model_id_ != GNA2_DISABLED) {
      Gna2ModelRelease(model_id_);
    }
    if (device_index_ != GNA2_DISABLED) {
      Gna2DeviceClose(device_index_);
    }
  }

  const char* type_key() const override { return "gna_json"; }

  void Run() override { LOG(FATAL) << "Use Run(PackedArgs) instead"; }

  void Init(const Array<NDArray>& consts) override {
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";

    SetupConstants(consts);
    BuildEngine();
  }

  void Run(ffi::PackedArgs args) {
    std::vector<const DLTensor*> dl_tensors(NumEntries());

    for (size_t i = 0; i < static_cast<size_t>(args.size()); i++) {
      auto eid = i < input_var_eid_.size() ? input_var_eid_[i]
                                           : EntryID(outputs_[i - input_var_eid_.size()]);

      const DLTensor* arg;
      if (auto opt_nd = args[i].as<NDArray>()) {
        NDArray arr = opt_nd.value();
        arg = arr.operator->();
      } else {
        arg = args[i].cast<DLTensor*>();
      }

      dl_tensors[eid] = arg;
    }

    MapTensorsToGNA(dl_tensors);

    uint32_t request_id;
    Gna2Status status = Gna2RequestEnqueue(request_config_id_, &request_id);
    CheckGnaStatus(status, "Gna2RequestEnqueue");

    status = Gna2RequestWait(request_id, 1000);
    CheckGnaStatus(status, "Gna2RequestWait");
  }

  void MapTensorsToGNA(const std::vector<const DLTensor*>& dl_tensors) {
    size_t input_idx = 0;
    size_t output_idx = 0;

    for (size_t i = 0; i < input_var_eid_.size() && input_idx < input_tensors_.size(); ++i) {
      auto eid = input_var_eid_[i];
      if (eid < dl_tensors.size() && dl_tensors[eid]) {
        input_tensors_[input_idx] = CreateGNATensor(dl_tensors[eid]);
        input_idx++;
      }
    }

    for (size_t i = 0; i < outputs_.size() && output_idx < output_tensors_.size(); ++i) {
      auto eid = EntryID(outputs_[i]);
      if (eid < dl_tensors.size() && dl_tensors[eid]) {
        output_tensors_[output_idx] = CreateGNATensor(dl_tensors[eid]);
        output_idx++;
      }
    }

    SetGNARequestBuffers();
  }

  void SetGNARequestBuffers() {
    if (input_tensors_.empty() || output_tensors_.empty()) {
      return;
    }

    if (output_tensors_.size() > 0) {
      Gna2Status status = Gna2RequestConfigEnableActiveList(request_config_id_, 0, 1, nullptr);
      if (status != Gna2StatusSuccess) {
        LOG(INFO) << "Active list not supported, continuing without it";
      }
    }
  }

  ffi::Function GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) override {
    if (this->symbol_name_ == name) {
      return ffi::Function([sptr_to_self, this](ffi::PackedArgs args, ffi::Any* rv) {
        ICHECK(this->initialized_) << "The module has not been initialized";
        this->Run(args);
      });
    } else {
      return JSONRuntimeBase::GetFunction(name, sptr_to_self);
    }
  }

 private:
  uint32_t device_index_;
  uint32_t model_id_;
  uint32_t request_config_id_;
  std::vector<Gna2Operation> gna_operations_;
  std::unique_ptr<Gna2Model> gna_model_;
  std::vector<Gna2Tensor> input_tensors_;
  std::vector<Gna2Tensor> output_tensors_;
  std::vector<Gna2Tensor> weight_tensors_;
  std::vector<std::vector<uint8_t>> tensor_buffers_;

  Gna2DataType GetGNADataType(DLDataType dl_type) {
    if (dl_type.code == kDLInt && dl_type.bits == 32) {
      return Gna2DataTypeInt32;
    } else if (dl_type.code == kDLInt && dl_type.bits == 16) {
      return Gna2DataTypeInt16;
    } else if (dl_type.code == kDLInt && dl_type.bits == 8) {
      return Gna2DataTypeInt8;
    }
    LOG(FATAL) << "Unsupported data type for GNA: " << static_cast<int>(dl_type.code)
               << " bits=" << static_cast<int>(dl_type.bits);
    return Gna2DataTypeInt32;
  }

  Gna2Tensor CreateGNATensor(const DLTensor* dl_tensor) {
    auto gna_dtype = GetGNADataType(dl_tensor->dtype);

    if (dl_tensor->ndim == 1) {
      return Gna2TensorInit1D(dl_tensor->shape[0], gna_dtype, dl_tensor->data);
    } else if (dl_tensor->ndim == 2) {
      return Gna2TensorInit2D(dl_tensor->shape[0], dl_tensor->shape[1], gna_dtype, dl_tensor->data);
    } else if (dl_tensor->ndim == 3) {
      return Gna2TensorInit3D(dl_tensor->shape[0], dl_tensor->shape[1], dl_tensor->shape[2],
                              gna_dtype, dl_tensor->data);
    } else if (dl_tensor->ndim == 4) {
      return Gna2TensorInit4D(dl_tensor->shape[0], dl_tensor->shape[1], dl_tensor->shape[2],
                              dl_tensor->shape[3], gna_dtype, dl_tensor->data);
    }
    LOG(FATAL) << "Unsupported tensor dimensionality for GNA: " << dl_tensor->ndim;
    return Gna2TensorInitDisabled();
  }

  void BuildEngine() {
    Gna2Status status = Gna2DeviceOpen(device_index_);
    CheckGnaStatus(status, "Gna2DeviceOpen");

    BuildGNAOperations();

    gna_model_ = std::make_unique<Gna2Model>();
    gna_model_->NumberOfOperations = gna_operations_.size();
    if (!gna_operations_.empty()) {
      gna_model_->Operations = gna_operations_.data();
    }

    status = Gna2ModelCreate(device_index_, gna_model_.get(), &model_id_);
    CheckGnaStatus(status, "Gna2ModelCreate");

    status = Gna2RequestConfigCreate(model_id_, &request_config_id_);
    CheckGnaStatus(status, "Gna2RequestConfigCreate");
  }

  void BuildGNAOperations() {
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() == "kernel") {
        CreateGNAOperation(nid, node);
      }
    }
  }

  void CreateGNAOperation(size_t nid, const JSONGraphNode& node) {
    auto op_name = node.GetOpName();
    Gna2Operation gna_op = {};

    auto inputs = node.GetInputs();
    if (inputs.empty()) {
      LOG(WARNING) << "GNA operation has no inputs, skipping: " << op_name;
      return;
    }

    size_t input_tensor_idx = input_tensors_.size();
    size_t output_tensor_idx = output_tensors_.size();

    input_tensors_.resize(input_tensor_idx + inputs.size());
    output_tensors_.resize(output_tensor_idx + 1);

    if (op_name.find("gna.dense") != std::string::npos) {
      Gna2Tensor dummy_weights = Gna2TensorInitDisabled();
      Gna2Tensor dummy_biases = Gna2TensorInitDisabled();
      Gna2Tensor dummy_activation = Gna2TensorInitDisabled();

      Gna2Status status = Gna2OperationInitFullyConnectedAffine(
          &gna_op, nullptr, &input_tensors_[input_tensor_idx], &output_tensors_[output_tensor_idx],
          &dummy_weights, &dummy_biases, &dummy_activation);
      CheckGnaStatus(status, "Gna2OperationInitFullyConnectedAffine");

    } else if (op_name.find("gna.conv1d") != std::string::npos) {
      Gna2Tensor dummy_filters = Gna2TensorInitDisabled();
      Gna2Tensor dummy_biases = Gna2TensorInitDisabled();
      Gna2Tensor dummy_activation = Gna2TensorInitDisabled();
      Gna2Shape dummy_stride = Gna2ShapeInit1D(1);
      Gna2BiasMode bias_mode = Gna2BiasModeDefault;

      Gna2Status status = Gna2OperationInitConvolution(
          &gna_op, nullptr, &input_tensors_[input_tensor_idx], &output_tensors_[output_tensor_idx],
          &dummy_filters, &dummy_biases, &dummy_activation, &dummy_stride, &bias_mode);
      CheckGnaStatus(status, "Gna2OperationInitConvolution");

    } else if (op_name.find("gna.relu") != std::string::npos) {
      Gna2Tensor dummy_weights = Gna2TensorInitDisabled();
      Gna2Tensor dummy_biases = Gna2TensorInitDisabled();
      Gna2Tensor dummy_activation = Gna2TensorInitDisabled();

      Gna2Status status = Gna2OperationInitElementWiseAffine(
          &gna_op, nullptr, &input_tensors_[input_tensor_idx], &output_tensors_[output_tensor_idx],
          &dummy_weights, &dummy_biases, &dummy_activation);
      CheckGnaStatus(status, "Gna2OperationInitElementWiseAffine");

    } else {
      LOG(FATAL) << "Unsupported GNA operation: " << op_name;
    }

    gna_operations_.push_back(gna_op);
  }
};

runtime::Module GNAJSONRuntimeCreate(String symbol_name, String graph_json,
                                     const Array<String>& const_names) {
  auto n = make_object<GNAJSONRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_FFI_STATIC_INIT_BLOCK({
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("runtime.GNAJSONRuntimeCreate", GNAJSONRuntimeCreate)
      .def("runtime.module.loadbinary_gna_json", JSONRuntimeBase::LoadFromBinary<GNAJSONRuntime>);
});

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
