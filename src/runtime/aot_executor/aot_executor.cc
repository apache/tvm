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
 * \brief Defines an implementation of Module-based Model Runtime Interface that works with
 *        Ahead-of-Time compilation.
 * \file aot_executor.cc
 */

#include "aot_executor.h"

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/name_transforms.h>

#include <limits>
#include <memory>

#include "../meta_data.h"

namespace tvm {
namespace runtime {

AotExecutor::AotExecutor(tvm::runtime::Module module, const std::vector<Device>& devs)
    : module_{module}, devices_{devs} {
  auto fmetadata = module->GetFunction("get_metadata");
  CHECK(fmetadata != nullptr) << "Expected a module with PackedFunc get_metadata";
  auto ret_value = fmetadata();
  metadata_ = ret_value.AsObjectRef<tvm::runtime::metadata::Metadata>();

  ICHECK_EQ(devices_.size(), 1) << "Expect exactly 1 device passed.";
  DLDevice expected_device{kDLCPU, 0};
  ICHECK_EQ(devices_[0].device_id, expected_device.device_id)
      << "At this time, AOTExecutor supports only execution on kDLCPU 0";
  // TODO(tvm-team): Temporary hack since Hexagon is defined different than kDLCPU.
  bool is_valid_device =
      (devices_[0].device_type == kDLHexagon) || (devices_[0].device_type == kDLCPU);
  CHECK(is_valid_device)
      << "At this time, AOTExecutor supports only execution on kDLCPU 0 or kDLHexagon 0";

  for (auto input : metadata_->inputs()) {
    // TODO(areusch): Encode device information in Metadata.
    args_.emplace_back(NDArray::Empty(ShapeTuple(input->shape().begin(), input->shape().end()),
                                      input->dtype(), devices_[0]));
  }

  for (auto output : metadata_->outputs()) {
    args_.emplace_back(NDArray::Empty(ShapeTuple(output->shape().begin(), output->shape().end()),
                                      output->dtype(), devices_[0]));
  }

  // USMP is used
  if (metadata_->num_workspace_pools()) {
    // merge all constants into one ndarray
    int64_t blob_len = 0;
    for (const auto& c : metadata_->constant_pools()) {
      auto data = c->data();
      int64_t byte_size = GetDataSize(*data.operator->()) + c->byte_offset();
      blob_len = blob_len > byte_size ? blob_len : byte_size;
    }
    ICHECK(blob_len < std::numeric_limits<int32_t>::max());
    NDArray ci = NDArray::Empty({blob_len}, DataType::UInt(8), devices_[0]);
    for (const auto& c : metadata_->constant_pools()) {
      auto data = c->data();
      data.CopyToBytes(static_cast<uint8_t*>(ci->data) + c->byte_offset(),
                       GetDataSize(*data.operator->()));
    }
    // Emplace constant node pool only if workspace pools supplied
    args_.emplace_back(ci);

    int32_t pool_len = 0;
    for (auto pool : metadata_->workspace_pools()) {
      pool_len =
          GetDataSize(*NDArray::Empty({pool->shape()}, pool->dtype(), devices_[0]).operator->());
      args_.emplace_back(NDArray::Empty({pool_len}, DataType::UInt(8), devices_[0]));
    }
  }
}

PackedFunc AotExecutor::GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) {
  // Return member functions during query.
  if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (String::CanConvertFrom(args[0])) {
        int in_idx = this->GetInputIndex(tvm::runtime::SanitizeName(args[0].operator String()));
        if (in_idx >= 0) this->SetInput(in_idx, args[1]);
      } else {
        this->SetInput(args[0], args[1]);
      }
    });
  } else if (name == "set_input_zero_copy") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (String::CanConvertFrom(args[0])) {
        int in_idx = this->GetInputIndex(tvm::runtime::SanitizeName(args[0].operator String()));
        if (in_idx >= 0) this->SetInputZeroCopy(in_idx, args[1]);
      } else {
        this->SetInputZeroCopy(args[0], args[1]);
      }
    });
  } else if (name == "set_output_zero_copy") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (String::CanConvertFrom(args[0])) {
        int out_idx = this->GetOutputIndex(tvm::runtime::SanitizeName(args[0].operator String()));
        if (out_idx >= 0) this->SetOutputZeroCopy(out_idx, args[1]);
      } else {
        this->SetOutputZeroCopy(args[0], args[1]);
      }
    });
  } else if (name == "get_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (args.num_args == 2) {
        this->CopyOutputTo(args[0], args[1]);
      } else {
        *rv = this->GetOutput(args[0]);
      }
    });
  } else if (name == "get_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      int in_idx = 0;
      if (String::CanConvertFrom(args[0])) {
        in_idx = this->GetInputIndex(tvm::runtime::SanitizeName(args[0].operator String()));
      } else {
        in_idx = args[0];
      }
      if (in_idx >= 0) {
        *rv = this->GetInput(in_idx);
      }
    });
  } else if (name == "get_num_outputs") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->NumOutputs(); });
  } else if (name == "get_num_inputs") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->NumInputs(); });
  } else if (name == "run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->Run(); });
  } else if (name == "get_input_index") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK(String::CanConvertFrom(args[0])) << "Input key is not a string";
      *rv = this->GetInputIndex(tvm::runtime::SanitizeName(args[0].operator String()));
    });
  } else if (name == "get_input_name") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->GetInputName(args[0]); });
  } else {
    return PackedFunc();
  }
}

void AotExecutor::Run() {
  auto pf = module_.GetFunction(
      get_name_mangled(metadata_->mod_name(), ::tvm::runtime::symbol::tvm_module_main),
      true /* query_imports */);
  ICHECK(pf != nullptr) << "Module entrypoint is not defined";

  const int num_args = args_.size();
  auto call_values = ::std::make_unique<TVMValue[]>(num_args);
  auto call_type_codes = ::std::make_unique<int[]>(num_args);
  for (int i = 0; i < num_args; ++i) {
    auto managed = args_[i].ToDLPack();
    call_values.get()[i].v_handle = &managed->dl_tensor;
    call_type_codes.get()[i] = kTVMDLTensorHandle;
  }

  TVMArgs args{call_values.get(), call_type_codes.get(), num_args};
  TVMRetValue rv;
  pf.CallPacked(args, &rv);
}

int AotExecutor::GetInputIndex(const std::string& name) {
  auto inputs = metadata_->inputs();
  for (unsigned int i = 0; i < inputs.size(); i++) {
    if (inputs[i]->name() == name) {
      return i;
    }
  }
  ICHECK(false) << "Invalid input name.";
}

std::string AotExecutor::GetInputName(int index) {
  auto inputs = metadata_->inputs();
  return inputs[index]->name();
}

int AotExecutor::GetOutputIndex(const std::string& name) {
  auto outputs = metadata_->outputs();
  for (unsigned int i = 0; i < outputs.size(); i++) {
    if (outputs[i]->name() == name) {
      return i;
    }
  }
  return -1;
}

void AotExecutor::SetInput(int index, DLTensor* data_ref) { args_[index].CopyFrom(data_ref); }

void AotExecutor::SetInputZeroCopy(int index, DLTensor* data_ref) {
  ICHECK(false) << "not implemented";
}

void AotExecutor::SetOutputZeroCopy(int index, DLTensor* data_ref) {
  ICHECK(false) << "not implemented";
}

int AotExecutor::NumOutputs() const { return metadata_->num_outputs(); }

int AotExecutor::NumInputs() const { return metadata_->num_inputs(); }

NDArray AotExecutor::GetInput(int index) const { return args_[index]; }

NDArray AotExecutor::GetOutput(int index) const { return args_[metadata_->num_inputs() + index]; }

void AotExecutor::CopyOutputTo(int index, DLTensor* data_out) { GetOutput(index).CopyTo(data_out); }

}  // namespace runtime
}  // namespace tvm
