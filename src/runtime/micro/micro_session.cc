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
 *  Copyright (c) 2019 by Contributors
 * \file micro_session.cc
 * \brief session to manage multiple micro modules
 */

#include <tvm/runtime/registry.h>
#include <memory>
#include <vector>
#include "micro_session.h"
#include "low_level_device.h"
#include "host_low_level_device.h"
#include "target_data_layout_encoder.h"

namespace tvm {
namespace runtime {

MicroSession::MicroSession() : valid_(false) { }

MicroSession::~MicroSession() { }

void MicroSession::InitSession(const TVMArgs& args) {
  valid_ = true;

  DevBaseOffset curr_start_offset = kDeviceStart;
  for (size_t i = 0; i < static_cast<size_t>(SectionKind::kNumKinds); i++) {
    size_t section_size = GetDefaultSectionSize(static_cast<SectionKind>(i));
    section_allocators_[i] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = curr_start_offset,
      .size = section_size,
    });
    curr_start_offset += section_size;
  }
  memory_size_ = curr_start_offset.cast_to<size_t>();

  const std::string& device_type = args[0];
  const std::string& binary_path = args[1];
  const std::string& toolchain_prefix = args[2];
  // TODO(weberlo): make device type enum
  if (device_type == "host") {
    low_level_device_ = HostLowLevelDeviceCreate(memory_size_);
  } else {
    LOG(FATAL) << "Unsupported micro low-level device";
  }
  SetInitBinaryPath(args[1]);
  CHECK(!init_binary_path_.empty()) << "init library not initialized";
  init_stub_info_ = LoadBinary(init_binary_path_);
  utvm_main_symbol_ = init_symbol_map()["UTVMMain"];
  utvm_done_symbol_ = init_symbol_map()["UTVMDone"];

  // Patch workspace pointers to the start of the workspace section.
  DevBaseOffset workspace_start_offset = GetAllocator(SectionKind::kWorkspace)->start_offset();
  DevBaseOffset workspace_end_offset = GetAllocator(SectionKind::kWorkspace)->max_end_offset();
  void* workspace_start_addr =
      (workspace_start_offset + low_level_device_->base_addr()).cast_to<void*>();
  void* workspace_end_addr =
      (workspace_end_offset + low_level_device_->base_addr()).cast_to<void*>();
  DevSymbolWrite(init_symbol_map(), "utvm_workspace_begin", workspace_start_addr);
  DevSymbolWrite(init_symbol_map(), "utvm_workspace_end", workspace_end_addr);
}

void MicroSession::EndSession() {
  valid_ = false;

  for (size_t i = 0; i < static_cast<size_t>(SectionKind::kNumKinds); i++) {
    section_allocators_[i] = nullptr;
  }

  low_level_device_ = nullptr;
}

DevBaseOffset MicroSession::AllocateInSection(SectionKind type, size_t size) {
  return GetAllocator(type)->Allocate(size);
}

void MicroSession::FreeInSection(SectionKind type, DevBaseOffset ptr) {
  return GetAllocator(type)->Free(ptr);
}

std::string MicroSession::ReadString(DevBaseOffset str_offset) {
  std::stringstream result;
  const size_t buf_size = 256;
  std::vector<char> buf(buf_size, 0);
  size_t i = buf_size;
  while (i == buf_size) {
    low_level_device()->Read(str_offset, buf.data(), buf_size);
    i = 0;
    while (i < buf_size) {
      if (buf[i] == 0) break;
      result << buf[i];
      i++;
    }
    str_offset = str_offset + i;
  }
  return result.str();
}

void MicroSession::PushToExecQueue(DevBaseOffset func, const TVMArgs& args) {
  int32_t (*func_dev_addr)(void*, void*, int32_t) =
      reinterpret_cast<int32_t (*)(void*, void*, int32_t)>(
      (func + low_level_device()->base_addr()).value());

  // Create an allocator stream for the memory region after the most recent
  // allocation in the args section.
  DevAddr args_addr =
      low_level_device()->base_addr() + GetAllocator(SectionKind::kArgs)->curr_end_offset();
  TargetDataLayoutEncoder encoder(args_addr);

  EncoderAppend(&encoder, args);
  // Flush `stream` to device memory.
  DevBaseOffset stream_dev_offset =
      GetAllocator(SectionKind::kArgs)->Allocate(encoder.buf_size());
  low_level_device()->Write(stream_dev_offset,
                            reinterpret_cast<void*>(encoder.data()),
                            encoder.buf_size());

  UTVMTask task = {
      .func = func_dev_addr,
      .args = args_addr.cast_to<UTVMArgs*>(),
  };
  // Write the task.
  low_level_device()->Write(init_symbol_map()["task"], &task, sizeof(UTVMTask));

  low_level_device()->Execute(utvm_main_symbol_, utvm_done_symbol_);

  // Check if there was an error during execution.  If so, log it.
  CheckDeviceError();

  GetAllocator(SectionKind::kArgs)->Free(stream_dev_offset);
}

BinaryInfo MicroSession::LoadBinary(std::string binary_path) {
  DevMemRegion text_section;
  DevMemRegion rodata_section;
  DevMemRegion data_section;
  DevMemRegion bss_section;

  text_section.size = GetSectionSize(binary_path, SectionKind::kText, toolchain_prefix_);
  rodata_section.size = GetSectionSize(binary_path, SectionKind::kRodata, toolchain_prefix_);
  data_section.size = GetSectionSize(binary_path, SectionKind::kData, toolchain_prefix_);
  bss_section.size = GetSectionSize(binary_path, SectionKind::kBss, toolchain_prefix_);

  text_section.start = AllocateInSection(SectionKind::kText, text_section.size);
  rodata_section.start = AllocateInSection(SectionKind::kRodata, rodata_section.size);
  data_section.start = AllocateInSection(SectionKind::kData, data_section.size);
  bss_section.start = AllocateInSection(SectionKind::kBss, bss_section.size);
  CHECK(text_section.start != nullptr && rodata_section.start != nullptr && data_section.start != nullptr &&
        bss_section.start != nullptr) << "not enough space to load module on device";

  const DevBaseAddr base_addr = low_level_device_->base_addr();
  std::string relocated_bin = RelocateBinarySections(
      binary_path,
      text_section.start + base_addr,
      rodata_section.start + base_addr,
      data_section.start + base_addr,
      bss_section.start + base_addr,
      toolchain_prefix_);
  std::string text_contents = ReadSection(relocated_bin, SectionKind::kText, toolchain_prefix_);
  std::string rodata_contents = ReadSection(relocated_bin, SectionKind::kRodata, toolchain_prefix_);
  std::string data_contents = ReadSection(relocated_bin, SectionKind::kData, toolchain_prefix_);
  std::string bss_contents = ReadSection(relocated_bin, SectionKind::kBss, toolchain_prefix_);
  low_level_device_->Write(text_section.start, &text_contents[0], text_section.size);
  low_level_device_->Write(rodata_section.start, &rodata_contents[0], rodata_section.size);
  low_level_device_->Write(data_section.start, &data_contents[0], data_section.size);
  low_level_device_->Write(bss_section.start, &bss_contents[0], bss_section.size);
  SymbolMap symbol_map {relocated_bin, base_addr, toolchain_prefix_};
  return BinaryInfo {
      .text_section = text_section,
      .rodata_section = rodata_section,
      .data_section = data_section,
      .bss_section = bss_section,
      .symbol_map = symbol_map,
  };
}

void MicroSession::SetInitBinaryPath(std::string path) {
  init_binary_path_ = path;
}

DevAddr MicroSession::EncoderAppend(TargetDataLayoutEncoder* encoder, const TVMArgs& args) {
  auto utvm_args_slot = encoder->Alloc<UTVMArgs>();

  const int* type_codes = args.type_codes;
  int num_args = args.num_args;

  auto tvm_vals_slot = encoder->Alloc<TVMValue>(num_args);
  auto type_codes_slot = encoder->Alloc<const int>(num_args);

  for (int i = 0; i < num_args; i++) {
    switch (type_codes[i]) {
      case kNDArrayContainer:
      case kArrayHandle: {
        TVMArray* base_arr_handle = args[i];
        // All uTVM arrays store a `DeviceSpace` struct in their `data` field,
        // which wraps the actual data and stores a reference to the session, in
        // order to prevent premature session destruction.
        void* old_data = base_arr_handle->data;
        // Mutate the array to unwrap the `data` field.
        base_arr_handle->data = reinterpret_cast<DeviceSpace*>(old_data)->data;
        // Now, encode the unwrapped version.
        void* arr_ptr = EncoderAppend(encoder, *base_arr_handle).cast_to<void*>();
        // And restore the original wrapped version.
        base_arr_handle->data = old_data;

        TVMValue val;
        val.v_handle = arr_ptr;
        tvm_vals_slot.WriteValue(val);
        break;
      }
      // TODO(weberlo): Implement `double` and `int64` case.
      case kDLFloat:
      case kDLInt:
      case kDLUInt:
      default:
        LOG(FATAL) << "Unsupported type code for writing args: " << type_codes[i];
        break;
    }
  }
  type_codes_slot.WriteArray(type_codes, num_args);

  UTVMArgs dev_args = {
    .values = tvm_vals_slot.start_addr().cast_to<TVMValue*>(),
    .type_codes = type_codes_slot.start_addr().cast_to<int*>(),
    .num_args = num_args,
  };
  utvm_args_slot.WriteValue(dev_args);
  return utvm_args_slot.start_addr();
}

DevAddr MicroSession::EncoderAppend(TargetDataLayoutEncoder* encoder, const TVMArray& arr) {
  auto tvm_arr_slot = encoder->Alloc<TVMArray>();
  auto shape_slot = encoder->Alloc<int64_t>(arr.ndim);

  // `shape` and `strides` are stored on the host, so we need to write them to
  // the device first. The `data` field is already allocated on the device and
  // is a device pointer, so we don't need to write it.
  shape_slot.WriteArray(arr.shape, arr.ndim);
  DevAddr shape_addr = shape_slot.start_addr();
  DevAddr strides_addr = DevAddr(nullptr);
  if (arr.strides != nullptr) {
    auto stride_slot = encoder->Alloc<int64_t>(arr.ndim);
    stride_slot.WriteArray(arr.strides, arr.ndim);
    strides_addr = stride_slot.start_addr();
  }

  // Copy `arr`, update the copy's pointers to be device pointers, then
  // write the copy to `tvm_arr_slot`.
  TVMArray dev_arr = arr;
  // Update the device type to look like a host, because codegen generates
  // checks that it is a host array.
  CHECK(dev_arr.ctx.device_type == static_cast<DLDeviceType>(kDLMicroDev))
    << "attempt to write TVMArray with non-micro device type";
  dev_arr.ctx.device_type = DLDeviceType::kDLCPU;
  // Add the base address of the device to the array's data's device offset to
  // get a device address.
  DevBaseOffset arr_offset(reinterpret_cast<std::uintptr_t>(arr.data));
  dev_arr.data = (low_level_device()->base_addr() + arr_offset).cast_to<void*>();
  dev_arr.shape = shape_addr.cast_to<int64_t*>();
  dev_arr.strides = strides_addr.cast_to<int64_t*>();
  tvm_arr_slot.WriteValue(dev_arr);
  return tvm_arr_slot.start_addr();
}

void MicroSession::CheckDeviceError() {
  int32_t return_code = DevSymbolRead<int32_t>(init_symbol_map(), "return_code");

  if (return_code) {
    std::uintptr_t last_error = DevSymbolRead<std::uintptr_t>(init_symbol_map(), "last_error");
    std::string last_error_str;
    if (last_error) {
      DevBaseOffset last_err_offset =
          DevAddr(last_error) - low_level_device()->base_addr();
      last_error_str = ReadString(last_err_offset);
    }
    LOG(FATAL) << "error during micro function execution:\n"
               << "  return code: " << std::dec << return_code << "\n"
               << "  dev str addr: 0x" << std::hex << last_error << "\n"
               << "  dev str data: " << last_error_str << std::endl;
  }
}

template <typename T>
T MicroSession::DevSymbolRead(SymbolMap& symbol_map, const std::string& symbol) {
  DevBaseOffset sym_offset = symbol_map[symbol];
  T result;
  low_level_device()->Read(sym_offset, &result, sizeof(T));
  return result;
}

template <typename T>
void MicroSession::DevSymbolWrite(SymbolMap& symbol_map, const std::string& symbol, T& value) {
  DevBaseOffset sym_offset = symbol_map[symbol];
  low_level_device()->Write(sym_offset, &value, sizeof(T));
}

// initializes micro session and low-level device from Python frontend
TVM_REGISTER_GLOBAL("micro._InitSession")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::shared_ptr<MicroSession> session = MicroSession::Global(true);
    session->InitSession(args);
    });

// ends micro session and destructs low-level device from Python frontend
TVM_REGISTER_GLOBAL("micro._EndSession")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::shared_ptr<MicroSession> session = MicroSession::Global();
    session->EndSession();
    });
}  // namespace runtime
}  // namespace tvm
