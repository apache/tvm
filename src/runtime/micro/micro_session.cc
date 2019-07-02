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
#include "target_data_layout_encoder.h"

namespace tvm {
namespace runtime {

MicroSession::MicroSession() : valid_(false) { }

MicroSession::~MicroSession() { }

void MicroSession::InitSession(const TVMArgs& args) {
  valid_ = true;

  text_allocator_ = std::unique_ptr<MicroSectionAllocator>(
      new MicroSectionAllocator(kTextStart,
                                kRodataStart));
  rodata_allocator_ = std::unique_ptr<MicroSectionAllocator>(
      new MicroSectionAllocator(kRodataStart,
                                kDataStart));
  data_allocator_ = std::unique_ptr<MicroSectionAllocator>(
      new MicroSectionAllocator(kDataStart,
                                kBssStart));
  bss_allocator_ = std::unique_ptr<MicroSectionAllocator>(
      new MicroSectionAllocator(kBssStart,
                                kArgsStart));
  args_allocator_ = std::unique_ptr<MicroSectionAllocator>(
      new MicroSectionAllocator(kArgsStart,
                                kStackStart));
  stack_allocator_ = std::unique_ptr<MicroSectionAllocator>(
      new MicroSectionAllocator(kStackStart,
                                kHeapStart));
  heap_allocator_ = std::unique_ptr<MicroSectionAllocator>(
      new MicroSectionAllocator(kHeapStart,
                                kWorkspaceStart));

  const std::string& device_type = args[0];
  const std::string& binary_path = args[1];
  SetInitBinaryPath(binary_path);
  if (device_type == "host") {
    low_level_device_ = HostLowLevelDeviceCreate(kMemorySize);
  } else if (device_type == "openocd") {
    int port = args[2];
    low_level_device_ = OpenOCDLowLevelDeviceCreate(port);
  } else {
    LOG(FATAL) << "Unsupported micro low-level device";
  }
  CHECK(!init_binary_path_.empty()) << "init library not initialized";
  init_stub_info_ = LoadBinary(init_binary_path_);
  utvm_main_symbol_addr_ = init_stub_info_.symbol_map["UTVMMain"];
  utvm_done_symbol_addr_ = init_stub_info_.symbol_map["UTVMDone"];

  // Patch workspace pointers to the start of the workspace section.
  DevBaseOffset workspace_start_hole_offset = init_symbol_map()["utvm_workspace_begin"];
  DevBaseOffset workspace_curr_hole_offset = init_symbol_map()["utvm_workspace_curr"];
  DevBaseOffset workspace_start(kWorkspaceStart.value());
  void* workspace_hole_fill =
      (workspace_start + low_level_device_->base_addr().value()).cast_to<void*>();
  low_level_device()->Write(workspace_start_hole_offset, &workspace_hole_fill, sizeof(void*));
  low_level_device()->Write(workspace_curr_hole_offset, &workspace_hole_fill, sizeof(void*));
}

void MicroSession::EndSession() {
  valid_ = false;

  text_allocator_ = nullptr;
  rodata_allocator_ = nullptr;
  data_allocator_ = nullptr;
  bss_allocator_ = nullptr;
  args_allocator_ = nullptr;
  stack_allocator_ = nullptr;
  heap_allocator_ = nullptr;

  low_level_device_ = nullptr;
}

DevBaseOffset MicroSession::AllocateInSection(SectionKind type, size_t size) {
  switch (type) {
    case SectionKind::kText:
      return text_allocator_->Allocate(size);
    case SectionKind::kRodata:
      return rodata_allocator_->Allocate(size);
    case SectionKind::kData:
      return data_allocator_->Allocate(size);
    case SectionKind::kBss:
      return bss_allocator_->Allocate(size);
    case SectionKind::kArgs:
      return args_allocator_->Allocate(size);
    case SectionKind::kStack:
      return stack_allocator_->Allocate(size);
    case SectionKind::kHeap:
      return heap_allocator_->Allocate(size);
    default:
      LOG(FATAL) << "Unsupported section type during allocation";
      return DevBaseOffset(nullptr);
  }
}

void MicroSession::FreeInSection(SectionKind type, DevBaseOffset ptr) {
  switch (type) {
    case SectionKind::kText:
      text_allocator_->Free(ptr);
      return;
    case SectionKind::kRodata:
      rodata_allocator_->Free(ptr);
      return;
    case SectionKind::kData:
      data_allocator_->Free(ptr);
      return;
    case SectionKind::kBss:
      bss_allocator_->Free(ptr);
      return;
    case SectionKind::kArgs:
      args_allocator_->Free(ptr);
      return;
    case SectionKind::kStack:
      stack_allocator_->Free(ptr);
      return;
    case SectionKind::kHeap:
      heap_allocator_->Free(ptr);
      return;
    default:
      LOG(FATAL) << "Unsupported section type during free";
  }
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
  void (*func_dev_addr)(void*, void*, int32_t) =
      reinterpret_cast<void (*)(void*, void*, int32_t)>(
      (func + low_level_device()->base_addr()).value());

  // Create an allocator stream for the memory region after the most recent
  // allocation in the args section.
  DevAddr args_addr = args_allocator_->section_max() + low_level_device()->base_addr();
  TargetDataLayoutEncoder encoder(args_addr);

  EncoderAppend(&encoder, args);
  // Flush `stream` to device memory.
  DevBaseOffset stream_dev_offset = args_allocator_->Allocate(encoder.buf_size());
  low_level_device()->Write(stream_dev_offset,
                            reinterpret_cast<void*>(encoder.data()),
                            encoder.buf_size());

  UTVMTask task = {
      .func = func_dev_addr,
      .args = args_addr.cast_to<UTVMArgs*>(),
  };
  // TODO(mutinifni): handle bits / endianness
  // Write the task.
  low_level_device()->Write(init_symbol_map()["task"], &task, sizeof(task));
  // Zero out the last error.
  std::uintptr_t last_error = 0;
  low_level_device()->Write(init_symbol_map()["last_error"], &last_error, sizeof(std::uintptr_t));

  low_level_device()->Execute(utvm_main_symbol_addr_, utvm_done_symbol_addr_);

  // Check if there was an error during execution.  If so, log it.
  CheckDeviceError();
}

BinaryInfo MicroSession::LoadBinary(std::string binary_path) {
  SectionLocation text;
  SectionLocation rodata;
  SectionLocation data;
  SectionLocation bss;

  text.size = GetSectionSize(binary_path, SectionKind::kText);
  rodata.size = GetSectionSize(binary_path, SectionKind::kRodata);
  data.size = GetSectionSize(binary_path, SectionKind::kData);
  bss.size = GetSectionSize(binary_path, SectionKind::kBss);

  text.start = AllocateInSection(SectionKind::kText, text.size);
  rodata.start = AllocateInSection(SectionKind::kRodata, rodata.size);
  data.start = AllocateInSection(SectionKind::kData, data.size);
  bss.start = AllocateInSection(SectionKind::kBss, bss.size);
  CHECK(text.start != nullptr && rodata.start != nullptr && data.start != nullptr &&
        bss.start != nullptr) << "not enough space to load module on device";
  const DevBaseAddr base_addr = low_level_device_->base_addr();
  std::string relocated_bin = RelocateBinarySections(
      binary_path,
      text.start + base_addr,
      rodata.start + base_addr,
      data.start + base_addr,
      bss.start + base_addr);
  std::string text_contents = ReadSection(relocated_bin, SectionKind::kText);
  std::string rodata_contents = ReadSection(relocated_bin, SectionKind::kRodata);
  std::string data_contents = ReadSection(relocated_bin, SectionKind::kData);
  std::string bss_contents = ReadSection(relocated_bin, SectionKind::kBss);
  low_level_device_->Write(text.start, &text_contents[0], text.size);
  low_level_device_->Write(rodata.start, &rodata_contents[0], rodata.size);
  low_level_device_->Write(data.start, &data_contents[0], data.size);
  low_level_device_->Write(bss.start, &bss_contents[0], bss.size);
  SymbolMap symbol_map {relocated_bin, base_addr};
  return BinaryInfo{
      .text = text,
      .rodata = rodata,
      .data = data,
      .bss = bss,
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
  DevBaseOffset last_err_offset = init_symbol_map()["last_error"];
  std::uintptr_t last_error;
  low_level_device()->Read(last_err_offset, &last_error, sizeof(std::uintptr_t));
  if (last_error) {
    // First, retrieve the string `last_error` points to.
    std::uintptr_t last_err_data_addr;
    low_level_device()->Read(last_err_offset, &last_err_data_addr, sizeof(std::uintptr_t));
    DevBaseOffset last_err_data_offset =
        DevAddr(last_err_data_addr) - low_level_device()->base_addr();
    // Then read the string from device to host and log it.
    std::string last_error_str = ReadString(last_err_data_offset);
    LOG(FATAL) << "error during micro function execution:\n"
               << "  dev str addr: 0x" << std::hex << last_err_data_addr << "\n"
               << "  dev str data: " << last_error_str;
  }
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
