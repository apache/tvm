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
 */

#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <memory>
#include <stack>
#include <tuple>
#include <vector>
#include "micro_session.h"
#include "low_level_device.h"
#include "target_data_layout_encoder.h"

namespace tvm {
namespace runtime {

struct TVMMicroSessionThreadLocalEntry {
  std::stack<std::shared_ptr<MicroSession>> session_stack;
};

typedef dmlc::ThreadLocalStore<TVMMicroSessionThreadLocalEntry> TVMMicroSessionThreadLocalStore;

std::shared_ptr<MicroSession>& MicroSession::Current() {
  TVMMicroSessionThreadLocalEntry *entry = TVMMicroSessionThreadLocalStore::Get();
  CHECK_GT(entry->session_stack.size(), 0) << "No current session";
  return entry->session_stack.top();
}

void MicroSession::EnterWithScope(std::shared_ptr<MicroSession> session) {
  TVMMicroSessionThreadLocalEntry *entry = TVMMicroSessionThreadLocalStore::Get();
  entry->session_stack.push(session);
}

void MicroSession::ExitWithScope() {
  TVMMicroSessionThreadLocalEntry *entry = TVMMicroSessionThreadLocalStore::Get();
  CHECK(!entry->session_stack.empty());
  entry->session_stack.pop();
}

MicroSession::MicroSession() {
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
}

MicroSession::~MicroSession() {
  for (size_t i = 0; i < static_cast<size_t>(SectionKind::kNumKinds); i++) {
    section_allocators_[i] = nullptr;
  }
  low_level_device_ = nullptr;
}

void MicroSession::CreateSession(const std::string& device_type,
                                 const std::string& binary_path,
                                 const std::string& toolchain_prefix,
                                 std::uintptr_t base_addr,
                                 const std::string& server_addr,
                                 int port) {
  // TODO(weberlo): make device type enum
  toolchain_prefix_ = toolchain_prefix;
  if (device_type == "host") {
    low_level_device_ = HostLowLevelDeviceCreate(memory_size_);
  } else if (device_type == "openocd") {
    // TODO(weberlo): We need a better way of configuring devices.
    low_level_device_ = OpenOCDLowLevelDeviceCreate(base_addr, server_addr, port);
  } else {
    LOG(FATAL) << "unsupported micro low-level device";
  }

  SetRuntimeBinaryPath(binary_path);
  CHECK(!runtime_binary_path_.empty()) << "uTVM runtime not initialized";
  runtime_bin_info_ = LoadBinary(runtime_binary_path_, /* patch_dylib_pointers */ false);
  utvm_main_symbol_ = low_level_device()->ToDevOffset(runtime_symbol_map()["UTVMMain"]);
  utvm_done_symbol_ = low_level_device()->ToDevOffset(runtime_symbol_map()["UTVMDone"]);

  if (device_type == "openocd") {
    // Set OpenOCD device's stack pointer.
    auto stack_section = GetAllocator(SectionKind::kStack);
    low_level_device_->SetStackTop(stack_section->max_end_offset());
  }

  // Patch workspace pointers to the start of the workspace section.
  DevBaseOffset workspace_start_offset = GetAllocator(SectionKind::kWorkspace)->start_offset();
  DevBaseOffset workspace_end_offset = GetAllocator(SectionKind::kWorkspace)->max_end_offset();
  void* workspace_start_addr =
      low_level_device_->ToDevPtr(workspace_start_offset).cast_to<void*>();
  void* workspace_end_addr =
      low_level_device_->ToDevPtr(workspace_end_offset).cast_to<void*>();
  DevSymbolWrite(runtime_symbol_map(), "utvm_workspace_begin", workspace_start_addr);
  DevSymbolWrite(runtime_symbol_map(), "utvm_workspace_end", workspace_end_addr);
}

void MicroSession::PushToExecQueue(DevBaseOffset func, const TVMArgs& args) {
  int32_t (*func_dev_addr)(void*, void*, int32_t) =
      reinterpret_cast<int32_t (*)(void*, void*, int32_t)>(
      low_level_device()->ToDevPtr(func).value());

  // Create an allocator stream for the memory region after the most recent
  // allocation in the args section.
  DevPtr args_addr =
      low_level_device()->ToDevPtr(GetAllocator(SectionKind::kArgs)->curr_end_offset());
  TargetDataLayoutEncoder encoder(args_addr);

  std::tuple<DevPtr, DevPtr> arg_field_addrs = EncoderAppend(&encoder, args);
  // Flush `stream` to device memory.
  DevBaseOffset stream_dev_offset =
      GetAllocator(SectionKind::kArgs)->Allocate(encoder.buf_size());
  low_level_device()->Write(stream_dev_offset,
                            reinterpret_cast<void*>(encoder.data()),
                            encoder.buf_size());

  UTVMTask task = {
      .func = func_dev_addr,
      .arg_values = std::get<0>(arg_field_addrs).cast_to<TVMValue*>(),
      .arg_type_codes = std::get<1>(arg_field_addrs).cast_to<int*>(),
      .num_args = args.num_args,
  };
  // Write the task.
  DevSymbolWrite(runtime_symbol_map(), "task", task);

  low_level_device()->Execute(utvm_main_symbol_, utvm_done_symbol_);
  // Check if there was an error during execution.  If so, log it.
  CheckDeviceError();

  GetAllocator(SectionKind::kArgs)->Free(stream_dev_offset);
}

std::tuple<DevPtr, DevPtr> MicroSession::EncoderAppend(
    TargetDataLayoutEncoder* encoder, const TVMArgs& args) {
  const int* type_codes = args.type_codes;
  int num_args = args.num_args;

  auto tvm_vals_slot = encoder->Alloc<TVMValue>(num_args);
  auto type_codes_slot = encoder->Alloc<const int>(num_args);

  for (int i = 0; i < num_args; i++) {
    switch (type_codes[i]) {
      case kNDArrayContainer:
      case kArrayHandle: {
        TVMArray* base_arr_handle = args[i];
        // All uTVM arrays store a `MicroDevSpace` struct in their `data` field,
        // which wraps the actual data and stores a reference to the session, in
        // order to prevent premature session destruction.
        void* old_data = base_arr_handle->data;
        // Mutate the array to unwrap the `data` field.
        base_arr_handle->data = reinterpret_cast<MicroDevSpace*>(old_data)->data;
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
        LOG(FATAL) << "unsupported type code for writing args: " << type_codes[i];
        break;
    }
  }
  type_codes_slot.WriteArray(type_codes, num_args);

  return std::make_tuple(tvm_vals_slot.start_addr(), type_codes_slot.start_addr());
}

DevPtr MicroSession::EncoderAppend(TargetDataLayoutEncoder* encoder, const TVMArray& arr) {
  auto tvm_arr_slot = encoder->Alloc<TVMArray>();
  auto shape_slot = encoder->Alloc<int64_t>(arr.ndim);

  // `shape` and `strides` are stored on the host, so we need to write them to
  // the device first. The `data` field is already allocated on the device and
  // is a device pointer, so we don't need to write it.
  shape_slot.WriteArray(arr.shape, arr.ndim);
  DevPtr shape_addr = shape_slot.start_addr();
  DevPtr strides_addr = DevPtr(nullptr);
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
  dev_arr.data = low_level_device()->ToDevPtr(arr_offset).cast_to<void*>();
  dev_arr.shape = shape_addr.cast_to<int64_t*>();
  dev_arr.strides = strides_addr.cast_to<int64_t*>();
  tvm_arr_slot.WriteValue(dev_arr);
  return tvm_arr_slot.start_addr();
}

void MicroSession::CheckDeviceError() {
  int32_t return_code = DevSymbolRead<int32_t>(runtime_symbol_map(), "utvm_return_code");

  if (return_code) {
    std::uintptr_t last_error =
        DevSymbolRead<std::uintptr_t>(runtime_symbol_map(), "utvm_last_error");
    std::string last_error_str;
    if (last_error) {
      DevBaseOffset last_err_offset = low_level_device()->ToDevOffset(DevPtr(last_error));
      last_error_str = ReadString(last_err_offset);
    }
    LOG(FATAL) << "error during micro function execution:\n"
               << "  return code: " << std::dec << return_code << "\n"
               << "  dev str addr: 0x" << std::hex << last_error << "\n"
               << "  dev str data: " << last_error_str << std::endl;
  }
}

BinaryInfo MicroSession::LoadBinary(const std::string& binary_path, bool patch_dylib_pointers) {
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
  CHECK(text_section.start != nullptr && rodata_section.start != nullptr &&
        data_section.start != nullptr && bss_section.start != nullptr)
      << "not enough space to load module on device";

  std::string relocated_bin = RelocateBinarySections(
      binary_path,
      low_level_device_->ToDevPtr(text_section.start),
      low_level_device_->ToDevPtr(rodata_section.start),
      low_level_device_->ToDevPtr(data_section.start),
      low_level_device_->ToDevPtr(bss_section.start),
      toolchain_prefix_);
  std::string text_contents = ReadSection(relocated_bin, SectionKind::kText, toolchain_prefix_);
  std::string rodata_contents = ReadSection(relocated_bin, SectionKind::kRodata, toolchain_prefix_);
  std::string data_contents = ReadSection(relocated_bin, SectionKind::kData, toolchain_prefix_);
  std::string bss_contents = ReadSection(relocated_bin, SectionKind::kBss, toolchain_prefix_);
  low_level_device_->Write(text_section.start, &text_contents[0], text_section.size);
  low_level_device_->Write(rodata_section.start, &rodata_contents[0], rodata_section.size);
  low_level_device_->Write(data_section.start, &data_contents[0], data_section.size);
  low_level_device_->Write(bss_section.start, &bss_contents[0], bss_section.size);
  SymbolMap symbol_map {relocated_bin, toolchain_prefix_};

  if (patch_dylib_pointers) {
    // Patch device lib pointers.
    PatchImplHole(symbol_map, "TVMBackendAllocWorkspace");
    PatchImplHole(symbol_map, "TVMBackendFreeWorkspace");
    PatchImplHole(symbol_map, "TVMAPISetLastError");
  }

  return BinaryInfo {
      .text_section = text_section,
      .rodata_section = rodata_section,
      .data_section = data_section,
      .bss_section = bss_section,
      .symbol_map = symbol_map,
  };
}

void MicroSession::PatchImplHole(const SymbolMap& symbol_map, const std::string& func_name) {
  void* runtime_impl_addr = runtime_symbol_map()[func_name].cast_to<void*>();
  std::ostringstream func_name_underscore;
  func_name_underscore << func_name << "_";
  DevSymbolWrite(symbol_map, func_name_underscore.str(), runtime_impl_addr);
}

void MicroSession::SetRuntimeBinaryPath(std::string path) {
  runtime_binary_path_ = path;
}

std::string MicroSession::ReadString(DevBaseOffset str_offset) {
  std::ostringstream result;
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

DevBaseOffset MicroSession::AllocateInSection(SectionKind type, size_t size) {
  return GetAllocator(type)->Allocate(size);
}

void MicroSession::FreeInSection(SectionKind type, DevBaseOffset ptr) {
  return GetAllocator(type)->Free(ptr);
}

template <typename T>
T MicroSession::DevSymbolRead(const SymbolMap& symbol_map, const std::string& symbol) {
  DevBaseOffset sym_offset = low_level_device()->ToDevOffset(symbol_map[symbol]);
  T result;
  low_level_device()->Read(sym_offset, &result, sizeof(T));
  return result;
}

template <typename T>
void MicroSession::DevSymbolWrite(const SymbolMap& symbol_map,
                                  const std::string& symbol,
                                  const T& value) {
  DevBaseOffset sym_offset = low_level_device()->ToDevOffset(symbol_map[symbol]);
  low_level_device()->Write(sym_offset, &value, sizeof(T));
}

PackedFunc MicroSession::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  if (name == "enter") {
    return PackedFunc([sptr_to_self](TVMArgs args, TVMRetValue* rv) {
      MicroSession::EnterWithScope(std::dynamic_pointer_cast<MicroSession>(sptr_to_self));
    });
  } else if (name == "exit") {
    return PackedFunc([sptr_to_self](TVMArgs args, TVMRetValue* rv) {
      MicroSession::ExitWithScope();
    });
  } else {
    return PackedFunc();
  }
}

// create micro session and low-level device from Python frontend
TVM_REGISTER_GLOBAL("micro._CreateSession")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    const std::string& device_type = args[0];
    const std::string& binary_path = args[1];
    const std::string& toolchain_prefix = args[2];
    uint64_t base_addr = args[3];
    const std::string& server_addr = args[4];
    int port = args[5];
    std::shared_ptr<MicroSession> session = std::make_shared<MicroSession>();
    session->CreateSession(
        device_type, binary_path, toolchain_prefix, base_addr, server_addr, port);
    *rv = Module(session);
    });

}  // namespace runtime
}  // namespace tvm
