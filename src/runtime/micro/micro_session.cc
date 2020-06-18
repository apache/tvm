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
  std::stack<ObjectPtr<MicroSession>> session_stack;
};

typedef dmlc::ThreadLocalStore<TVMMicroSessionThreadLocalEntry> TVMMicroSessionThreadLocalStore;

ObjectPtr<MicroSession>& MicroSession::Current() {
  TVMMicroSessionThreadLocalEntry *entry = TVMMicroSessionThreadLocalStore::Get();
  CHECK_GT(entry->session_stack.size(), 0) << "No current session";
  return entry->session_stack.top();
}

void MicroSession::EnterWithScope(ObjectPtr<MicroSession> session) {
  TVMMicroSessionThreadLocalEntry *entry = TVMMicroSessionThreadLocalStore::Get();
  entry->session_stack.push(session);
}

void MicroSession::ExitWithScope() {
  TVMMicroSessionThreadLocalEntry *entry = TVMMicroSessionThreadLocalStore::Get();
  CHECK(!entry->session_stack.empty());
  entry->session_stack.pop();
}

MicroSession::MicroSession(
    const std::string& comms_method,
    const std::string& binary_path,
    const std::string& toolchain_prefix,
    uint64_t text_start,
    size_t text_size,
    uint64_t rodata_start,
    size_t rodata_size,
    uint64_t data_start,
    size_t data_size,
    uint64_t bss_start,
    size_t bss_size,
    uint64_t args_start,
    size_t args_size,
    uint64_t heap_start,
    size_t heap_size,
    uint64_t workspace_start,
    size_t workspace_size,
    uint64_t stack_start,
    size_t stack_size,
    size_t word_size,
    bool thumb_mode,
    const std::string& server_addr,
    int port)
    : toolchain_prefix_(toolchain_prefix)
    , word_size_(word_size)
    , thumb_mode_(thumb_mode) {
  CHECK(word_size_ == 4 || word_size_ == 8) << "unsupported word size " << word_size_;
  if (comms_method == "host") {
    // TODO(weberlo): move checks to python
    CHECK(
        text_start == 0 &&
        rodata_start == 0 &&
        data_start == 0 &&
        bss_start == 0 &&
        args_start == 0 &&
        heap_start == 0 &&
        workspace_start == 0 &&
        stack_start == 0) << "unable to specify section addresses for host device";
    size_t memory_size =
      text_size + rodata_size + data_size + bss_size +
      args_size + heap_size + workspace_size + stack_size;
    void* base_addr;
    low_level_device_ = HostLowLevelDeviceCreate(memory_size, &base_addr);
    CHECK_EQ(reinterpret_cast<std::uintptr_t>(base_addr) % word_size_, 0)
      << "base address not aligned to " << word_size_ << " bytes";
    DevPtr curr_addr = DevPtr(reinterpret_cast<std::uintptr_t>(base_addr));

    section_allocators_[0] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = curr_addr,
      .size = text_size,
    }, word_size_);
    curr_addr += text_size;
    section_allocators_[1] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = curr_addr,
      .size = rodata_size,
    }, word_size_);
    curr_addr += rodata_size;
    section_allocators_[2] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = curr_addr,
      .size = data_size,
    }, word_size_);
    curr_addr += data_size;
    section_allocators_[3] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = curr_addr,
      .size = bss_size,
    }, word_size_);
    curr_addr += bss_size;
    section_allocators_[4] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = curr_addr,
      .size = args_size,
    }, word_size_);
    curr_addr += args_size;
    section_allocators_[5] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = curr_addr,
      .size = heap_size,
    }, word_size_);
    curr_addr += heap_size;
    section_allocators_[6] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = curr_addr,
      .size = workspace_size,
    }, word_size_);
    curr_addr += workspace_size;
    section_allocators_[7] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = curr_addr,
      .size = stack_size,
    }, word_size_);
    curr_addr += stack_size;
  } else if (comms_method == "openocd") {
    low_level_device_ = OpenOCDLowLevelDeviceCreate(server_addr, port);
    section_allocators_[0] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = DevPtr(text_start),
      .size = text_size,
    }, word_size_);
    section_allocators_[1] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = DevPtr(rodata_start),
      .size = rodata_size,
    }, word_size_);
    section_allocators_[2] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = DevPtr(data_start),
      .size = data_size,
    }, word_size_);
    section_allocators_[3] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = DevPtr(bss_start),
      .size = bss_size,
    }, word_size_);
    section_allocators_[4] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = DevPtr(args_start),
      .size = args_size,
    }, word_size_);
    section_allocators_[5] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = DevPtr(heap_start),
      .size = heap_size,
    }, word_size_);
    section_allocators_[6] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = DevPtr(workspace_start),
      .size = workspace_size,
    }, word_size_);
    section_allocators_[7] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = DevPtr(stack_start),
      .size = stack_size,
    }, word_size_);
  } else {
    LOG(FATAL) << "unsupported micro low-level device";
  }

  runtime_symbol_map_ = LoadBinary(binary_path, false).symbol_map;

  // Patch pointers to define the bounds of the workspace section and the word
  // size (for allocation alignment).
  std::shared_ptr<MicroSectionAllocator> ws_allocator = GetAllocator(SectionKind::kWorkspace);
  TargetVal ws_start = ws_allocator->start_addr().value();
  TargetVal ws_end = ws_allocator->max_addr().value();
  TargetVal target_word_size { .val64 = word_size_ };
  if (word_size_ == 4) {
    DevSymbolWrite(runtime_symbol_map_, "utvm_workspace_start", ws_start.val32);
    DevSymbolWrite(runtime_symbol_map_, "utvm_workspace_end", ws_end.val32);
    DevSymbolWrite(runtime_symbol_map_, "utvm_word_size", target_word_size.val32);
  } else if (word_size_ == 8) {
    DevSymbolWrite(runtime_symbol_map_, "utvm_workspace_start", ws_start.val64);
    DevSymbolWrite(runtime_symbol_map_, "utvm_workspace_end", ws_end.val64);
    DevSymbolWrite(runtime_symbol_map_, "utvm_word_size", target_word_size.val64);
  }
}

MicroSession::~MicroSession() {
  for (size_t i = 0; i < static_cast<size_t>(SectionKind::kNumKinds); i++) {
    section_allocators_[i] = nullptr;
  }
  low_level_device_ = nullptr;
}

double MicroSession::PushToExecQueue(DevPtr func_ptr, const TVMArgs& args) {
  if (thumb_mode_) {
    func_ptr += 1;
  }

  // Create an allocator stream for the memory region after the most recent
  // allocation in the args section.
  DevPtr args_addr = GetAllocator(SectionKind::kArgs)->curr_end_addr();
  TargetDataLayoutEncoder encoder(args_addr, word_size_);

  std::tuple<DevPtr, DevPtr> arg_field_addrs = EncoderAppend(&encoder, args);

  // Flush `stream` to device memory.
  DevPtr stream_dev_addr =
      GetAllocator(SectionKind::kArgs)->Allocate(encoder.buf_size());
  low_level_device()->Write(stream_dev_addr,
                            reinterpret_cast<void*>(encoder.data()),
                            encoder.buf_size());

  TargetVal arg_values_dev_addr = std::get<0>(arg_field_addrs).value();
  TargetVal arg_type_codes_dev_addr = std::get<1>(arg_field_addrs).value();
  if (word_size_ == 4) {
    UTVMTask32 task = {
      .func = func_ptr.value().val32,
      .arg_values = arg_values_dev_addr.val32,
      .arg_type_codes = arg_type_codes_dev_addr.val32,
      .num_args = args.num_args,
    };
    // Write the task.
    DevSymbolWrite(runtime_symbol_map_, "utvm_task", task);
  } else if (word_size_ == 8) {
    UTVMTask64 task = {
      .func = func_ptr.value().val64,
      .arg_values = arg_values_dev_addr.val64,
      .arg_type_codes = arg_type_codes_dev_addr.val64,
      .num_args = args.num_args,
    };
    // Write the task.
    DevSymbolWrite(runtime_symbol_map_, "utvm_task", task);
  }

  DevPtr utvm_init_addr = runtime_symbol_map_["UTVMInit"];
  DevPtr utvm_done_addr = runtime_symbol_map_["UTVMDone"];
  if (thumb_mode_) {
    utvm_init_addr += 1;
  }

  low_level_device()->Execute(utvm_init_addr, utvm_done_addr);
  // Check if there was an error during execution.  If so, log it.
  CheckDeviceError();
  uint32_t task_time = DevSymbolRead<uint32_t>(runtime_symbol_map_, "utvm_task_time");
  GetAllocator(SectionKind::kArgs)->Free(stream_dev_addr);
  return static_cast<double>(task_time);
}

BinaryInfo MicroSession::LoadBinary(const std::string& binary_path, bool patch_dylib_pointers) {
  DevMemRegion text_section;
  DevMemRegion rodata_section;
  DevMemRegion data_section;
  DevMemRegion bss_section;

  text_section.size = GetSectionSize(
      binary_path, SectionKind::kText, toolchain_prefix_, word_size_);
  rodata_section.size = GetSectionSize(
      binary_path, SectionKind::kRodata, toolchain_prefix_, word_size_);
  data_section.size = GetSectionSize(
      binary_path, SectionKind::kData, toolchain_prefix_, word_size_);
  bss_section.size = GetSectionSize(
      binary_path, SectionKind::kBss, toolchain_prefix_, word_size_);

  text_section.start = AllocateInSection(SectionKind::kText, text_section.size);
  rodata_section.start = AllocateInSection(SectionKind::kRodata, rodata_section.size);
  data_section.start = AllocateInSection(SectionKind::kData, data_section.size);
  bss_section.start = AllocateInSection(SectionKind::kBss, bss_section.size);
  CHECK(text_section.start != nullptr && rodata_section.start != nullptr &&
        data_section.start != nullptr && bss_section.start != nullptr)
      << "not enough space to load module on device";

  std::string relocated_bin = RelocateBinarySections(
      binary_path,
      word_size_,
      text_section.start,
      rodata_section.start,
      data_section.start,
      bss_section.start,
      GetAllocator(SectionKind::kStack)->max_addr(),
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

std::tuple<DevPtr, DevPtr> MicroSession::EncoderAppend(
    TargetDataLayoutEncoder* encoder, const TVMArgs& args) {
  const int* type_codes = args.type_codes;
  int num_args = args.num_args;

  auto tvm_vals_slot = encoder->Alloc<TVMValue>(num_args);
  auto type_codes_slot = encoder->Alloc<const int>(num_args);

  for (int i = 0; i < num_args; i++) {
    switch (type_codes[i]) {
      case kTVMNDArrayHandle:
      case kTVMDLTensorHandle: {
        DLTensor* base_arr_handle = args[i];
        // All uTVM arrays store a `MicroDevSpace` struct in their `data` field,
        // which wraps the actual data and stores a reference to the session, in
        // order to prevent premature session destruction.
        void* old_data = base_arr_handle->data;
        // Mutate the array to unwrap the `data` field.
        base_arr_handle->data = reinterpret_cast<MicroDevSpace*>(old_data)->data;
        // Now, encode the unwrapped version.
        void* arr_ptr = nullptr;
        if (word_size_ == 4) {
          arr_ptr = EncoderAppend<TVMArray32>(encoder, *base_arr_handle).cast_to<void*>();
        } else if (word_size_ == 8) {
          arr_ptr = EncoderAppend<TVMArray64>(encoder, *base_arr_handle).cast_to<void*>();
        }
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

template <typename T>
DevPtr MicroSession::EncoderAppend(TargetDataLayoutEncoder* encoder, const DLTensor& arr) {
  auto tvm_arr_slot = encoder->Alloc<T>();
  auto shape_slot = encoder->Alloc<int64_t>(arr.ndim);

  // `shape` and `strides` are stored on the host, so we need to write them to
  // the device first. The `data` field is already allocated on the device and
  // is a device pointer, so we don't need to write it.
  shape_slot.WriteArray(arr.shape, arr.ndim);
  DevPtr shape_dev_addr = shape_slot.start_addr();
  DevPtr strides_dev_addr = DevPtr(nullptr);
  if (arr.strides != nullptr) {
    auto stride_slot = encoder->Alloc<int64_t>(arr.ndim);
    stride_slot.WriteArray(arr.strides, arr.ndim);
    strides_dev_addr = stride_slot.start_addr();
  }

  T dev_arr(
      TargetVal { .val64 = reinterpret_cast<uint64_t>(arr.data) },
      arr.ctx,
      arr.ndim,
      arr.dtype,
      shape_dev_addr.value(),
      strides_dev_addr.value(),
      TargetVal { .val64 = arr.byte_offset });
  CHECK(dev_arr.ctx.device_type == static_cast<DLDeviceType>(kDLMicroDev))
    << "attempt to write DLTensor with non-micro device type";
  // Update the device type to CPU, because from the microcontroller's
  // perspective, it is.
  dev_arr.ctx.device_type = DLDeviceType::kDLCPU;
  tvm_arr_slot.WriteValue(dev_arr);
  return tvm_arr_slot.start_addr();
}

void MicroSession::CheckDeviceError() {
  int32_t return_code = DevSymbolRead<int32_t>(runtime_symbol_map_, "utvm_return_code");

  if (return_code) {
    std::uintptr_t last_error =
        DevSymbolRead<std::uintptr_t>(runtime_symbol_map_, "utvm_last_error");
    std::string last_error_str;
    if (last_error) {
      DevPtr last_err_addr = DevPtr(last_error);
      last_error_str = ReadString(last_err_addr);
    }
    LOG(FATAL) << "error during micro function execution:\n"
               << "  return code: " << std::dec << return_code << "\n"
               << "  dev str addr: 0x" << std::hex << last_error << "\n"
               << "  dev str data: " << last_error_str << std::endl;
  }
}

void MicroSession::PatchImplHole(const SymbolMap& symbol_map, const std::string& func_name) {
  DevPtr runtime_impl_addr = runtime_symbol_map_[func_name];
  if (thumb_mode_) {
    runtime_impl_addr += 1;
  }
  std::ostringstream func_name_underscore;
  func_name_underscore << func_name << "_";
  if (word_size_ == 4) {
    DevSymbolWrite(symbol_map, func_name_underscore.str(), runtime_impl_addr.value().val32);
  } else if (word_size_ == 8) {
    DevSymbolWrite(symbol_map, func_name_underscore.str(), runtime_impl_addr.value().val64);
  }
}

std::string MicroSession::ReadString(DevPtr str_addr) {
  std::ostringstream result;
  const size_t buf_size = 256;
  std::vector<char> buf(buf_size, 0);
  size_t i = buf_size;
  while (i == buf_size) {
    low_level_device()->Read(str_addr, buf.data(), buf_size);
    i = 0;
    while (i < buf_size) {
      if (buf[i] == 0) break;
      result << buf[i];
      i++;
    }
    str_addr = str_addr + i;
  }
  return result.str();
}

DevPtr MicroSession::AllocateInSection(SectionKind type, size_t size) {
  return GetAllocator(type)->Allocate(size);
}

void MicroSession::FreeInSection(SectionKind type, DevPtr addr) {
  return GetAllocator(type)->Free(addr);
}

template <typename T>
T MicroSession::DevSymbolRead(const SymbolMap& symbol_map, const std::string& symbol) {
  DevPtr sym_addr = symbol_map[symbol];
  T result;
  low_level_device()->Read(sym_addr, &result, sizeof(T));
  return result;
}

template <typename T>
void MicroSession::DevSymbolWrite(const SymbolMap& symbol_map,
                                  const std::string& symbol,
                                  const T& value) {
  DevPtr sym_addr = symbol_map[symbol];
  low_level_device()->Write(sym_addr, &value, sizeof(T));
}

PackedFunc MicroSession::GetFunction(
    const std::string& name,
    const ObjectPtr<Object>& sptr_to_self) {
  if (name == "enter") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      MicroSession::EnterWithScope(GetObjectPtr<MicroSession>(this));
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
    const std::string& comms_method = args[0];
    const std::string& binary_path = args[1];
    const std::string& toolchain_prefix = args[2];
    uint64_t text_start = args[3];
    size_t text_size = args[4];
    uint64_t rodata_start = args[5];
    size_t rodata_size = args[6];
    uint64_t data_start = args[7];
    size_t data_size = args[8];
    uint64_t bss_start = args[9];
    size_t bss_size = args[10];
    uint64_t args_start = args[11];
    size_t args_size = args[12];
    uint64_t heap_start = args[13];
    size_t heap_size = args[14];
    uint64_t workspace_start = args[15];
    size_t workspace_size = args[16];
    uint64_t stack_start = args[17];
    size_t stack_size = args[18];
    size_t word_size = args[19];
    bool thumb_mode = args[20];
    const std::string& server_addr = args[21];
    int port = args[22];
    ObjectPtr<MicroSession> session = make_object<MicroSession>(
        comms_method,
        binary_path,
        toolchain_prefix,
        text_start,
        text_size,
        rodata_start,
        rodata_size,
        data_start,
        data_size,
        bss_start,
        bss_size,
        args_start,
        args_size,
        heap_start,
        heap_size,
        workspace_start,
        workspace_size,
        stack_start,
        stack_size,
        word_size,
        thumb_mode,
        server_addr,
        port);
    *rv = Module(session);
    });

}  // namespace runtime
}  // namespace tvm
