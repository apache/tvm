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
#include <tvm/node/container.h>
#include <tvm/ir.h>
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
    int port) : word_size_(word_size), thumb_mode_(thumb_mode) {
  //DevBaseOffset curr_start_offset = DevBaseOffset(0x20000180);
  //DevBaseOffset curr_start_offset = DevBaseOffset(0x200001c8);
  //for (size_t i = 0; i < static_cast<size_t>(SectionKind::kNumKinds); i++) {
  //  size_t section_size = GetDefaultSectionSize(static_cast<SectionKind>(i));
  //  section_allocators_[i] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
  //    .start = curr_start_offset,
  //    .size = section_size,
  //  });
  //  curr_start_offset += section_size;
  //}

  //CHECK(curr_start_offset.value() < 0x20050000) << "exceeded available RAM on device (" << std::endl;

  // NOTE: we don't use this for openocd
  memory_size_ = 0;
  //memory_size_ = curr_start_offset.cast_to<size_t>();
  // TODO(weberlo): make device type enum
  toolchain_prefix_ = toolchain_prefix;
  if (comms_method == "host") {
    CHECK(
        text_start == 0 &&
        rodata_start == 0 &&
        data_start == 0 &&
        bss_start == 0 &&
        args_start == 0 &&
        heap_start == 0 &&
        workspace_start == 0 &&
        stack_start == 0) << "unable to specify section addresses for host device";
    size_t memory_size = text_size + rodata_size + data_size + bss_size + args_size + heap_size + workspace_size + stack_size;
    void* base_addr;
    low_level_device_ = HostLowLevelDeviceCreate(memory_size, &base_addr);
    CHECK(reinterpret_cast<std::uintptr_t>(base_addr) % word_size_ == 0) << "base address not aligned to " << word_size_ << " bytes";
    std::cout << "base addr is " << base_addr << std::endl;
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
    // TODO(weberlo): We need a better way of configuring devices.
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

  std::cout << "[Memory Layout]" << std::endl;
  std::cout << "  text (size = " << (section_allocators_[0]->capacity() / 1000.0) << " KB): " << section_allocators_[0]->start_addr().cast_to<void*>() << std::endl;
  std::cout << "  rodata (size = " << (section_allocators_[1]->capacity() / 1000.0) << " KB): " << section_allocators_[1]->start_addr().cast_to<void*>() << std::endl;
  std::cout << "  data (size = " << (section_allocators_[2]->capacity() / 1000.0) << " KB): " << section_allocators_[2]->start_addr().cast_to<void*>() << std::endl;
  std::cout << "  bss (size = " << (section_allocators_[3]->capacity() / 1000.0) << " KB): " << section_allocators_[3]->start_addr().cast_to<void*>() << std::endl;
  std::cout << "  args (size = " << (section_allocators_[4]->capacity() / 1000.0) << " KB): " << section_allocators_[4]->start_addr().cast_to<void*>() << std::endl;
  std::cout << "  heap (size = " << (section_allocators_[5]->capacity() / 1000.0) << " KB): " << section_allocators_[5]->start_addr().cast_to<void*>() << std::endl;
  std::cout << "  workspace (size = " << (section_allocators_[6]->capacity() / 1000.0) << " KB): " << section_allocators_[6]->start_addr().cast_to<void*>() << std::endl;
  std::cout << "  stack (size = " << (section_allocators_[7]->capacity() / 1000.0) << " KB): " << section_allocators_[7]->start_addr().cast_to<void*>() << std::endl;

  runtime_symbol_map_ = LoadBinary(binary_path, false).symbol_map;
  std::cout << runtime_symbol_map_["UTVMMain"].cast_to<void*>() << std::endl;
  std::cout << runtime_symbol_map_["utvm_task"].cast_to<void*>() << std::endl;

  DevSymbolWrite(runtime_symbol_map_, "utvm_word_size", word_size_);
  // Patch workspace pointers to the start of the workspace section.
  void* workspace_start_addr = GetAllocator(SectionKind::kWorkspace)->start_addr().cast_to<void*>();
  void* workspace_end_addr = GetAllocator(SectionKind::kWorkspace)->max_addr().cast_to<void*>();
  // TODO(weberlo): A lot of these symbol writes can be converted into symbols
  // in the C source, where the symbols are created by the linker script we
  // generate in python.
  DevSymbolWrite(runtime_symbol_map_, "utvm_workspace_start", workspace_start_addr);
  DevSymbolWrite(runtime_symbol_map_, "utvm_workspace_end", workspace_end_addr);
}

MicroSession::~MicroSession() {
  for (size_t i = 0; i < static_cast<size_t>(SectionKind::kNumKinds); i++) {
    section_allocators_[i] = nullptr;
  }
  low_level_device_ = nullptr;
}

uint32_t MicroSession::PushToExecQueue(DevPtr func_ptr, const TVMArgs& args) {
  std::cout << "[MicroSession::PushToExecQueue]" << std::endl;
  if (thumb_mode_) {
    func_ptr += 1;
  }
  int32_t (*func_dev_addr)(void*, void*, int32_t) =
      reinterpret_cast<int32_t (*)(void*, void*, int32_t)>(func_ptr.value());

  // Create an allocator stream for the memory region after the most recent
  // allocation in the args section.
  DevPtr args_addr = GetAllocator(SectionKind::kArgs)->curr_end_addr();
  TargetDataLayoutEncoder encoder(args_addr, word_size_);

  std::cout << "  after encoder alloc" << std::endl;

  std::tuple<DevPtr, DevPtr> arg_field_addrs = EncoderAppend(&encoder, args);

  std::cout << "  after encoder append" << std::endl;
  // Flush `stream` to device memory.
  DevPtr stream_dev_addr =
      GetAllocator(SectionKind::kArgs)->Allocate(encoder.buf_size());
  std::cout << "  low-level device: " << low_level_device() << std::endl;
  low_level_device()->Write(stream_dev_addr,
                            reinterpret_cast<void*>(encoder.data()),
                            encoder.buf_size());
  std::cout << "  after encoder write" << std::endl;

  if (word_size_ == 4) {
    TVMValue* arg_values_dev_addr = std::get<0>(arg_field_addrs).cast_to<TVMValue*>();
    int* arg_type_codes_dev_addr = std::get<1>(arg_field_addrs).cast_to<int*>();
    UTVMTask32 task = {
      .func = *((uint32_t*) &func_dev_addr),
      .arg_values = *((uint32_t*) &arg_values_dev_addr),
      .arg_type_codes = *((uint32_t*) &arg_type_codes_dev_addr),
      .num_args = args.num_args,
    };
    // Write the task.
    DevSymbolWrite(runtime_symbol_map_, "utvm_task", task);
  } else if (word_size_ == 8) {
    TVMValue* arg_values_dev_addr = std::get<0>(arg_field_addrs).cast_to<TVMValue*>();
    int* arg_type_codes_dev_addr = std::get<1>(arg_field_addrs).cast_to<int*>();
    UTVMTask64 task = {
      .func = *((uint64_t*) &func_dev_addr),
      .arg_values = *((uint64_t*) &arg_values_dev_addr),
      .arg_type_codes = *((uint64_t*) &arg_type_codes_dev_addr),
      .num_args = args.num_args,
    };
    // Write the task.
    DevSymbolWrite(runtime_symbol_map_, "utvm_task", task);
  } else {
    // TODO hoist word size check to initialization
    CHECK(false) << "unsupported word size " << word_size_;
  }

  std::cout << "  after task write" << std::endl;

  DevPtr utvm_init_addr = runtime_symbol_map_["UTVMInit"];
  DevPtr utvm_done_addr = runtime_symbol_map_["UTVMDone"];
  if (thumb_mode_) {
    utvm_init_addr += 1;
  }

  std::cout << "  UTVMInit loc: " << utvm_init_addr.cast_to<void*>() << std::endl;
  std::cout << "  UTVMDone loc: " << utvm_done_addr.cast_to<void*>() << std::endl;
  //std::cout << "  do execution things: ";
  //char tmp;
  //std::cin >> tmp;

  low_level_device()->Execute(utvm_init_addr, utvm_done_addr);

  //// Check if there was an error during execution.  If so, log it.
  //CheckDeviceError();

  uint64_t workspace_start = DevSymbolRead<uint64_t>(runtime_symbol_map_, "utvm_workspace_start");
  std::cout << "  workspace start: " << workspace_start << std::endl;
  uint64_t workspace_end = DevSymbolRead<uint64_t>(runtime_symbol_map_, "utvm_workspace_end");
  std::cout << "  workspace end: " << workspace_end << std::endl;
  uint64_t word_size = DevSymbolRead<uint64_t>(runtime_symbol_map_, "utvm_word_size");
  std::cout << "  word size: " << word_size << std::endl;

  //std::uintptr_t workspace_curr = DevSymbolRead<std::uintptr_t>(runtime_symbol_map_, "utvm_workspace_curr");
  //std::cout << "  workspace curr: " << workspace_curr << std::endl;
  //size_t num_active_allocs = DevSymbolRead<size_t>(runtime_symbol_map_, "utvm_num_active_allocs");
  //std::cout << "  num active allocs: " << num_active_allocs << std::endl;
  //std::uintptr_t last_error = DevSymbolRead<std::uintptr_t>(runtime_symbol_map_, "utvm_last_error");
  //std::cout << "  last error: " << last_error << std::endl;
  //int32_t return_code = DevSymbolRead<int32_t>(runtime_symbol_map_, "utvm_return_code");
  //std::cout << "  return code: " << return_code << std::endl;
  uint32_t task_time = DevSymbolRead<uint32_t>(runtime_symbol_map_, "utvm_task_time");
  std::cout << "  task time was " << task_time << std::endl;
  std::cout << "  --------------------------------------------------------------------------------" << std::endl;

  GetAllocator(SectionKind::kArgs)->Free(stream_dev_addr);
  return task_time;
}

BinaryInfo MicroSession::LoadBinary(const std::string& binary_path, bool patch_dylib_pointers) {
  DevMemRegion text_section;
  DevMemRegion rodata_section;
  DevMemRegion data_section;
  DevMemRegion bss_section;

  text_section.size = GetSectionSize(binary_path, SectionKind::kText, toolchain_prefix_, word_size_);
  rodata_section.size = GetSectionSize(binary_path, SectionKind::kRodata, toolchain_prefix_, word_size_);
  data_section.size = GetSectionSize(binary_path, SectionKind::kData, toolchain_prefix_, word_size_);
  bss_section.size = GetSectionSize(binary_path, SectionKind::kBss, toolchain_prefix_, word_size_);
  std::cout << "text size: " << std::hex << text_section.size << std::endl;
  std::cout << "rodata size: " << std::hex << rodata_section.size << std::endl;
  std::cout << "data size: " << std::hex << data_section.size << std::endl;
  std::cout << "bss size: " << std::hex << bss_section.size << std::endl;

  text_section.start = AllocateInSection(SectionKind::kText, text_section.size);
  rodata_section.start = AllocateInSection(SectionKind::kRodata, rodata_section.size);
  data_section.start = AllocateInSection(SectionKind::kData, data_section.size);
  bss_section.start = AllocateInSection(SectionKind::kBss, bss_section.size);
  CHECK(text_section.start != nullptr && rodata_section.start != nullptr &&
        data_section.start != nullptr && bss_section.start != nullptr)
      << "not enough space to load module on device";

  std::cout << "text start: " << std::hex << text_section.start.value() << std::endl;
  std::cout << "rodata start: " << std::hex << rodata_section.start.value() << std::endl;
  std::cout << "data start: " << std::hex << data_section.start.value() << std::endl;
  std::cout << "bss start: " << std::hex << bss_section.start.value() << std::endl;
  std::string relocated_bin = RelocateBinarySections(
      binary_path,
      word_size_,
      // TODO fill in new args
      text_section.start,
      rodata_section.start,
      data_section.start,
      bss_section.start,
      GetAllocator(SectionKind::kWorkspace)->start_addr(),
      GetAllocator(SectionKind::kWorkspace)->max_addr(),
      GetAllocator(SectionKind::kStack)->max_addr(),
      toolchain_prefix_);
  std::string text_contents = ReadSection(relocated_bin, SectionKind::kText, toolchain_prefix_);
  std::string rodata_contents = ReadSection(relocated_bin, SectionKind::kRodata, toolchain_prefix_);
  std::string data_contents = ReadSection(relocated_bin, SectionKind::kData, toolchain_prefix_);
  std::string bss_contents = ReadSection(relocated_bin, SectionKind::kBss, toolchain_prefix_);

  std::cout << "writing text (size = " << std::dec << text_contents.size() << ")" << std::endl;
  low_level_device_->Write(text_section.start, &text_contents[0], text_section.size);
  std::cout << "writing rodata (size = " << std::dec << rodata_contents.size() << ")" << std::endl;
  low_level_device_->Write(rodata_section.start, &rodata_contents[0], rodata_section.size);
  std::cout << "writing data (size = " << std::dec << data_contents.size() << ")" << std::endl;
  low_level_device_->Write(data_section.start, &data_contents[0], data_section.size);
  std::cout << "writing bss (size = " << std::dec << bss_contents.size() << ")" << std::endl;
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
  std::cout << "[MicroSession::EncoderAppend]" << std::endl;
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
  // TODO make this code mux on the word size
  /*
  typedef struct StructARMTVMArray {
    uint32_t data;
    DLContext ctx;
    int ndim;
    DLDataType dtype;
    uint32_t shape;
    uint32_t strides;
    uint32_t pad1;
    uint32_t byte_offset;
    uint32_t pad2;
  } ARMTVMArray;
  */

  if (word_size_ == 4) {
    auto tvm_arr_slot = encoder->Alloc<TVMArray32>();
    //CHECK(false) << "should we be allocing int32_t?";
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

    int64_t* dev_shape = shape_addr.cast_to<int64_t*>();
    int64_t* dev_strides = strides_addr.cast_to<int64_t*>();
    TVMArray32 dev_arr = {
      .data = *((uint32_t*) &arr.data),
      .ctx = arr.ctx,
      .ndim = arr.ndim,
      .dtype = arr.dtype,
      .shape = *((uint32_t*) &dev_shape),
      .strides = *((uint32_t*) &dev_strides),
      .pad1 = 0,
      .byte_offset = *((uint32_t*) &arr.byte_offset),
      .pad2 = 0,
    };
    // Update the device type to look like a host, because codegen generates
    // checks that it is a host array.
    CHECK(dev_arr.ctx.device_type == static_cast<DLDeviceType>(kDLMicroDev)) << "attempt to write TVMArray with non-micro device type";
    dev_arr.ctx.device_type = DLDeviceType::kDLCPU;
    tvm_arr_slot.WriteValue(dev_arr);
    return tvm_arr_slot.start_addr();
  } else if (word_size_ == 8) {
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

    int64_t* dev_shape = shape_addr.cast_to<int64_t*>();
    int64_t* dev_strides = strides_addr.cast_to<int64_t*>();

    // Copy `arr`, update the copy's pointers to be device pointers, then
    // write the copy to `tvm_arr_slot`.
    TVMArray dev_arr = arr;
    // Update the device type to look like a host, because codegen generates
    // checks that it is a host array.
    CHECK(dev_arr.ctx.device_type == static_cast<DLDeviceType>(kDLMicroDev)) << "attempt to write TVMArray with non-micro device type";
    dev_arr.ctx.device_type = DLDeviceType::kDLCPU;
    //// Add the base address of the device to the array's data's device offset to
    //// get a device address.
    //DevPtr arr_offset(reinterpret_cast<std::uintptr_t>(arr.data));
    //dev_arr.data = low_level_device()->ToDevPtr(arr_offset).cast_to<void*>();
    dev_arr.shape = shape_addr.cast_to<int64_t*>();
    dev_arr.strides = strides_addr.cast_to<int64_t*>();
    tvm_arr_slot.WriteValue(dev_arr);
    return tvm_arr_slot.start_addr();
  } else {
    CHECK(false) << "invalid word size";
  }
}

void MicroSession::CheckDeviceError() {
  std::cout << "[MicroSession::CheckDeviceError]" << std::endl;
  int32_t return_code = DevSymbolRead<int32_t>(runtime_symbol_map_, "utvm_return_code");
  std::cout << "  return_code: " << return_code << std::endl;
  std::cout << "  return_code loc: " << runtime_symbol_map_["utvm_return_code"].cast_to<void*>() << std::endl;

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
  std::cout << "patching " << func_name << " with addr " << runtime_impl_addr.cast_to<void*>() << std::endl;
  std::ostringstream func_name_underscore;
  func_name_underscore << func_name << "_";
  if (word_size_ == 4) {
    DevSymbolWrite(symbol_map, func_name_underscore.str(), (uint32_t) runtime_impl_addr.value());
  } else if (word_size_ == 8) {
    DevSymbolWrite(symbol_map, func_name_underscore.str(), (uint64_t) runtime_impl_addr.value());
  } else {
    CHECK(false) << "ayy";
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
  std::cout << "sym offset for " << symbol << " is " << sym_addr.cast_to<void*>() << std::endl;
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
    //session->CreateSession(
    //    "openocd", binary_path, "arm-none-eabi-", 0, "127.0.0.1", 6666);
    *rv = Module(session);
    });

}  // namespace runtime
}  // namespace tvm
