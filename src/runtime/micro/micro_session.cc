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

MicroSession::MicroSession() {
  DevBaseOffset curr_start_offset = DevBaseOffset(0x200001C8);
  for (size_t i = 0; i < static_cast<size_t>(SectionKind::kNumKinds); i++) {
    size_t section_size = GetDefaultSectionSize(static_cast<SectionKind>(i));
    section_allocators_[i] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = curr_start_offset,
      .size = section_size,
    });
    curr_start_offset += section_size;
  }


  CHECK(curr_start_offset.value() < 0x20050000) << "exceeded available RAM on device (" << std::endl;

  std::cout << "[Memory Layout]" << std::endl;
  std::cout << "  text (size = " << (section_allocators_[0]->capacity() / 1000.0) << " KB): " << section_allocators_[0]->start_offset().cast_to<void*>() << std::endl;
  std::cout << "  rodata (size = " << (section_allocators_[1]->capacity() / 1000.0) << " KB): " << section_allocators_[1]->start_offset().cast_to<void*>() << std::endl;
  std::cout << "  data (size = " << (section_allocators_[2]->capacity() / 1000.0) << " KB): " << section_allocators_[2]->start_offset().cast_to<void*>() << std::endl;
  std::cout << "  bss (size = " << (section_allocators_[3]->capacity() / 1000.0) << " KB): " << section_allocators_[3]->start_offset().cast_to<void*>() << std::endl;
  std::cout << "  args (size = " << (section_allocators_[4]->capacity() / 1000.0) << " KB): " << section_allocators_[4]->start_offset().cast_to<void*>() << std::endl;
  std::cout << "  heap (size = " << (section_allocators_[5]->capacity() / 1000.0) << " KB): " << section_allocators_[5]->start_offset().cast_to<void*>() << std::endl;
  std::cout << "  workspace (size = " << (section_allocators_[6]->capacity() / 1000.0) << " KB): " << section_allocators_[6]->start_offset().cast_to<void*>() << std::endl;
  std::cout << "  stack (size = " << (section_allocators_[7]->capacity() / 1000.0) << " KB): " << section_allocators_[7]->start_offset().cast_to<void*>() << std::endl;

  // NOTE: we don't use this for openocd
  memory_size_ = 0;
  //memory_size_ = curr_start_offset.cast_to<size_t>();

  /* Linker script sample
   * #if !defined(MBED_APP_START)
   *   #define MBED_APP_START 0x08000000
   * #endif
   *
   * #if !defined(MBED_APP_SIZE)
   *   #define MBED_APP_SIZE 1024K
   * #endif
   *
   * MEMORY
   * {
   *   FLASH (rx) : ORIGIN = MBED_APP_START, LENGTH = MBED_APP_SIZE
   *   RAM (rwx)  : ORIGIN = 0x200001C8, LENGTH = 320K - 0x1C8
   * }
   */

   /*
  size_t half_flash_size = 512000;  // 0.5 MB
  DevBaseOffset curr_start_offset = DevBaseOffset(0x08000000);
  section_allocators_[static_cast<size_t>(SectionKind::kText)] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
    .start = curr_start_offset,
    .size = half_flash_size,
  });
  curr_start_offset += half_flash_size;
  section_allocators_[static_cast<size_t>(SectionKind::kRodata)] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
    .start = curr_start_offset,
    .size = half_flash_size,
  });
  curr_start_offset += half_flash_size;

  curr_start_offset = DevBaseOffset(0x200001C8);
  size_t one_sixth_ram_size = 53256;
  section_allocators_[static_cast<size_t>(SectionKind::kData)] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
    .start = curr_start_offset,
    .size = one_sixth_ram_size,
  });
  curr_start_offset += one_sixth_ram_size;
  section_allocators_[static_cast<size_t>(SectionKind::kBss)] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
    .start = curr_start_offset,
    .size = one_sixth_ram_size,
  });
  curr_start_offset += one_sixth_ram_size;
  section_allocators_[static_cast<size_t>(SectionKind::kArgs)] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
    .start = curr_start_offset,
    .size = one_sixth_ram_size,
  });
  curr_start_offset += one_sixth_ram_size;
  section_allocators_[static_cast<size_t>(SectionKind::kStack)] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
    .start = curr_start_offset,
    .size = one_sixth_ram_size,
  });
  curr_start_offset += one_sixth_ram_size;
  section_allocators_[static_cast<size_t>(SectionKind::kHeap)] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
    .start = curr_start_offset,
    .size = one_sixth_ram_size,
  });
  curr_start_offset += one_sixth_ram_size;
  section_allocators_[static_cast<size_t>(SectionKind::kWorkspace)] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
    .start = curr_start_offset,
    .size = one_sixth_ram_size,
  });
  curr_start_offset += one_sixth_ram_size;

  memory_size_ = 0;
  */

  /*
  // TODO: why do we need to start 0x1c8 bytes after the start of ram?
  //curr_start_offset = DevBaseOffset(0x200001C8);
  size_t one_eighth_ram_size = 40000;
  DevBaseOffset curr_start_offset = DevBaseOffset(0x200001C8);

  std::cout << "[Memory Layout]" << std::endl;
  std::cout << "text start: " << curr_start_offset.cast_to<void*>() << std::endl;
  size_t text_size
  section_allocators_[static_cast<size_t>(SectionKind::kText)] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = curr_start_offset,
      .size = one_eighth_ram_size,
      });
  curr_start_offset += one_eighth_ram_size;
  std::cout << "rodata start: " << curr_start_offset.cast_to<void*>() << std::endl;
  section_allocators_[static_cast<size_t>(SectionKind::kRodata)] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = curr_start_offset,
      .size = one_eighth_ram_size,
      });
  curr_start_offset += one_eighth_ram_size;
  std::cout << "data start: " << curr_start_offset.cast_to<void*>() << std::endl;
  section_allocators_[static_cast<size_t>(SectionKind::kData)] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = curr_start_offset,
      .size = one_eighth_ram_size,
      });
  curr_start_offset += one_eighth_ram_size;
  std::cout << "bss start: " << curr_start_offset.cast_to<void*>() << std::endl;
  section_allocators_[static_cast<size_t>(SectionKind::kBss)] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = curr_start_offset,
      .size = one_eighth_ram_size,
      });
  curr_start_offset += one_eighth_ram_size;
  std::cout << "args start: " << curr_start_offset.cast_to<void*>() << std::endl;
  section_allocators_[static_cast<size_t>(SectionKind::kArgs)] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = curr_start_offset,
      .size = one_eighth_ram_size,
      });
  curr_start_offset += one_eighth_ram_size;
  std::cout << "stack start: " << curr_start_offset.cast_to<void*>() << std::endl;
  section_allocators_[static_cast<size_t>(SectionKind::kStack)] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = curr_start_offset,
      .size = one_eighth_ram_size,
      });
  curr_start_offset += one_eighth_ram_size;
  std::cout << "heap start: " << curr_start_offset.cast_to<void*>() << std::endl;
  section_allocators_[static_cast<size_t>(SectionKind::kHeap)] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = curr_start_offset,
      .size = one_eighth_ram_size,
      });
  curr_start_offset += one_eighth_ram_size;
  std::cout << "workspace start: " << curr_start_offset.cast_to<void*>() << std::endl;
  section_allocators_[static_cast<size_t>(SectionKind::kWorkspace)] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
      .start = curr_start_offset,
      .size = one_eighth_ram_size,
      });
  curr_start_offset += one_eighth_ram_size;
  std::cout << "WHAT THE FUCK" << std::endl;
  */
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

  runtime_symbol_map_ = LoadBinary(binary_path, false).symbol_map;
  std::cout << runtime_symbol_map_["UTVMMain"].cast_to<void*>() << std::endl;
  std::cout << runtime_symbol_map_["utvm_task"].cast_to<void*>() << std::endl;

  //if (device_type == "openocd") {
  //  // Set OpenOCD device's stack pointer.
  //  auto stack_section = GetAllocator(SectionKind::kStack);
  //  low_level_device_->SetStackTop(stack_section->max_end_offset());
  //}

  // Patch workspace pointers to the start of the workspace section.
  DevBaseOffset workspace_start_offset = GetAllocator(SectionKind::kWorkspace)->start_offset();
  DevBaseOffset workspace_end_offset = GetAllocator(SectionKind::kWorkspace)->max_end_offset();
  void* workspace_start_addr =
      low_level_device_->ToDevPtr(workspace_start_offset).cast_to<void*>();
  void* workspace_end_addr =
      low_level_device_->ToDevPtr(workspace_end_offset).cast_to<void*>();
  // TODO(weberlo): A lot of these symbol writes can be converted into symbols
  // in the C source, where the symbols are created by the linker script we
  // generate in python.
  DevSymbolWrite(runtime_symbol_map_, "utvm_workspace_begin", workspace_start_addr);
  DevSymbolWrite(runtime_symbol_map_, "utvm_workspace_end", workspace_end_addr);
}

//void MicroSession::BakeSession(const std::string& binary_path) {
//  //runtime_symbol_map_ = SymbolMap(binary, toolchain_prefix_);
//  runtime_symbol_map_ = LoadBinary(binary_path, false).symbol_map;
//  std::cout << runtime_symbol_map_["UTVMMain"].value() << std::endl;
//  std::cout << runtime_symbol_map_["utvm_task"].value() << std::endl;
//}

// ARM and other manufacturers use the LSB of a function address to determine
// whether it's a "thumb mode" function (TODO: figure out what that means).
const bool kRequiresThumbModeBit = true;
uint32_t MicroSession::PushToExecQueue(DevPtr func_ptr, const TVMArgs& args) {
  std::cout << "[MicroSession::PushToExecQueue]" << std::endl;
  // TODO: make this a parameter.
  if (kRequiresThumbModeBit) {
    func_ptr += 1;
  }
  int32_t (*func_dev_addr)(void*, void*, int32_t) =
      reinterpret_cast<int32_t (*)(void*, void*, int32_t)>(func_ptr.value());

  // Create an allocator stream for the memory region after the most recent
  // allocation in the args section.
  DevPtr args_addr =
      low_level_device()->ToDevPtr(GetAllocator(SectionKind::kArgs)->curr_end_offset());
  TargetDataLayoutEncoder encoder(args_addr);

  std::cout << "  after encoder alloc" << std::endl;

  std::tuple<DevPtr, DevPtr> arg_field_addrs = EncoderAppend(&encoder, args);

  std::cout << "  after encoder append" << std::endl;
  // Flush `stream` to device memory.
  DevBaseOffset stream_dev_offset =
      GetAllocator(SectionKind::kArgs)->Allocate(encoder.buf_size());
  std::cout << "  low-level device: " << low_level_device() << std::endl;
  low_level_device()->Write(stream_dev_offset,
                            reinterpret_cast<void*>(encoder.data()),
                            encoder.buf_size());
  std::cout << "  after encoder write" << std::endl;

  //UTVMTask task = {
  //  .func = func_dev_addr,
  //  .arg_values = std::get<0>(arg_field_addrs).cast_to<TVMValue*>(),
  //  .arg_type_codes = std::get<1>(arg_field_addrs).cast_to<int*>(),
  //  .num_args = args.num_args,
  //};
  typedef struct StructARMUTVMTask {
    /*! \brief Pointer to function to call for this task */
    uint32_t func;
    /*! \brief Array of argument values */
    uint32_t arg_values;
    /*! \brief Array of type codes for each argument value */
    uint32_t arg_type_codes;
    /*! \brief Number of arguments */
    uint32_t num_args;
  } ARMUTVMTask;
  TVMValue* arg_values_dev_addr = std::get<0>(arg_field_addrs).cast_to<TVMValue*>();
  int* arg_type_codes_dev_addr = std::get<1>(arg_field_addrs).cast_to<int*>();
  ARMUTVMTask task = {
    .func = *((uint32_t*) &func_dev_addr),
    .arg_values = *((uint32_t*) &arg_values_dev_addr),
    .arg_type_codes = *((uint32_t*) &arg_type_codes_dev_addr),
    .num_args = (uint32_t) args.num_args,
  };
  // Write the task.
  DevSymbolWrite(runtime_symbol_map_, "utvm_task", task);

  std::cout << "  after task write" << std::endl;

  //DevBaseOffset utvm_main_loc = DevBaseOffset(runtime_symbol_map_["UTVMMain"].value());
  DevBaseOffset utvm_init_loc = DevBaseOffset(runtime_symbol_map_["UTVMInit"].value());
  DevBaseOffset utvm_done_loc = DevBaseOffset(runtime_symbol_map_["UTVMDone"].value());
  if (kRequiresThumbModeBit) {
    utvm_init_loc += 1;
  }

  std::cout << "  UTVMInit loc: " << utvm_init_loc.cast_to<void*>() << std::endl;
  std::cout << "  UTVMDone loc: " << utvm_done_loc.cast_to<void*>() << std::endl;
  //std::cout << "do execution things: ";
  //char tmp;
  //std::cin >> tmp;
  low_level_device()->Execute(utvm_init_loc, utvm_done_loc);

  // Check if there was an error during execution.  If so, log it.
  CheckDeviceError();

  uint32_t task_time = DevSymbolRead<uint32_t>(runtime_symbol_map_, "utvm_task_time");

  GetAllocator(SectionKind::kArgs)->Free(stream_dev_offset);
  return task_time;
}

BinaryInfo MicroSession::LoadBinary(const std::string& binary_path, bool patch_dylib_pointers) {
  DevMemRegion text_section;
  DevMemRegion rodata_section;
  DevMemRegion data_section;
  DevMemRegion bss_section;

  text_section.size = GetSectionSize(binary_path, SectionKind::kText, toolchain_prefix_, kWordSize);
  rodata_section.size = GetSectionSize(binary_path, SectionKind::kRodata, toolchain_prefix_, kWordSize);
  data_section.size = GetSectionSize(binary_path, SectionKind::kData, toolchain_prefix_, kWordSize);
  bss_section.size = GetSectionSize(binary_path, SectionKind::kBss, toolchain_prefix_, kWordSize);
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
      low_level_device_->ToDevPtr(text_section.start),
      low_level_device_->ToDevPtr(rodata_section.start),
      low_level_device_->ToDevPtr(data_section.start),
      low_level_device_->ToDevPtr(bss_section.start),
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

  std::cout << "  num_args: " << num_args << std::endl;
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
        std::cout << "  before array encode" << std::endl;
        void* arr_ptr = EncoderAppend(encoder, *base_arr_handle).cast_to<void*>();
        std::cout << "  after array encode" << num_args << std::endl;
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
  std::cout << "  past args loop" << std::endl;
  type_codes_slot.WriteArray(type_codes, num_args);
  std::cout << "  tvm_vals_slot.start_addr(): " << tvm_vals_slot.start_addr().cast_to<void*>() << std::endl;
  std::cout << "  type_codes_slot.start_addr(): " << type_codes_slot.start_addr().cast_to<void*>() << std::endl;

  return std::make_tuple(tvm_vals_slot.start_addr(), type_codes_slot.start_addr());
}

DevPtr MicroSession::EncoderAppend(TargetDataLayoutEncoder* encoder, const TVMArray& arr) {
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

  //auto tvm_arr_slot = encoder->Alloc<TVMArray>();
  auto tvm_arr_slot = encoder->Alloc<ARMTVMArray>();
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
  ARMTVMArray dev_arr = {
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

  // Copy `arr`, update the copy's pointers to be device pointers, then
  // write the copy to `tvm_arr_slot`.
  //TVMArray dev_arr = arr;
  // Update the device type to look like a host, because codegen generates
  // checks that it is a host array.
  CHECK(dev_arr.ctx.device_type == static_cast<DLDeviceType>(kDLMicroDev)) << "attempt to write TVMArray with non-micro device type";
  dev_arr.ctx.device_type = DLDeviceType::kDLCPU;
  // Add the base address of the device to the array's data's device offset to
  // get a device address.
  //DevBaseOffset arr_offset(reinterpret_cast<std::uintptr_t>(arr.data));
  //dev_arr.data = low_level_device()->ToDevPtr(arr_offset).cast_to<void*>();
  //dev_arr.shape = shape_addr.cast_to<int64_t*>();
  //dev_arr.strides = strides_addr.cast_to<int64_t*>();
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
      DevBaseOffset last_err_offset = low_level_device()->ToDevOffset(DevPtr(last_error));
      last_error_str = ReadString(last_err_offset);
    }
    LOG(FATAL) << "error during micro function execution:\n"
               << "  return code: " << std::dec << return_code << "\n"
               << "  dev str addr: 0x" << std::hex << last_error << "\n"
               << "  dev str data: " << last_error_str << std::endl;
  }
}

void MicroSession::PatchImplHole(const SymbolMap& symbol_map, const std::string& func_name) {
  //void* runtime_impl_addr = runtime_symbol_map_[func_name].cast_to<void*>();
  DevPtr runtime_impl_addr = runtime_symbol_map_[func_name];
  if (kRequiresThumbModeBit) {
    runtime_impl_addr += 1;
  }
  std::cout << "patching " << func_name << " with addr " << runtime_impl_addr.cast_to<void*>() << std::endl;
  std::ostringstream func_name_underscore;
  func_name_underscore << func_name << "_";
  DevSymbolWrite(symbol_map, func_name_underscore.str(), (int32_t) runtime_impl_addr.value());
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
    const std::string& device_type = args[0];
    const std::string& binary_path = args[1];
    const std::string& toolchain_prefix = args[2];
    uint64_t base_addr = args[3];
    const std::string& server_addr = args[4];
    int port = args[5];
    ObjectPtr<MicroSession> session = make_object<MicroSession>();
    //session->CreateSession(
    //    device_type, binary_path, toolchain_prefix, base_addr, server_addr, port);
    session->CreateSession(
        "openocd", binary_path, "arm-none-eabi-", 0, "127.0.0.1", 6666);
    *rv = Module(session);
    });

}  // namespace runtime
}  // namespace tvm
