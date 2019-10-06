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
   std::cout << "sizeof(TVMArray): " << sizeof(TVMArray) << std::endl;
   std::cout << "sizeof(TVMValue): " << sizeof(TVMValue) << std::endl;

  //DevBaseOffset curr_start_offset = kDeviceStart;
  //for (size_t i = 0; i < static_cast<size_t>(SectionKind::kNumKinds); i++) {
  //  size_t section_size = GetDefaultSectionSize(static_cast<SectionKind>(i));
  //  section_allocators_[i] = std::make_shared<MicroSectionAllocator>(DevMemRegion {
  //    .start = curr_start_offset,
  //    .size = section_size,
  //  });
  //  curr_start_offset += section_size;
  //}
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

  //CHECK(!binary_path.empty()) << "uTVM runtime not initialized";
  //runtime_bin_info_ = LoadBinary(binary_path, /* patch_dylib_pointers */ false);
  //utvm_main_symbol_ = low_level_device()->ToDevOffset(symbol_map_["UTVMMain"]);
  //utvm_done_symbol_ = low_level_device()->ToDevOffset(symbol_map_["UTVMDone"]);

  //if (device_type == "openocd") {
  //  // Set OpenOCD device's stack pointer.
  //  auto stack_section = GetAllocator(SectionKind::kStack);
  //  low_level_device_->SetStackTop(stack_section->max_end_offset());
  //}

  // Patch workspace pointers to the start of the workspace section.
  //DevBaseOffset workspace_start_offset = GetAllocator(SectionKind::kWorkspace)->start_offset();
  //DevBaseOffset workspace_end_offset = GetAllocator(SectionKind::kWorkspace)->max_end_offset();
  //void* workspace_start_addr =
  //    low_level_device_->ToDevPtr(workspace_start_offset).cast_to<void*>();
  //void* workspace_end_addr =
  //    low_level_device_->ToDevPtr(workspace_end_offset).cast_to<void*>();
  // TODO(weberlo): A lot of these symbol writes can be converted into symbols
  // in the C source, where the symbols are created by the linker script we
  // generate in python.
  //DevSymbolWrite(symbol_map_, "utvm_workspace_begin", workspace_start_addr);
  //DevSymbolWrite(symbol_map_, "utvm_workspace_end", workspace_end_addr);
}

void MicroSession::BakeSession(const std::string& binary) {
  symbol_map_ = SymbolMap(binary, toolchain_prefix_);
  std::cout << symbol_map_["UTVMMain"].value() << std::endl;
  std::cout << symbol_map_["utvm_task"].value() << std::endl;
  low_level_device()->Connect();
}

// ARM and other manufacturers use the LSB of a function address to determine
// whether it's a "thumb mode" function (TODO: figure out what that means).
const bool kRequiresThumbModeBit = true;
void MicroSession::PushToExecQueue(DevPtr func_ptr, const TVMArgs& args) {
  int32_t (*func_dev_addr)(void*, void*, int32_t) =
      reinterpret_cast<int32_t (*)(void*, void*, int32_t)>(func_ptr.value());
  // TODO: make this a parameter.
  if (kRequiresThumbModeBit) {
    func_dev_addr += 1;
  }

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
  std::cout << "utvm_task loc: " << GetSymbolLoc("utvm_task").value() << std::endl;
  DevSymbolWrite(symbol_map_, "utvm_task", task);

  //low_level_device()->Execute(utvm_main_symbol_, utvm_done_symbol_);
  std::cout << "do execution things: ";
  char tmp;
  std::cin >> tmp;
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
  std::cout << "writing byte_offset " << arr.byte_offset << std::endl;
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

  std::cout << "sizeof(ARMTVMArray): " << sizeof(ARMTVMArray) << std::endl;
  std::cout << "data offs: " << (((uintptr_t) &(dev_arr.data)) - ((uintptr_t) &dev_arr)) << std::endl;
  std::cout << "ctx offs: " << (((uintptr_t) &(dev_arr.ctx)) - ((uintptr_t) &dev_arr)) << std::endl;
  std::cout << "ndim offs: " << (((uintptr_t) &(dev_arr.ndim)) - ((uintptr_t) &dev_arr)) << std::endl;
  std::cout << "dtype offs: " << (((uintptr_t) &(dev_arr.dtype)) - ((uintptr_t) &dev_arr)) << std::endl;
  std::cout << "strides offs: " << (((uintptr_t) &(dev_arr.strides)) - ((uintptr_t) &dev_arr)) << std::endl;
  std::cout << "byte_offset offs: " << (((uintptr_t) &(dev_arr.byte_offset)) - ((uintptr_t) &dev_arr)) << std::endl;


  // Copy `arr`, update the copy's pointers to be device pointers, then
  // write the copy to `tvm_arr_slot`.
  //TVMArray dev_arr = arr;
  // Update the device type to look like a host, because codegen generates
  // checks that it is a host array.
  CHECK(dev_arr.ctx.device_type == static_cast<DLDeviceType>(kDLMicroDev))
    << "attempt to write TVMArray with non-micro device type";
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
  int32_t return_code = DevSymbolRead<int32_t>(symbol_map_, "utvm_return_code");

  if (return_code) {
    std::uintptr_t last_error =
        DevSymbolRead<std::uintptr_t>(symbol_map_, "utvm_last_error");
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
  void* runtime_impl_addr = symbol_map_[func_name].cast_to<void*>();
  std::ostringstream func_name_underscore;
  func_name_underscore << func_name << "_";
  DevSymbolWrite(symbol_map, func_name_underscore.str(), runtime_impl_addr);
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

class MicroWrappedFunc {
 public:
  MicroWrappedFunc(std::shared_ptr<MicroSession> session,
                   DevPtr func_ptr) {
    session_ = session;
    func_ptr_ = func_ptr;
  }

  void operator()(TVMArgs args, TVMRetValue* rv) const {
    session_->PushToExecQueue(func_ptr_, args);
  }

 private:
  /*! \brief reference to the session for this function (to keep the session alive) */
  std::shared_ptr<MicroSession> session_;
  /*! \brief offset of the function to be called */
  DevPtr func_ptr_;
};

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

TVM_REGISTER_GLOBAL("micro._BakeSession")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    const std::string& binary = args[0];

    std::shared_ptr<MicroSession>& session = MicroSession::Current();
    session->BakeSession(binary);
    });

TVM_REGISTER_GLOBAL("micro._GetFunction")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    const std::string& name = args[0];
    std::shared_ptr<MicroSession>& session = MicroSession::Current();

    DevPtr func_ptr = session->GetSymbolLoc(name);
    *rv = PackedFunc(MicroWrappedFunc(session, func_ptr));
    });

}  // namespace runtime
}  // namespace tvm
