/*!
 *  Copyright (c) 2019 by Contributors
 * \file micro_session.cc
 * \brief session to manage multiple micro modules
 */

#include <memory>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include "./micro_session.h"
#include "./low_level_device.h"
#include "./allocator_stream.h"

namespace tvm {
namespace runtime {

MicroSession::MicroSession() {
  text_allocator_ = std::unique_ptr<MicroSectionAllocator>(
      new MicroSectionAllocator(reinterpret_cast<void*>(kTextStart),
                                reinterpret_cast<void*>(kDataStart)));
  data_allocator_ = std::unique_ptr<MicroSectionAllocator>(
      new MicroSectionAllocator(reinterpret_cast<void*>(kDataStart),
                                reinterpret_cast<void*>(kBssStart)));
  bss_allocator_ = std::unique_ptr<MicroSectionAllocator>(
      new MicroSectionAllocator(reinterpret_cast<void*>(kBssStart),
                                reinterpret_cast<void*>(kArgsStart)));
  args_allocator_ = std::unique_ptr<MicroSectionAllocator>(
      new MicroSectionAllocator(reinterpret_cast<void*>(kArgsStart),
                                reinterpret_cast<void*>(kStackStart)));
  stack_allocator_ = std::unique_ptr<MicroSectionAllocator>(
      new MicroSectionAllocator(reinterpret_cast<void*>(kStackStart),
                                reinterpret_cast<void*>(kHeapStart)));
  heap_allocator_ = std::unique_ptr<MicroSectionAllocator>(
      new MicroSectionAllocator(reinterpret_cast<void*>(kHeapStart),
                                reinterpret_cast<void*>(kWorkspaceStart)));
  workspace_allocator_ = std::unique_ptr<MicroSectionAllocator>(
      new MicroSectionAllocator(reinterpret_cast<void*>(kWorkspaceStart),
                                reinterpret_cast<void*>(kMemorySize)));
}

MicroSession::~MicroSession() {
}

void MicroSession::InitSession(TVMArgs args) {
  std::string device_type = args[0];
  if (device_type == "host") {
    low_level_device_ = HostLowLevelDeviceCreate(kMemorySize);
    SetInitBinaryPath(args[1]);
  } else if (device_type == "openocd") {
    low_level_device_ = OpenOCDLowLevelDeviceCreate(args[2]);
    SetInitBinaryPath(args[1]);
  } else {
    LOG(FATAL) << "Unsupported micro low-level device";
  }
  LoadInitStub();
}

void* MicroSession::AllocateInSection(SectionKind type, size_t size) {
  void* alloc_ptr = nullptr;
  switch (type) {
    case kText:
      alloc_ptr = text_allocator_->Allocate(size);
      break;
    case kData:
      alloc_ptr = data_allocator_->Allocate(size);
      break;
    case kBss:
      alloc_ptr = bss_allocator_->Allocate(size);
      break;
    case kArgs:
      alloc_ptr = args_allocator_->Allocate(size);
      break;
    case kStack:
      alloc_ptr = stack_allocator_->Allocate(size);
      break;
    case kHeap:
      alloc_ptr = heap_allocator_->Allocate(size);
      break;
    case kWorkspace:
      alloc_ptr = workspace_allocator_->Allocate(size);
      break;
    default:
      LOG(FATAL) << "Unsupported section type during allocation";
  }
  return alloc_ptr;
}

void MicroSession::FreeInSection(SectionKind type, void* ptr) {
  switch (type) {
    case kText:
      text_allocator_->Free(ptr);
      break;
    case kData:
      data_allocator_->Free(ptr);
      break;
    case kBss:
      bss_allocator_->Free(ptr);
      break;
    case kArgs:
      args_allocator_->Free(ptr);
      break;
    case kStack:
      stack_allocator_->Free(ptr);
      break;
    case kHeap:
      heap_allocator_->Free(ptr);
      break;
    case kWorkspace:
      workspace_allocator_->Free(ptr);
      break;
    default:
      LOG(FATAL) << "Unsupported section type during free";
  }
}

void MicroSession::PushToExecQueue(void* func, TVMArgs args) {
  int (*func_dev_addr)(void*, void*, int32_t) =
      reinterpret_cast<int (*)(void*, void*, int32_t)>(
      GetAddr(func, low_level_device()->base_addr()));

  // Create an allocator stream for the memory region after the most recent
  // allocation in the args section.
  void* args_dev_addr = GetAddr(args_allocator_->section_max(),
                                low_level_device()->base_addr());
  AllocatorStream stream(args_dev_addr);
  UTVMArgs u_args = {
      .values = const_cast<TVMValue*>(args.values),
      .type_codes = const_cast<int*>(args.type_codes),
      .num_args = args.num_args,
  };
  StreamWrite(&u_args, &stream);
  // Flush `stream` to device memory.
  void* stream_dev_addr = args_allocator_->Allocate(stream.size());
  low_level_device()->Write(stream_dev_addr, (void*) stream.data(),
                            stream.size());

  UTVMTask task = {
      .func = func_dev_addr,
      .args = reinterpret_cast<UTVMArgs*>(args_dev_addr),
  };
  // TODO(mutinifni): handle bits / endianness
  void* task_dev_addr = GetSymbol(init_symbol_map_, "task",
                                  low_level_device()->base_addr());
  low_level_device()->Write(task_dev_addr, &task, sizeof(task));
  low_level_device()->Execute(utvm_main_symbol_addr_, utvm_done_symbol_addr_);
}

void MicroSession::LoadInitStub() {
  CHECK(!init_binary_path_.empty()) << "init library not initialized";
  // relocate and load binary on low-level device
  init_text_size_ = GetSectionSize(init_binary_path_, kText);
  init_data_size_ = GetSectionSize(init_binary_path_, kData);
  init_bss_size_ = GetSectionSize(init_binary_path_, kBss);
  init_text_start_ = AllocateInSection(kText, init_text_size_);
  init_data_start_ = AllocateInSection(kData, init_data_size_);
  init_bss_start_ = AllocateInSection(kBss, init_bss_size_);
  CHECK(init_text_start_ != nullptr &&
        init_data_start_ != nullptr &&
        init_bss_start_ != nullptr)
      << "Not enough space to load init binary on device";
  std::string relocated_bin = RelocateBinarySections(
      init_binary_path_,
      GetAddr(init_text_start_, low_level_device()->base_addr()),
      GetAddr(init_data_start_, low_level_device()->base_addr()),
      GetAddr(init_bss_start_, low_level_device()->base_addr()));
  std::string text_contents = ReadSection(relocated_bin, kText);
  std::string data_contents = ReadSection(relocated_bin, kData);
  std::string bss_contents = ReadSection(relocated_bin, kBss);
  low_level_device()->Write(init_text_start_, &text_contents[0], init_text_size_);
  low_level_device()->Write(init_data_start_, &data_contents[0], init_data_size_);
  low_level_device()->Write(init_bss_start_, &bss_contents[0], init_bss_size_);
  // obtain init stub binary metadata
  init_symbol_map_ = GetSymbolMap(relocated_bin);
  utvm_main_symbol_addr_ = GetSymbol(init_symbol_map_, "UTVMMain", nullptr);
  utvm_done_symbol_addr_ = GetSymbol(init_symbol_map_, "UTVMDone", nullptr);
}

void MicroSession::SetInitBinaryPath(std::string path) {
  init_binary_path_ = path;
}

// TODO(mutinifni): overload StreamWrite with more val types as needed

void* MicroSession::StreamWrite(TVMArray* arr, AllocatorStream* stream) {
  Slot tvm_arr_slot = stream->Alloc<TVMArray>();
  Slot shape_slot = stream->AllocArray<int64_t>(arr->ndim);

  // `shape` and `strides` are stored on the host, so we need to write them to
  // the device first. The `data` field is already allocated on the device and
  // is a device pointer, so we don't need to write it.
  shape_slot.WriteEntire(arr->shape);
  void* shape_addr = shape_slot.dev_start_addr();
  void* strides_addr = nullptr;
  if (arr->strides != nullptr) {
    Slot stride_slot = stream->AllocArray<int64_t>(arr->ndim);
    stride_slot.WriteEntire(arr->strides);
    strides_addr = stride_slot.dev_start_addr();
  }

  // Copy `arr`, update the copy's pointers to be on-device pointers, then write
  // the copy to `tvm_arr_slot`.
  TVMArray dev_arr = *arr;
  dev_arr.data = reinterpret_cast<uint8_t*>(const_cast<void*>(low_level_device()->base_addr())) +
                 reinterpret_cast<std::uintptr_t>(arr->data);
  dev_arr.shape = static_cast<int64_t*>(shape_addr);
  dev_arr.strides = static_cast<int64_t*>(strides_addr);
  tvm_arr_slot.Write(&dev_arr);
  return tvm_arr_slot.dev_start_addr();
}

void* MicroSession::StreamWrite(UTVMArgs* args, AllocatorStream* stream) {
  Slot utvm_args_slot = stream->Alloc<UTVMArgs>();

  const int* type_codes = args->type_codes;
  int num_args = args->num_args;

  Slot tvm_vals_slot = stream->AllocArray<TVMValue*>(num_args);
  Slot type_codes_slot = stream->AllocArray<const int>(num_args);

  for (int i = 0; i < num_args; i++) {
    switch (type_codes[i]) {
      case kNDArrayContainer: {
        void* val_addr = StreamWrite((TVMArray*) args->values[i].v_handle, stream);
        tvm_vals_slot.Write(&val_addr);
        break;
      }
      // TODO(mutinifni): implement other cases if needed
      default:
        LOG(FATAL) << "Unsupported type code for writing args: " << type_codes[i];
        break;
    }
  }
  type_codes_slot.WriteEntire(type_codes);

  UTVMArgs dev_args = {
    .values = reinterpret_cast<TVMValue*>(tvm_vals_slot.dev_start_addr()),
    .type_codes = reinterpret_cast<int*>(type_codes_slot.dev_start_addr()),
    .num_args = num_args,
  };
  utvm_args_slot.Write(&dev_args);
  return utvm_args_slot.dev_start_addr();
}

// initializes micro session and low-level device from Python frontend
TVM_REGISTER_GLOBAL("micro._MicroInit")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::shared_ptr<MicroSession> session = MicroSession::Global();
    session->InitSession(args);
    });
}  // namespace runtime
}  // namespace tvm
