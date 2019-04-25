/*!
 *  Copyright (c) 2019 by Contributors
 * \file micro_session.cc
 * \brief session to manage multiple micro modules
 */

#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include <memory>
#include "micro_session.h"
#include "low_level_device.h"
#include "target_data_layout_encoder.h"

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
  TargetDataLayoutEncoder encoder(args_dev_addr, low_level_device()->base_addr());
  UTVMArgs u_args = {
      .values = const_cast<TVMValue*>(args.values),
      .type_codes = const_cast<int*>(args.type_codes),
      .num_args = args.num_args,
  };
  encoder.Write(&u_args);
  // Flush `stream` to device memory.
  void* stream_dev_addr = args_allocator_->Allocate(encoder.buf_size());
  low_level_device()->Write(stream_dev_addr,
                            reinterpret_cast<void*>(const_cast<uint8_t*>(encoder.data())),
                            encoder.buf_size());

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

// initializes micro session and low-level device from Python frontend
TVM_REGISTER_GLOBAL("micro._MicroInit")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::shared_ptr<MicroSession> session = MicroSession::Global();
    session->InitSession(args);
    });
}  // namespace runtime
}  // namespace tvm
