/*!
 *  Copyright (c) 2019 by Contributors
 * \file micro_session.cc
 * \brief session to manage multiple micro modules
 */

#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>
#include "micro_session.h"
#include "low_level_device.h"
#include "allocator_stream.h"
#include <cstdio>

namespace tvm {
namespace runtime {

MicroSession::MicroSession() {
  text_allocator_ = new MicroSectionAllocator((void*) kTextStart,
                                              (void*) kDataStart);
  data_allocator_ = new MicroSectionAllocator((void*) kDataStart,
                                              (void*) kBssStart);
  bss_allocator_ = new MicroSectionAllocator((void*) kBssStart,
                                             (void*) kArgsStart);
  args_allocator_ = new MicroSectionAllocator((void*) kArgsStart,
                                              (void*) kStackStart);
  stack_allocator_ = new MicroSectionAllocator((void*) kStackStart,
                                               (void*) kHeapStart);
  heap_allocator_ = new MicroSectionAllocator((void*) kHeapStart,
                                              (void*) kWorkspaceStart);
  workspace_allocator_ = new MicroSectionAllocator((void*) kWorkspaceStart,
                                                   (void*) kMemorySize);
}

MicroSession::~MicroSession() {
}

void MicroSession::InitSession(TVMArgs args) {
  std::string device_type = args[0];
  if (device_type == "host") {
    low_level_device_ = HostLowLevelDeviceCreate(kMemorySize);
    SetInitSource(args[1]);
  } else if (device_type == "openocd") {
    low_level_device_ = OpenOCDLowLevelDeviceCreate(args[2]);
    SetInitSource(args[1]);
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
  int num_args = args.num_args;
  int (*func_addr)(void*, void*, int32_t) =
      (int (*)(void*, void*, int32_t)) GetAddr(func, low_level_device()->base_addr());
  void* args_addr = AllocateTVMArgs(args);
  void* arg_type_ids_addr = reinterpret_cast<uint8_t*>(args_addr) +
                            sizeof(TVMValue*) * num_args;
  void* num_args_addr = reinterpret_cast<uint8_t*>(arg_type_ids_addr) +
                        sizeof(const int*) * num_args;
  void* task_addr = GetSymbol(init_symbol_map_, "task",
                              low_level_device()->base_addr());
  UTVMTask task = {.func = func_addr,
                   .args = args_addr,
                   .arg_type_ids = arg_type_ids_addr,
                   .num_args = reinterpret_cast<int32_t*>(num_args_addr)};
  // TODO(mutinifni): handle bits / endianness
  low_level_device()->Write(task_addr, &task, sizeof(task));
  low_level_device()->Execute(utvm_main_symbol_addr_, utvm_done_symbol_addr_);
}

void MicroSession::SetInitSource(std::string source) {
  init_source_ = source;
}

void MicroSession::LoadInitStub() {
  // compile init stub
  const auto* f = Registry::Get("tvm_callback_compile_micro");
  CHECK(f != nullptr) << "Require tvm_callback_compile_micro to exist in registry";
  std::string binary_path = (*f)(init_source_, low_level_device()->device_type());
  // relocate and load binary on low-level device
  init_text_size_ = GetSectionSize(binary_path, kText);
  init_data_size_ = GetSectionSize(binary_path, kData);
  init_bss_size_ = GetSectionSize(binary_path, kBss);
  init_text_start_ = AllocateInSection(kText, init_text_size_);
  init_data_start_ = AllocateInSection(kData, init_data_size_);
  init_bss_start_ = AllocateInSection(kBss, init_bss_size_);
  CHECK(init_text_start_ != nullptr &&
        init_data_start_ != nullptr &&
        init_bss_start_ != nullptr)
      << "Not enough space to load init binary on device";
  std::string relocated_bin = RelocateBinarySections(
      binary_path,
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

// TODO(mutinifni): overload TargetAwareWrite with different val types as need be

void* MicroSession::TargetAwareWrite(int64_t* val, size_t n,
                                     AllocatorStream* stream) {
  Slot arr_slot(stream, stream->AllocInt64Array(n));
  arr_slot.Write(val, n);
  return arr_slot.addr();
}

void* MicroSession::TargetAwareWrite(TVMArray* val, AllocatorStream* stream) {
  TVMArray arr = *val;
  Slot tarr_slot(stream, stream->AllocTVMArray());
  TargetAwareWrite(val->shape, val->ndim, stream);
  void* shape_addr = TargetAwareWrite(val->shape, val->ndim, stream);
  void* strides_addr = nullptr;
  if (val->strides != nullptr) {
    strides_addr = TargetAwareWrite(val->strides, val->ndim, stream);
  }
  void* data_addr = (uint8_t*) low_level_device()->base_addr() +
                    reinterpret_cast<std::uintptr_t>(val->data);
  arr.data = data_addr;
  arr.shape = static_cast<int64_t*>(shape_addr);
  arr.strides = static_cast<int64_t*>(strides_addr);
  tarr_slot.Write(&arr);
  return tarr_slot.addr();
}

void* MicroSession::AllocateTVMArgs(TVMArgs args) {
  std::string args_buf;
  // TODO(mutinifni): this part is a bit weird
  void* base_addr = GetAddr(args_allocator_->section_max(),
                            low_level_device()->base_addr());
  AllocatorStream* stream = new AllocatorStream(&args_buf, base_addr);
  const TVMValue* values = args.values;
  const int* type_codes = args.type_codes;
  int num_args = args.num_args;
  size_t args_offset = stream->Allocate(sizeof(TVMValue*) * num_args +
                                        sizeof(const int*) * num_args +
                                        sizeof(int));
  stream->Seek(args_offset + sizeof(TVMValue*) * num_args);
  stream->Write(type_codes, sizeof(const int*) * num_args);
  stream->Write(&num_args, sizeof(int));
  for (int i = 0; i < num_args; i++) {
    switch (type_codes[i]) {
      case kNDArrayContainer: {
        void* val_addr = TargetAwareWrite((TVMArray*) values[i].v_handle, stream);
        stream->Seek(args_offset + sizeof(TVMValue*) * i);
        stream->Write(&val_addr, sizeof(void*));
        break;
      }
      // TODO(mutinifni): implement other cases if needed
      default:
        LOG(FATAL) << "Unsupported type code for writing args: " << type_codes[i];
        break;
    }
  }
  void* stream_addr = args_allocator_->Allocate(stream->GetBufferSize());
  low_level_device()->Write(stream_addr, (void*) args_buf.c_str(),
                            stream->GetBufferSize());
  return base_addr;
}

// initializes micro session and low-level device from Python frontend
TVM_REGISTER_GLOBAL("micro._MicroInit")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::shared_ptr<MicroSession> session = MicroSession::Global();
    session->InitSession(args);
    });
}  // namespace runtime
}  // namespace tvm
