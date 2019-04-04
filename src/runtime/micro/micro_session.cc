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

void MicroSession::InitSession(TVMArgs args) {
  if (args[0] == "host") {
    low_level_device_ = HostLowLevelDeviceCreate(kMemorySize);
  } else if (args[0] == "openocd") {
    low_level_device_ = OpenOCDLowLevelDeviceCreate(args[1]);
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
  AllocateTVMArgs(args);
  int num_args = args.num_args;
  // TODO: setup init stub args to execute
  void* func_addr = GetAddr(func, low_level_device()->base_addr());
  //low_level_device()->Write(GetSymbol("UTVM_task", low_level_device()->base_addr()),
  // UTVMMain()
  // UTVMTask task
  void* func_end = GetSymbol(init_symbol_map_, "UTVMDone",
                             low_level_device()->base_addr());
  low_level_device()->Execute(func, func_end);
}

void MicroSession::LoadInitStub() {
  // TODO: this is the utvm device binary, probably alright to hard code (need path)
  std::string binary = "utvm_runtime.o";
  init_text_size_ = GetSectionSize(binary, kText);
  init_data_size_ = GetSectionSize(binary, kData);
  init_bss_size_ = GetSectionSize(binary, kBss);
  init_text_start_ = AllocateInSection(kText, init_text_size_);
  init_data_start_ = AllocateInSection(kData, init_data_size_);
  init_bss_start_ = AllocateInSection(kBss, init_bss_size_);
  CHECK(init_text_start_ != nullptr &&
        init_data_start_ != nullptr && 
        init_bss_start_ != nullptr)
    << "Not enough space to load init binary on device";
  std::string relocated_bin = RelocateBinarySections(binary,
                                                     init_text_start_,
                                                     init_data_start_,
                                                     init_bss_start_);
  std::string text_contents = ReadSection(relocated_bin, kText);
  std::string data_contents = ReadSection(relocated_bin, kData);
  std::string bss_contents = ReadSection(relocated_bin, kBss);
  low_level_device()->Write(init_text_start_, &text_contents[0], init_text_size_);
  low_level_device()->Write(init_data_start_, &data_contents[0], init_data_size_);
  low_level_device()->Write(init_bss_start_, &bss_contents[0], init_bss_size_);
  init_symbol_map_ = GetSymbolMap(relocated_bin);
}

// TODO: make target aware write functions for everything
// TODO: these need to be device-based sizeof
// TODO: what about kBytes, kHandle, kNull, kNodeHandle, kArrayHandle, kTVMType, kFuncHandle, kModuleHandle?
void MicroSession::TargetAwareWrite(int64_t val, AllocatorStream* stream) {
}

void MicroSession::TargetAwareWrite(uint64_t val, AllocatorStream* stream) {
}

void MicroSession::TargetAwareWrite(double val, AllocatorStream* stream) {
}

void MicroSession::TargetAwareWrite(const char* val, AllocatorStream* stream) {
}

void MicroSession::TargetAwareWrite(TVMType val, AllocatorStream* stream) {
}

void MicroSession::TargetAwareWrite(TVMContext* val, AllocatorStream* stream) {
}

// TODO: rename based on func arg
void MicroSession::TargetAwareWrite(TVMArray* val, AllocatorStream* stream) {
  TVMArray* tarr = (TVMArray*)(values[i].v_handle);
  size_t tarr_offset = stream->Allocate(sizeof(TVMArray));
  size_t shape_size = 1;
  for (int dim = 0; dim < tarr->ndim; dim++)
    shape_size *= tarr->shape[dim];
  size_t shape_offset = stream->Allocate(sizeof(int64_t) * tarr->ndim);
  stream->Seek(shape_offset);
  stream->Write(tarr->shape, sizeof(int64_t) * tarr->ndim);
  size_t strides_offset = 0;
  if (tarr->strides != NULL) {
    strides_offset = stream->Allocate(sizeof(int64_t) * tarr->ndim);
    stream->Seek(strides_offset);
    stream->Write(tarr->strides, sizeof(int64_t) * tarr->ndim);
  }
  stream->Seek(tarr_offset);
  stream->Write(tarr, sizeof(TVMArray)); 
  void* data_addr = (uint8_t*) base_addr +
                    reinterpret_cast<std::uintptr_t>(tarr->data) -
                    kArgsStart;
  void* shape_addr = (uint8_t*) base_addr + shape_offset;
  void* strides_addr = NULL;
  if (tarr->strides != NULL)
    strides_addr = (uint8_t*) base_addr + strides_offset;
  stream->Seek(tarr_offset);
  stream->Write(&data_addr, sizeof(void*));
  stream->Seek(tarr_offset + sizeof(void*) + sizeof(DLContext) +
               sizeof(int) + sizeof(DLDataType));
  stream->Write(&shape_addr, sizeof(void*));
  stream->Write(&strides_addr, sizeof(void*));
  void* tarr_addr = (uint8_t*) base_addr + tarr_offset;
  stream->Seek(args_offset + sizeof(TVMValue*) * i);
  stream->Write(&tarr_addr, sizeof(void*));
}

void MicroSession::AllocateTVMArgs(TVMArgs args) {
  std::string args_buf;
  AllocatorStream* stream = new AllocatorStream(&args_buf);
  // TODO: this needs to be args section base addr, not lldevice base_addr
  // but make it generic by allocating a sufficiently large enough region first?
  const void* base_addr = low_level_device()->base_addr();
  const TVMValue* values = args.values;
  const int* type_codes = args.type_codes;
  int num_args = args.num_args;
  size_t args_offset = stream->Allocate(sizeof(TVMValue*) * num_args +
                                        sizeof(const int*) * num_args +
                                        sizeof(int));
  stream->Seek(args_offset + sizeof(TVMValue*) * num_args);
  stream->Write(type_codes, sizeof(const int*) * num_args);
  stream->Write(&num_args, sizeof(int));
  // TODO: implement all cases
  for (int i = 0; i < num_args; i++) {
    switch(type_codes[i]) {
      case kDLInt:
        TargetAwareWrite(values[i].v_int64, stream);
        break;
      case kDLUInt:
        // TODO: is this fine? (how is uint passed?)
        TargetAwareWrite(values[i].v_int64, stream);
        break;
      case kDLFloat:
        TargetAwareWrite(values[i].v_float64, stream);
        break;
      case kStr:
        TargetAwareWrite(values[i].v_str, stream);
        break;
      case kBytes:
        printf("was bytes\n");
        break;
      case kHandle:
        printf("was handle\n");
        break;
      case kNull:
        printf("was null\n");
        break;
      case kNodeHandle:
        printf("was nodehandle\n");
        break;
      case kArrayHandle:
        printf("was arrayhandle\n");
        break;
      case kTVMType:
        TargetAwareWrite(values[i].v_type, stream);
        break;
      case kTVMContext:
        TargetAwareWrite(values[i].v_ctx, stream);
        break;
      case kFuncHandle:
        printf("was funchandle\n");
        break;
      case kModuleHandle:
        printf("was modulehandle\n");
        break;
      case kNDArrayContainer:
        TargetAwareWrite((TVMArray*) values[i].v_handle, stream);
        break;
      default:
        LOG(FATAL) << "Could not process type code: " << type_codes[i];
        break;
    }
  }
}

// initializes micro session and low-level device from Python frontend
TVM_REGISTER_GLOBAL("micro.init")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  std::shared_ptr<MicroSession> session = MicroSession::Global();
  session->InitSession(args);
  });
}  // namespace runtime
}  // namespace tvm
