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
      new MicroSectionAllocator(kTextStart,
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
  // TODO(weberlo): We shouldn't need a workspace allocator, because every
  // library will share the same one.
  workspace_allocator_ = std::unique_ptr<MicroSectionAllocator>(
      new MicroSectionAllocator(kWorkspaceStart,
                                dev_base_offset(kMemorySize)));
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

dev_base_offset MicroSession::AllocateInSection(SectionKind type, size_t size) {
  switch (type) {
    case kText:
      return text_allocator_->Allocate(size);
    case kData:
      return data_allocator_->Allocate(size);
    case kBss:
      return bss_allocator_->Allocate(size);
    case kArgs:
      return args_allocator_->Allocate(size);
    case kStack:
      return stack_allocator_->Allocate(size);
    case kHeap:
      return heap_allocator_->Allocate(size);
    case kWorkspace:
      return workspace_allocator_->Allocate(size);
    default:
      LOG(FATAL) << "Unsupported section type during allocation";
      return dev_base_offset(nullptr);
  }
}

void MicroSession::FreeInSection(SectionKind type, dev_base_offset ptr) {
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

std::string MicroSession::ReadString(dev_base_offset str_offset) {
  std::stringstream result;
  dev_base_offset str_data_offset;
  low_level_device()->Read(str_offset, (void*) (&str_data_offset.val_), sizeof(void*));
  std::cout << "str_data_offset: " << std::hex << str_data_offset.val_ << std::endl;
  static char buf[256];
  size_t i = 256;
  while (i == 256) {
    low_level_device()->Read(str_data_offset, (void*) buf, 256);
    i = 0;
    while (i < 256) {
      if (buf[i] == 0) break;
      result << buf[i];
      i++;
    }
    str_offset.val_ += i;
  }
  return result.str();
}

void MicroSession::PushToExecQueue(dev_base_offset func, TVMArgs args) {
  uint64_t (*func_dev_addr)(void*, void*, int32_t) =
      reinterpret_cast<uint64_t (*)(void*, void*, int32_t)>(
      GetAddr(func, low_level_device()->base_addr()).val_);

  // Create an allocator stream for the memory region after the most recent
  // allocation in the args section.
  dev_addr args_addr = GetAddr(args_allocator_->section_max(), low_level_device()->base_addr());
  TargetDataLayoutEncoder encoder(args_addr);

  UTVMArgs u_args = {
      .values = const_cast<TVMValue*>(args.values),
      .type_codes = const_cast<int*>(args.type_codes),
      .num_args = args.num_args,
  };
  EncoderWrite(&encoder, &u_args);
  // Flush `stream` to device memory.
  dev_base_offset stream_dev_offset = args_allocator_->Allocate(encoder.buf_size());
  low_level_device()->Write(stream_dev_offset,
                            reinterpret_cast<void*>(const_cast<uint8_t*>(encoder.data())),
                            encoder.buf_size());

  UTVMTask task = {
      .func = func_dev_addr,
      .args = reinterpret_cast<UTVMArgs*>(args_addr.val_),
  };
  // TODO(mutinifni): handle bits / endianness
  dev_base_offset task_dev_addr = init_symbol_map_["task"];
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
  CHECK(init_text_start_.val_ != 0 &&
        init_data_start_.val_ != 0 &&
        init_bss_start_.val_ != 0)
      << "Not enough space to load init binary on device";
  const dev_base_addr base_addr = low_level_device()->base_addr();
  std::string relocated_bin = RelocateBinarySections(
      init_binary_path_,
      (void*) GetAddr(init_text_start_, base_addr).val_,
      (void*) GetAddr(init_data_start_, base_addr).val_,
      (void*) GetAddr(init_bss_start_, base_addr).val_);
  std::string text_contents = ReadSection(relocated_bin, kText);
  std::string data_contents = ReadSection(relocated_bin, kData);
  std::string bss_contents = ReadSection(relocated_bin, kBss);
  low_level_device()->Write(init_text_start_, &text_contents[0], init_text_size_);
  low_level_device()->Write(init_data_start_, &data_contents[0], init_data_size_);
  low_level_device()->Write(init_bss_start_, &bss_contents[0], init_bss_size_);
  // obtain init stub binary metadata
  init_symbol_map_ = SymbolMap(relocated_bin, base_addr);
  utvm_main_symbol_addr_ = init_symbol_map_["UTVMMain"];
  utvm_done_symbol_addr_ = init_symbol_map_["UTVMDone"];
}

void MicroSession::SetInitBinaryPath(std::string path) {
  init_binary_path_ = path;
}

dev_addr MicroSession::EncoderWrite(TargetDataLayoutEncoder* encoder, UTVMArgs* args) {
  auto utvm_args_slot = encoder->Alloc<UTVMArgs>();

  const int* type_codes = args->type_codes;
  int num_args = args->num_args;

  auto tvm_vals_slot = encoder->Alloc<TVMValue*>(num_args);
  auto type_codes_slot = encoder->Alloc<const int>(num_args);

  for (int i = 0; i < num_args; i++) {
    switch (type_codes[i]) {
      case kNDArrayContainer: {
        TVMValue* val_addr = reinterpret_cast<TVMValue*>(
            EncoderWrite(encoder, reinterpret_cast<TVMArray*>(args->values[i].v_handle)).val_);
        tvm_vals_slot.Write(&val_addr);
        break;
      }
      // TODO(mutinifni): implement other cases if needed
      default:
        LOG(FATAL) << "Unsupported type code for writing args: " << type_codes[i];
        break;
    }
  }
  type_codes_slot.Write(type_codes, num_args);

  UTVMArgs dev_args = {
    .values = reinterpret_cast<TVMValue*>(tvm_vals_slot.start_addr().val_),
    .type_codes = reinterpret_cast<int*>(type_codes_slot.start_addr().val_),
    .num_args = num_args,
  };
  utvm_args_slot.Write(&dev_args);
  return utvm_args_slot.start_addr();
}

dev_addr MicroSession::EncoderWrite(TargetDataLayoutEncoder* encoder, TVMArray* arr) {
  auto tvm_arr_slot = encoder->Alloc<TVMArray>();
  auto shape_slot = encoder->Alloc<int64_t>(arr->ndim);

  // `shape` and `strides` are stored on the host, so we need to write them to
  // the device first. The `data` field is already allocated on the device and
  // is a device pointer, so we don't need to write it.
  shape_slot.Write(arr->shape, arr->ndim);
  dev_addr shape_addr = shape_slot.start_addr();
  dev_addr strides_addr = dev_addr(nullptr);
  if (arr->strides != nullptr) {
    auto stride_slot = encoder->Alloc<int64_t>(arr->ndim);
    stride_slot.Write(arr->strides, arr->ndim);
    strides_addr = stride_slot.start_addr();
  }

  // Copy `arr`, update the copy's pointers to be device pointers, then
  // write the copy to `tvm_arr_slot`.
  TVMArray dev_arr = *arr;
  // Add the base address of the device to the array's data's device offset to
  // get a device address.
  dev_arr.data = reinterpret_cast<uint8_t*>(low_level_device()->base_addr().val_) +
                  reinterpret_cast<std::uintptr_t>(arr->data);
  dev_arr.shape = reinterpret_cast<int64_t*>(shape_addr.val_);
  dev_arr.strides = reinterpret_cast<int64_t*>(strides_addr.val_);
  tvm_arr_slot.Write(&dev_arr);
  return tvm_arr_slot.start_addr();
}

// initializes micro session and low-level device from Python frontend
TVM_REGISTER_GLOBAL("micro._MicroInit")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::shared_ptr<MicroSession> session = MicroSession::Global();
    session->InitSession(args);
    });
}  // namespace runtime
}  // namespace tvm
