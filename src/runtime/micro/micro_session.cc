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
                                kRodataStart));
  rodata_allocator_ = std::unique_ptr<MicroSectionAllocator>(
      new MicroSectionAllocator(kRodataStart,
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
  CHECK(!init_binary_path_.empty()) << "init library not initialized";
  init_stub_info_ = LoadBinary(init_binary_path_);
  utvm_main_symbol_addr_ = init_stub_info_.symbol_map["UTVMMain"];
  utvm_done_symbol_addr_ = init_stub_info_.symbol_map["UTVMDone"];

  // TODO(weberlo): Move the patching below to the init stub.
  dev_base_offset workspace_start_hole_offset = init_symbol_map()["workspace_start"];
  dev_base_offset workspace_curr_hole_offset = init_symbol_map()["workspace_curr"];
  void* workspace_hole_fill = (void*) (kWorkspaceStart.val_ + low_level_device_->base_addr().val_);

  void* tmp;
  low_level_device()->Read(workspace_start_hole_offset, &tmp, sizeof(void*));
  std::cout << "workspace start addr (before): 0x" << std::hex << tmp << std::endl;
  low_level_device()->Write(workspace_start_hole_offset, &workspace_hole_fill, sizeof(void*));
  low_level_device()->Read(workspace_start_hole_offset, &tmp, sizeof(void*));
  std::cout << "workspace start addr (after): 0x" << std::hex << tmp << std::endl;

  low_level_device()->Read(workspace_curr_hole_offset, &tmp, sizeof(void*));
  std::cout << "workspace curr addr (before): 0x" << std::hex << tmp << std::endl;
  low_level_device()->Write(workspace_curr_hole_offset, &workspace_hole_fill, sizeof(void*));
  low_level_device()->Read(workspace_curr_hole_offset, &tmp, sizeof(void*));
  std::cout << "workspace curr addr (after): 0x" << std::hex << tmp << std::endl;

  std::cout << "SESSION INIT SUCCESS" << std::endl;
}

dev_base_offset MicroSession::AllocateInSection(SectionKind type, size_t size) {
  switch (type) {
    case kText:
      return text_allocator_->Allocate(size);
    case kRodata:
      return rodata_allocator_->Allocate(size);
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
      return;
    case kRodata:
      rodata_allocator_->Free(ptr);
      return;
    case kData:
      data_allocator_->Free(ptr);
      return;
    case kBss:
      bss_allocator_->Free(ptr);
      return;
    case kArgs:
      args_allocator_->Free(ptr);
      return;
    case kStack:
      stack_allocator_->Free(ptr);
      return;
    case kHeap:
      heap_allocator_->Free(ptr);
      return;
    case kWorkspace:
      workspace_allocator_->Free(ptr);
      return;
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
  dev_base_offset task_dev_addr = init_symbol_map()["task"];
  std::cout << "PREPARE SHIP" << std::endl;
  low_level_device()->Write(task_dev_addr, &task, sizeof(task));
  std::cout << "prepare ship" << std::endl;
  low_level_device()->Execute(utvm_main_symbol_addr_, utvm_done_symbol_addr_);
  std::cout << "for ludicorosufs spedddd" << std::endl;
}

BinaryInfo MicroSession::LoadBinary(std::string binary_path) {
  SectionLocation text;
  SectionLocation rodata;
  SectionLocation data;
  SectionLocation bss;

  text.size = GetSectionSize(binary_path, kText);
  rodata.size = GetSectionSize(binary_path, kRodata);
  data.size = GetSectionSize(binary_path, kData);
  bss.size = GetSectionSize(binary_path, kBss);

  text.start = AllocateInSection(kText, text.size);
  rodata.start = AllocateInSection(kRodata, rodata.size);
  data.start = AllocateInSection(kData, data.size);
  bss.start = AllocateInSection(kBss, bss.size);
  std::cout << "binary path: " << binary_path << std::endl;
  std::cout << "  text size: " << text.size << std::endl;
  std::cout << "  rodata size: " << rodata.size << std::endl;
  std::cout << "  data size: " << data.size << std::endl;
  std::cout << "  bss size: " << bss.size << std::endl;
  std::cout << std::endl;
  CHECK(text.start.val_ != 0 && rodata.start.val_ != 0 && data.start.val_ != 0 && bss.start.val_ != 0)
      << "not enough space to load module on device";
  const dev_base_addr base_addr = low_level_device_->base_addr();
  std::string relocated_bin = RelocateBinarySections(
      binary_path, (void*)GetAddr(text.start, base_addr).val_,
      (void*)GetAddr(rodata.start, base_addr).val_, (void*)GetAddr(data.start, base_addr).val_,
      (void*)GetAddr(bss.start, base_addr).val_);
  std::string text_contents = ReadSection(relocated_bin, kText);
  std::string rodata_contents = ReadSection(relocated_bin, kRodata);
  std::string data_contents = ReadSection(relocated_bin, kData);
  std::string bss_contents = ReadSection(relocated_bin, kBss);
  low_level_device_->Write(text.start, &text_contents[0], text.size);
  low_level_device_->Write(rodata.start, &rodata_contents[0], rodata.size);
  low_level_device_->Write(data.start, &data_contents[0], data.size);
  low_level_device_->Write(bss.start, &bss_contents[0], bss.size);
  SymbolMap symbol_map {relocated_bin, base_addr};
  return BinaryInfo{
      .text = text,
      .rodata = rodata,
      .data = data,
      .bss = bss,
      .symbol_map = symbol_map,
  };
}

void MicroSession::SetInitBinaryPath(std::string path) {
  init_binary_path_ = path;
}

dev_addr MicroSession::EncoderWrite(TargetDataLayoutEncoder* encoder, UTVMArgs* args) {
  std::cout << "A" << std::endl;
  auto utvm_args_slot = encoder->Alloc<UTVMArgs>();

  const int* type_codes = args->type_codes;
  int num_args = args->num_args;

  std::cout << "B" << std::endl;
  auto tvm_vals_slot = encoder->Alloc<TVMValue*>(num_args);
  std::cout << "BAA" << std::endl;
  auto type_codes_slot = encoder->Alloc<const int>(num_args);
  std::cout << "BAB" << std::endl;

  std::cout << "type codes: " << type_codes[0] << std::endl;
  for (int i = 0; i < num_args; i++) {
    switch (type_codes[i]) {
      case kNDArrayContainer: {
        std::cout << "BA" << std::endl;
        TVMValue* val_addr = reinterpret_cast<TVMValue*>(
            EncoderWrite(encoder, reinterpret_cast<TVMArray*>(args->values[i].v_handle)).val_);
        std::cout << "BB" << std::endl;
        tvm_vals_slot.Write(&val_addr);
        std::cout << "BC" << std::endl;
        break;
      }
      case kArrayHandle: {
        std::cout << "CA" << std::endl;
        TVMValue* val_addr = reinterpret_cast<TVMValue*>(
            EncoderWrite(encoder, reinterpret_cast<TVMArray*>(args->values[i].v_handle)).val_);
        std::cout << "CB" << std::endl;
        tvm_vals_slot.Write(&val_addr);
        std::cout << "CC" << std::endl;
        break;
      }
      // TODO(mutinifni): implement other cases if needed
      default:
        CHECK(false) << "Unsupported type code for writing args: " << type_codes[i];
        LOG(FATAL) << "Unsupported type code for writing args: " << type_codes[i];
        break;
    }
  }
  type_codes_slot.Write(type_codes, num_args);

  std::cout << "C" << std::endl;
  UTVMArgs dev_args = {
    .values = reinterpret_cast<TVMValue*>(tvm_vals_slot.start_addr().val_),
    .type_codes = reinterpret_cast<int*>(type_codes_slot.start_addr().val_),
    .num_args = num_args,
  };
  utvm_args_slot.Write(&dev_args);
  std::cout << "D" << std::endl;
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
