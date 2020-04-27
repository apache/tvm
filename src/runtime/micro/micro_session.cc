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
#include <tvm/runtime/device_api.h>
#include <chrono>
#include <memory>
#include <locale>
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
    TargetWordSize word_size,
    bool thumb_mode,
    bool use_device_timer,
    const std::string& server_addr,
    int port)
    : toolchain_prefix_(toolchain_prefix),
      word_size_(word_size),
      thumb_mode_(thumb_mode),
      use_device_timer_(use_device_timer),
      batch_args_encoder_(args_size, word_size) {
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
    TargetPtr base_addr;
    low_level_device_ = HostLowLevelDeviceCreate(memory_size, &base_addr);
    CHECK_EQ(base_addr.value().uint64() % word_size.bytes(), 0)
      << "base address not aligned to " << word_size.bytes() << " bytes";
    TargetPtr curr_addr = base_addr;

    section_allocators_[0] = std::make_shared<MicroSectionAllocator>(
      "text",
      DevMemRegion {
      .start = curr_addr,
      .size = text_size,
      }, word_size_);
    curr_addr += text_size;
    section_allocators_[1] = std::make_shared<MicroSectionAllocator>(
      "rodata",
      DevMemRegion {
        .start = curr_addr,
        .size = rodata_size,
      }, word_size_);
    curr_addr += rodata_size;
    section_allocators_[2] = std::make_shared<MicroSectionAllocator>(
      "data",
      DevMemRegion {
        .start = curr_addr,
        .size = data_size,
      }, word_size_);
    curr_addr += data_size;
    section_allocators_[3] = std::make_shared<MicroSectionAllocator>(
      "bss",
      DevMemRegion {
        .start = curr_addr,
        .size = bss_size,
      }, word_size_);
    curr_addr += bss_size;
    section_allocators_[4] = std::make_shared<MicroSectionAllocator>(
      "args",
      DevMemRegion {
        .start = curr_addr,
        .size = args_size,
      }, word_size_);
    curr_addr += args_size;
    section_allocators_[5] = std::make_shared<MicroSectionAllocator>(
      "heap",
      DevMemRegion {
        .start = curr_addr,
        .size = heap_size,
      }, word_size_);
    curr_addr += heap_size;
    section_allocators_[6] = std::make_shared<MicroSectionAllocator>(
      "workspace",
      DevMemRegion {
        .start = curr_addr,
        .size = workspace_size,
      }, word_size_);
    curr_addr += workspace_size;
    section_allocators_[7] = std::make_shared<MicroSectionAllocator>(
      "stack",
      DevMemRegion {
        .start = curr_addr,
        .size = stack_size,
      }, word_size_);
    curr_addr += stack_size;
  } else if (comms_method == "openocd") {
    low_level_device_ = OpenOCDLowLevelDeviceCreate(server_addr, port);
    section_allocators_[0] = std::make_shared<MicroSectionAllocator>(
      "text",
      DevMemRegion {
        .start = TargetPtr(word_size_, text_start),
        .size = text_size,
      }, word_size_);
    section_allocators_[1] = std::make_shared<MicroSectionAllocator>(
      "rodata",
      DevMemRegion {
        .start = TargetPtr(word_size_, rodata_start),
        .size = rodata_size,
      }, word_size_);
    section_allocators_[2] = std::make_shared<MicroSectionAllocator>(
      "data",
      DevMemRegion {
        .start = TargetPtr(word_size_, data_start),
        .size = data_size,
      }, word_size_);
    section_allocators_[3] = std::make_shared<MicroSectionAllocator>(
      "bss",
      DevMemRegion {
        .start = TargetPtr(word_size_, bss_start),
        .size = bss_size,
      }, word_size_);
    section_allocators_[4] = std::make_shared<MicroSectionAllocator>(
      "args",
      DevMemRegion {
        .start = TargetPtr(word_size_, args_start),
        .size = args_size,
      }, word_size_);
    section_allocators_[5] = std::make_shared<MicroSectionAllocator>(
      "heap",
      DevMemRegion {
        .start = TargetPtr(word_size_, heap_start),
        .size = heap_size,
      }, word_size_);
    section_allocators_[6] = std::make_shared<MicroSectionAllocator>(
      "workspace",
      DevMemRegion {
        .start = TargetPtr(word_size_, workspace_start),
        .size = workspace_size,
      }, word_size_);
    section_allocators_[7] = std::make_shared<MicroSectionAllocator>(
      "stack",
      DevMemRegion {
        .start = TargetPtr(word_size_, stack_start),
        .size = stack_size,
      }, word_size_);
  } else {
    LOG(FATAL) << "unsupported micro low-level device";
  }

  TargetPtr args_start_addr = GetAllocator(SectionKind::kArgs)->start_addr();
  batch_args_encoder_.set_start_addr(args_start_addr);

  runtime_symbol_map_ = LoadBinary(binary_path, false).symbol_map;

  // Patch pointers to define the bounds of the workspace section and the word
  // size (for allocation alignment).
  std::shared_ptr<MicroSectionAllocator> ws_allocator = GetAllocator(SectionKind::kWorkspace);
  DevSymbolWrite(runtime_symbol_map_, "utvm_workspace_start", ws_allocator->start_addr());
  DevSymbolWrite(runtime_symbol_map_, "utvm_workspace_end", ws_allocator->max_addr());
  if (word_size.bytes() == 4) {
    DevSymbolWrite(runtime_symbol_map_, "utvm_word_size", uint32_t(word_size.bytes()));
  } else if (word_size.bytes() == 8) {
    DevSymbolWrite(runtime_symbol_map_, "utvm_word_size", uint64_t(word_size.bytes()));
  } else {
    CHECK(false) << "Unsupported word size unexpectedly here";
  }
}

MicroSession::~MicroSession() {
  for (size_t i = 0; i < static_cast<size_t>(SectionKind::kNumKinds); i++) {
    section_allocators_[i] = nullptr;
  }
  low_level_device_ = nullptr;
}

void MicroSession::PushToTaskQueue(TargetPtr func_ptr, const TVMArgs& args) {
  if (thumb_mode_) {
    // TODO(areusch): should be |=
    func_ptr += 1;
  }
  TargetVal func_dev_addr = func_ptr.value();

  std::tuple<TargetPtr, TargetPtr> arg_field_addrs = EncoderAppend(&batch_args_encoder_, args);
  TargetVal arg_values_dev_addr{std::get<0>(arg_field_addrs).value()};
  TargetVal arg_type_codes_dev_addr{std::get<1>(arg_field_addrs).value()};

  task_queue_.push_back(
      DevTask {
        .func = func_dev_addr,
        .arg_values = arg_values_dev_addr,
        .arg_type_codes = arg_type_codes_dev_addr,
        .num_args = args.num_args
      });

  if (task_queue_.size() == MicroSession::kTaskQueueCapacity) {
    FlushTaskQueue();
  }
}

void MicroSession::FlushTaskQueue() {
  if (task_queue_.size() == 0) {
    // nothing to run
    return;
  }
  if (word_size_.bytes() == 4) {
    FlushTaskQueuePriv<StructUTVMTask32>();
  } else if (word_size_.bytes() == 8) {
    FlushTaskQueuePriv<StructUTVMTask64>();
  }
}

template <typename T>
void MicroSession::FlushTaskQueuePriv() {
  std::vector<T> prepped_tasks;
  for (const auto& task : task_queue_) {
    prepped_tasks.push_back(T(task));
  }

  // Flush `args` to device memory.
  low_level_device()->Write(
      batch_args_encoder_.start_addr(),
      reinterpret_cast<void*>(batch_args_encoder_.data()),
      batch_args_encoder_.buf_size());

  // Flush `tasks` to device memory.
  TargetPtr dev_tasks_addr = runtime_symbol_map_["utvm_tasks"];
  low_level_device()->Write(
      dev_tasks_addr,
      reinterpret_cast<void*>(prepped_tasks.data()),
      prepped_tasks.size() * sizeof(T));
  DevSymbolWrite<uint32_t>(runtime_symbol_map_, "utvm_num_tasks", prepped_tasks.size());

  TargetPtr utvm_init_addr = runtime_symbol_map_["UTVMInit"];
  TargetPtr utvm_done_addr = runtime_symbol_map_["UTVMDone"];
  if (thumb_mode_) {
    // TODO(areusch): should be |=
    utvm_init_addr += 1;
  }

  std::chrono::time_point<
    std::chrono::high_resolution_clock, std::chrono::nanoseconds> tbegin, tend;
  tbegin = std::chrono::high_resolution_clock::now();
  // std::string tmp;
  // while (tmp[0] != 'd' && tmp[0] != 'e') {
  //   std::cout << "How to proceed? [Debug / Execute] ";
  //   getline(std::cin, tmp);
  //   CHECK(std::cin.good()) << "Stdin closed";
  //   tmp[0] = std::tolower(tmp[0]);
  // }
  // if (tmp[0] == 'd') {
  //   std::cout << "Launch debugger; [Enter] to resume automated execution";
  //   getline(std::cin, tmp);
  // } else {
  low_level_device()->Execute(utvm_init_addr, utvm_done_addr);
  // }
  tend = std::chrono::high_resolution_clock::now();

  // Check if there was an error during execution.  If so, log it.
  CheckDeviceError();

  if (use_device_timer_) {
    uint64_t sum = 0;
    std::vector<uint32_t> times;
    times.resize(task_queue_.size());
    low_level_device()->Read(runtime_symbol_map_["utvm_task_times"],
                             times.data(),
                             task_queue_.size() * sizeof(uint32_t));
    int i = 0;
    for (uint32_t time : times) {
      LOG(INFO) << "Time " << i++ << ": " << time;
      sum += time;
    }
    last_batch_time_ += static_cast<double>(sum) / 1e3;
  } else {
    last_batch_time_ += std::chrono::duration_cast<std::chrono::duration<double> >
        (tend - tbegin).count() * 1000;
    // TODO(weberlo): Reading internal data structure is hacky.
    uint64_t sum = 0;
    std::vector<uint32_t> times;
    times.resize(task_queue_.size());
    low_level_device()->Read(runtime_symbol_map_["utvm_task_times"],
                             times.data(),
                             task_queue_.size() * sizeof(uint32_t));
    for (uint32_t time : times) {
      sum += time;
    }
    last_batch_cycles_ += static_cast<double>(sum);
  }

  batch_args_encoder_.Clear();
  task_queue_.clear();
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
  SymbolMap symbol_map {relocated_bin, toolchain_prefix_, word_size_};

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

std::tuple<TargetPtr, TargetPtr> MicroSession::EncoderAppend(
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
        MicroDevSpace* dev_arr_ptr = reinterpret_cast<MicroDevSpace*>(old_data);
        base_arr_handle->data = reinterpret_cast<void*>(dev_arr_ptr->data.value().uint64());
        // Now, encode the unwrapped version.
        void* arr_ptr = nullptr;
        if (word_size_.bytes() == 4) {
          arr_ptr = EncoderAppend<TVMArray32>(encoder, *base_arr_handle).cast_to<void*>();
        } else if (word_size_.bytes() == 8) {
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
TargetPtr MicroSession::EncoderAppend(TargetDataLayoutEncoder* encoder, const DLTensor& arr) {
  auto tvm_arr_slot = encoder->Alloc<T>();
  auto shape_slot = encoder->Alloc<int64_t>(arr.ndim);

  // `shape` and `strides` are stored on the host, so we need to write them to
  // the device first. The `data` field is already allocated on the device and
  // is a device pointer, so we don't need to write it.
  shape_slot.WriteArray(arr.shape, arr.ndim);
  TargetPtr shape_dev_addr = shape_slot.start_addr();
  TargetPtr strides_dev_addr = TargetPtr(word_size_, nullptr);
  if (arr.strides != nullptr) {
    auto stride_slot = encoder->Alloc<int64_t>(arr.ndim);
    stride_slot.WriteArray(arr.strides, arr.ndim);
    strides_dev_addr = stride_slot.start_addr();
  }

  T dev_arr(
      TargetVal { word_size_.bits(), reinterpret_cast<uint64_t>(arr.data) },
      arr.ctx,
      arr.ndim,
      arr.dtype,
      shape_dev_addr.value(),
      strides_dev_addr.value(),
      TargetVal { word_size_.bits(), arr.byte_offset });
  CHECK(dev_arr.ctx.device_type == static_cast<DLDeviceType>(kDLMicroDev))
    << "attempt to write DLTensor with non-micro device type";
  // Update the device type to CPU, because from the microcontroller's
  // perspective, it is.
  dev_arr.ctx.device_type = DLDeviceType::kDLCPU;
  tvm_arr_slot.WriteValue(dev_arr);
  return tvm_arr_slot.start_addr();
}

// TODO(weberlo): switch over entirely to error codes that expand to error
// messages on the host side.
void MicroSession::CheckDeviceError() {
  int32_t last_error = DevSymbolRead<int32_t>(runtime_symbol_map_, "utvm_last_error");

  if (last_error) {
    if (!use_device_timer_ &&
        (last_error == UTVM_ERR_TIMER_OVERFLOW ||
         last_error == UTVM_ERR_TIMER_NOT_IMPLEMENTED)) {
      // these errors don't matter if we're not using the on-device timer
      return;
    }
    std::string err_msg;
    switch (last_error) {
      case UTVM_ERR_NOT_FINISHED:
        err_msg = "execution timed out";
        break;
      case UTVM_ERR_TIMER_NOT_IMPLEMENTED:
        err_msg = "timer is not implemented for the target device";
        break;
      case UTVM_ERR_TIMER_OVERFLOW:
        // TODO(weberlo): this should be remedied by using interrupts to accumulate the
        // timer into a larger datatype (ARM timers are only 24 bits)
        err_msg = "timer overflowed during execution";
        break;
      case UTVM_ERR_WS_DOUBLE_FREE:
        err_msg = "free called with no active workspace allocations";
        break;
      case UTVM_ERR_WS_OUT_OF_SPACE:
        err_msg = "ran out of space in workspace section";
        break;
      case UTVM_ERR_WS_TOO_MANY_ALLOCS:
        err_msg = "exceeded number of allocs the runtime can keep track of";
        break;
      case UTVM_ERR_WS_ZERO_SIZE_ALLOC:
        err_msg = "attempt to allocate scratchpad of size zero";
        break;
      case UTVM_ERR_WS_UNALIGNED_START:
        err_msg = "start of workspace section is not word-aligned";
        break;
      case UTVM_ERR_WS_UNALIGNED_ALLOC_SIZE:
        err_msg = "scratchpad allocation size is not a multiple of the word size";
        break;
      default:
        err_msg = "unknown error code";
        break;
    }
    LOG(FATAL) << "error during micro function execution:\n"
               << "  error ID: " << std::dec << last_error << std::endl
               << "  error message: " << err_msg;
  }
}

void MicroSession::PatchImplHole(const SymbolMap& symbol_map, const std::string& func_name) {
  TargetPtr runtime_impl_addr = runtime_symbol_map_[func_name];
  if (thumb_mode_) {
    runtime_impl_addr += 1;
  }
  std::ostringstream func_name_underscore;
  func_name_underscore << func_name << "_";
  DevSymbolWrite(symbol_map, func_name_underscore.str(), runtime_impl_addr);
}

std::string MicroSession::ReadString(TargetPtr str_addr) {
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

TargetPtr MicroSession::AllocateInSection(SectionKind type, size_t size) {
  return GetAllocator(type)->Allocate(size);
}

void MicroSession::FreeInSection(SectionKind type, TargetPtr addr) {
  return GetAllocator(type)->Free(addr);
}

template <typename T>
T MicroSession::DevSymbolRead(const SymbolMap& symbol_map, const std::string& symbol) {
  TargetPtr sym_addr = symbol_map[symbol];
  T result;
  low_level_device()->Read(sym_addr, &result, sizeof(T));
  return result;
}

void MicroSession::DevSymbolWrite(const SymbolMap& symbol_map,
                                  const std::string& symbol,
                                  const TargetPtr& ptr) {
  if (word_size_.bytes() == 4) {
    DevSymbolWrite(symbol_map, symbol, ptr.value().uint32());
  } else if (word_size_.bytes() == 8) {
    DevSymbolWrite(symbol_map, symbol, ptr.value().uint64());
  } else {
    CHECK(false) << "Unsupported word size unexpectedly here";
  }
}

template <typename T>
void MicroSession::DevSymbolWrite(const SymbolMap& symbol_map,
                                  const std::string& symbol,
                                  const T& value) {
  TargetPtr sym_addr = symbol_map[symbol];
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
    // TODO(weberlo): add a `clear_batch_timer` func
  } else if (name == "get_last_batch_time") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = this->GetLastBatchTime();
    });
    // TODO(weberlo): remove this func
  } else if (name == "get_last_batch_cycles") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      *rv = this->GetLastBatchCycles();
    });
  } else {
    return PackedFunc();
  }
}

TVM_REGISTER_GLOBAL("micro._GetMicroTimeEvaluator")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  PackedFunc pf = args[0];
  TVMContext ctx = args[1];
  uint64_t number = args[2];
  uint64_t repeat = args[3];

  auto ftimer = [pf, ctx, number, repeat](TVMArgs args, TVMRetValue *rv) mutable {
    TVMRetValue temp;
    std::ostringstream os;

    for (unsigned int i = 0; i < repeat; ++i) {
      // start timing
      CHECK(number < MicroSession::kTaskQueueCapacity)
        << "`number` must be less than uTVM task queue capacity";
      for (unsigned int j = 0; j < number; ++j) {
        pf.CallPacked(args, &temp);
      }
      ObjectPtr<MicroSession> session = MicroSession::Current();
      DeviceAPI::Get(ctx)->StreamSync(ctx, nullptr);
      double time_per_batch = session->GetLastBatchTime() / number;
      os.write(reinterpret_cast<char*>(&time_per_batch), sizeof(time_per_batch));
    }
    std::string blob = os.str();
    TVMByteArray arr;
    arr.size = blob.length();
    arr.data = blob.data();
    // return the time.
    *rv = arr;
  };
  *rv = PackedFunc(ftimer);
});


// create micro session and low-level device from Python frontend
TVM_REGISTER_GLOBAL("micro._CreateSession")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    const std::string& comms_method = args[0];
    const std::string& binary_path = args[1];
    const std::string& toolchain_prefix = args[2];
    uint64_t text_start = args[3];
    size_t text_size = uint64_t(args[4]);
    uint64_t rodata_start = args[5];
    size_t rodata_size = uint64_t(args[6]);
    uint64_t data_start = args[7];
    size_t data_size = uint64_t(args[8]);
    uint64_t bss_start = args[9];
    size_t bss_size = uint64_t(args[10]);
    uint64_t args_start = args[11];
    size_t args_size = uint64_t(args[12]);
    uint64_t heap_start = args[13];
    size_t heap_size = uint64_t(args[14]);
    uint64_t workspace_start = args[15];
    size_t workspace_size = uint64_t(args[16]);
    uint64_t stack_start = args[17];
    size_t stack_size = uint64_t(args[18]);
    TargetWordSize word_size{uint64_t(args[19])};
    bool thumb_mode = args[20];
    bool use_device_timer = args[21];
    const std::string& server_addr = args[22];
    int port = args[23];
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
        use_device_timer,
        server_addr,
        port);
    *rv = Module(session);
    });

}  // namespace runtime
}  // namespace tvm
