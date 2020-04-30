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
 * \file micro_session.h
 * \brief session to manage multiple micro modules
 *
 * Each session consists of an interaction with a *single* logical device.
 * Within that interaction, multiple TVM modules can be loaded on the logical
 * device.
 *
 * Multiple sessions can exist simultaneously, but there is only ever one
 * *active* session. The idea of an active session mainly has implications for
 * the frontend, in that one must make a session active in order to allocate
 * new TVM objects on it. Aside from that, previously allocated objects can be
 * used even if the session which they belong to is not currently active.
 */
#ifndef TVM_RUNTIME_MICRO_MICRO_SESSION_H_
#define TVM_RUNTIME_MICRO_MICRO_SESSION_H_

#include "micro_common.h"
#include "micro_section_allocator.h"

#include <tvm/runtime/registry.h>
#include <tvm/runtime/c_runtime_api.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <tuple>

#include "low_level_device.h"
#include "target_data_layout_encoder.h"

namespace tvm {
namespace runtime {

struct DevTask;

/*!
 * \brief session for facilitating micro device interaction
 */
class MicroSession : public ModuleNode {
 public:
  /*!
   * \brief Get member function to front-end
   * \param name The name of the function.
   * \param sptr_to_self The pointer to the module node.
   * \return The corresponding member function.
   */
  virtual PackedFunc GetFunction(const std::string& name,
                                 const ObjectPtr<Object>& sptr_to_self);

  // todo having this decoupled from the value in utvm_runtime.c gives me stress dreams
  static const size_t kTaskQueueCapacity = 20;

  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const final {
    return "MicroSession";
  }

  /*!
   * \brief creates session by setting up a low-level device and initting allocators for it
   * \param comms_method method of communication with the device (e.g., "openocd")
   * \param binary_path file system path to the runtime binary
   * \param toolchain_prefix GCC toolchain prefix
   * \param text_start text section start address
   * \param text_size text section size
   * \param rodata_start text section start address
   * \param rodata_size rodata section size
   * \param data_start data section start address
   * \param data_size data section size
   * \param bss_start bss section start address
   * \param bss_size bss section size
   * \param args_start args section start address
   * \param args_size args section size
   * \param heap_start heap section start address
   * \param heap_size heap section size
   * \param workspace_start workspace section start address
   * \param workspace_size workspace section size
   * \param stack_start stack section start address
   * \param stack_size stack section size
   * \param word_size_bytes number of bytes in a word on the target device
   * \param thumb_mode whether the target device requires a thumb-mode bit on function addresses
   * \param server_addr address of the OpenOCD server to connect to (if `comms_method == "openocd"`)
   * \param port port of the OpenOCD server to connect to (if `comms_method == "openocd"`)
   */
  MicroSession(
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
      int port);

  /*!
   * \brief destructor
   */
  ~MicroSession();

  static ObjectPtr<MicroSession>& Current();

  /*!
   * \brief sets up runtime metadata for `func` and copies arguments for on-device execution
   * \param func address of the function to be executed
   * \param args args to the packed function
   * \return elapsed time during function execution on the device
   */
  void PushToTaskQueue(TargetPtr func, const TVMArgs& args);

  /*!
   * \brief serialize runtime metadata to the device for enqueued tasks and execute
   * \return elapsed time during function execution on the device
   */
  void FlushTaskQueue();

  /*!
   * \brief TODO
   */
  template <typename T>
  void FlushTaskQueuePriv();

  /*!
   * \brief loads binary onto device
   * \param binary_path path to binary object file
   * \param patch_dylib_pointers whether to patch runtime API function pointers
   * \return info about loaded binary
   */
  BinaryInfo LoadBinary(const std::string& binary_path, bool patch_dylib_pointers);

  /*!
   * \brief allocate memory in section
   * \param type type of section to allocate in
   * \param size size of allocated memory in bytes
   * \return pointer to allocated memory region in section, nullptr if out of space
   */
  TargetPtr AllocateInSection(SectionKind type, size_t size);

  /*!
   * \brief free prior allocation from section
   * \param type type of section to allocate in
   * \param addr device address of allocated memory
   */
  void FreeInSection(SectionKind type, TargetPtr addr);

  /*!
   * \brief read string from device to host
   * \param str_addr device address of first character of string
   * \return host copy of device string that was read
   */
  std::string ReadString(TargetPtr str_addr);

  /*!
  * \brief read value of symbol from device memory
  * \param symbol_map symbol map to read location of symbol from
  * \param symbol name of symbol being read from
  * \return value at symbol in memory
  */
  template <typename T>
  T DevSymbolRead(const SymbolMap& symbol_map, const std::string& symbol);

  /*!
   * \brief write pointer value into device memory corresponding to symbol
  * \param symbol_map symbol map to read location of symbol from
  * \param symbol name of symbol being written to
  * \param ptr pointer value to write into symbol
   */
  void DevSymbolWrite(const SymbolMap& symbol_map,
                      const std::string& symbol,
                      const TargetPtr& ptr);

  /*!
  * \brief write value into device memory corresponding to symbol
  * \param symbol_map symbol map to read location of symbol from
  * \param symbol name of symbol being written to
  * \param value value being written into symbol
   */
  template <typename T>
  void DevSymbolWrite(const SymbolMap& symbol_map, const std::string& symbol, const T& value);

  /*!
   * \brief returns low-level device pointer
   * \note assumes low-level device has been initialized
   */
  const std::shared_ptr<LowLevelDevice>& low_level_device() const {
    CHECK(low_level_device_ != nullptr) << "attempt to get uninitialized low-level device";
    return low_level_device_;
  }

  const double GetLastBatchTime() {
    double result = last_batch_time_;
    last_batch_time_ = 0.0;
    return result;
  }

  const double GetLastBatchCycles() {
    double result = last_batch_cycles_;
    last_batch_cycles_ = 0.0;
    return result;
  }

 private:
  /*! \brief low-level device pointer */
  std::shared_ptr<LowLevelDevice> low_level_device_;
  /*! \brief prefix for binary names in target compiler toolchain */
  std::string toolchain_prefix_;
  /*! \brief array of memory allocators for each on-device section */
  std::shared_ptr<MicroSectionAllocator>
      section_allocators_[static_cast<size_t>(SectionKind::kNumKinds)];
  /*! \brief number of bytes in a word on the target device */
  TargetWordSize word_size_;
  /*! \brief whether the target device requires a thumb-mode bit on function addresses
   *
   * ARM and other manufacturers use the lowest bit of a function address to determine
   * whether it's a "thumb mode" function.  The Thumb ISA is more restricted, but
   * results in more compact binaries.
   */
  bool thumb_mode_;
  /*! \brief TODO */
  bool use_device_timer_;
  /*! \brief symbol map for the device runtime */
  SymbolMap runtime_symbol_map_;
  /*! \brief TODO */
  std::vector<DevTask> task_queue_;
  // TODO(weberlo): we don't even need an allocator mechanism for the args
  // section. there's only ever one allocation.
  /*! \brief TODO hack */
  TargetDataLayoutEncoder batch_args_encoder_;
  /*! \brief TODO hack */
  double last_batch_time_;
  /*! \brief TODO hack */
  double last_batch_cycles_;

  /*!
   * \brief patches a function pointer in this module to an implementation
   * \param func_name name of the function pointer being patched
   */
  void PatchImplHole(const SymbolMap& symbol_map, const std::string& func_name);

  /*!
   * \brief appends arguments to the host-side buffer of `encoder`
   * \param encoder encoder being used to append `args`
   * \param args args to be appended
   * \return device address of the allocated args
   */
  std::tuple<TargetPtr, TargetPtr> EncoderAppend(TargetDataLayoutEncoder* encoder,
                                                 const TVMArgs& args);

  /*!
   * \brief appends a `DLTensor` to the host-side buffer of `encoder`
   * \param encoder encoder being used to append `arr`
   * \param arr DLTensor to be appended
   * \return device address of the allocated `DLTensor`
   */
  template <typename T>
  TargetPtr EncoderAppend(TargetDataLayoutEncoder* encoder, const DLTensor& arr);

  /*!
   * \brief checks and logs if there was an error during the device's most recent execution
   */
  void CheckDeviceError();

  /*!
   * \brief returns section allocator corresponding to the given section kind
   * \param kind kind of target section
   * \return shared pointer to section allocator
   */
  std::shared_ptr<MicroSectionAllocator> GetAllocator(SectionKind kind) {
    return section_allocators_[static_cast<size_t>(kind)];
  }

  /*!
    * \brief Push a new session context onto the thread-local stack.
    *  The session on top of the stack is used as the current global session.
    */
  static void EnterWithScope(ObjectPtr<MicroSession> session);

  /*!
    * \brief Pop a session off the thread-local context stack,
    *  restoring the previous session as the current context.
    */
  static void ExitWithScope();
};

/*!
 * \brief a device memory region associated with the session that allocated it
 *
 * We use this to store a reference to the session in each allocated object and
 * only deallocate the session once there are no more references to it.
 */
struct MicroDevSpace {
  /*! \brief data being wrapped */
  TargetPtr data;
  /*! \brief shared ptr to session where this data is valid */
  ObjectPtr<MicroSession> session;
};

// TODO(weberlo): maybe templatize serialization to reduce redundancy

/*! \brief TVM array for serialization to 32-bit devices */
struct TVMArray32 {
  TVMArray32(
      TargetVal data,
      DLContext ctx,
      int32_t ndim,
      DLDataType dtype,
      TargetVal shape,
      TargetVal strides,
      TargetVal byte_offset)
    : data(data.uint32()),
      ctx(ctx),
      ndim(ndim),
      pad0(0),
      dtype(dtype),
      shape(shape.uint32()),
      strides(strides.uint32()),
      pad1(0),
      byte_offset(byte_offset.uint32()),
      pad2(0) { }

  /*!
   * \brief The opaque data pointer points to the allocated data.
   *  This will be CUDA device pointer or cl_mem handle in OpenCL.
   *  This pointer is always aligns to 256 bytes as in CUDA.
   */
  uint32_t data;
  /*! \brief The device context of the tensor */
  DLContext ctx;
  /*! \brief Number of dimensions */
  int32_t ndim;
  /*! \brief Padding to enforce struct alignment */
  uint32_t pad0;
  /*! \brief The data type of the pointer */
  DLDataType dtype;
  /*! \brief The shape of the tensor */
  uint32_t shape;
  /*!
   * \brief strides of the tensor,
   *  can be NULL, indicating tensor is compact.
   */
  uint32_t strides;
  /*! \brief Padding to enforce struct alignment */
  uint32_t pad1;
  /*! \brief The offset in bytes to the beginning pointer to data */
  uint32_t byte_offset;
  /*! \brief Padding to enforce struct alignment */
  uint32_t pad2;
};

/*! \brief TVM array for serialization to 64-bit devices */
struct TVMArray64 {
  TVMArray64(
      TargetVal data,
      DLContext ctx,
      int32_t ndim,
      DLDataType dtype,
      TargetVal shape,
      TargetVal strides,
      TargetVal byte_offset)
    : data(data.uint64()),
      ctx(ctx),
      ndim(ndim),
      pad0(0),
      dtype(dtype),
      shape(shape.uint64()),
      strides(strides.uint64()),
      byte_offset(byte_offset.uint64()) { }
  /*!
   * \brief The opaque data pointer points to the allocated data.
   *  This will be CUDA device pointer or cl_mem handle in OpenCL.
   *  This pointer is always aligns to 256 bytes as in CUDA.
   */
  uint64_t data;
  /*! \brief The device context of the tensor */
  DLContext ctx;
  /*! \brief Number of dimensions */
  int32_t ndim;
  /*! \brief Padding to enforce struct alignment */
  uint32_t pad0;
  /*! \brief The data type of the pointer */
  DLDataType dtype;
  /*! \brief The shape of the tensor */
  uint64_t shape;
  /*!
   * \brief strides of the tensor,
   *  can be NULL, indicating tensor is compact.
   */
  uint64_t strides;
  /*! \brief The offset in bytes to the beginning pointer to data */
  uint64_t byte_offset;
};

/*! \brief MicroTVM task to store in task queue before specializing to word size */
struct DevTask {
  /*! \brief Pointer to function to call for this task */
  TargetVal func;
  /*! \brief Array of argument values */
  TargetVal arg_values;
  /*! \brief Array of type codes for each argument value */
  TargetVal arg_type_codes;
  /*! \brief Number of arguments */
  int32_t num_args;
};

/*! \brief MicroTVM task for serialization to 32-bit devices */
typedef struct StructUTVMTask32 {
  StructUTVMTask32(DevTask task)
    : func(task.func.uint32()),
      arg_values(task.arg_values.uint32()),
      arg_type_codes(task.arg_type_codes.uint32()),
      num_args(task.num_args) { }

  /*! \brief Pointer to function to call for this task */
  uint32_t func;
  /*! \brief Array of argument values */
  uint32_t arg_values;
  /*! \brief Array of type codes for each argument value */
  uint32_t arg_type_codes;
  /*! \brief Number of arguments */
  int32_t num_args;
} StructUTVMTask32;

/*! \brief MicroTVM task for serialization to 64-bit devices */
typedef struct StructUTVMTask64 {
  StructUTVMTask64(DevTask task)
    : func(task.func.uint64()),
      arg_values(task.arg_values.uint64()),
      arg_type_codes(task.arg_type_codes.uint64()),
      num_args(task.num_args) { }

  /*! \brief Pointer to function to call for this task */
  uint64_t func;
  /*! \brief Array of argument values */
  uint64_t arg_values;
  /*! \brief Array of type codes for each argument value */
  uint64_t arg_type_codes;
  /*! \brief Number of arguments */
  int32_t num_args;
} StructUTVMTask64;

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_MICRO_SESSION_H_
