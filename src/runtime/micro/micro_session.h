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
 *  Copyright (c) 2019 by Contributors
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
#include "device/utvm_runtime.h"
#include "target_data_layout_encoder.h"

namespace tvm {
namespace runtime {

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
                                 const std::shared_ptr<ModuleNode>& sptr_to_self);

  /*!
   * \return The type key of the executor.
   */
  const char* type_key() const final {
    return "MicroSession";
  }

  /*!
   * \brief constructor
   */
  MicroSession();

  /*!
   * \brief destructor
   */
  ~MicroSession();

  static std::shared_ptr<MicroSession>& Current();

  /*!
   * \brief creates session by setting up a low-level device and initting allocators for it
   * \param args TVMArgs passed into the micro.init packedfunc
   */
  void CreateSession(const std::string& device_type,
                     const std::string& binary_path,
                     const std::string& toolchain_prefix,
                     std::uintptr_t base_addr,
                     const std::string& server_addr,
                     int port);

  /*!
   * \brief ends the session by destructing the low-level device and its allocators
   */
  void EndSession();

  /*!
   * \brief allocate memory in section
   * \param type type of section to allocate in
   * \param size size of allocated memory in bytes
   * \return pointer to allocated memory region in section, nullptr if out of space
   */
  DevBaseOffset AllocateInSection(SectionKind type, size_t size);

  /*!
   * \brief free prior allocation from section
   * \param type type of section to allocate in
   * \param ptr pointer to allocated memory
   */
  void FreeInSection(SectionKind type, DevBaseOffset ptr);

  /*!
   * \brief read string from device to host
   * \param str_offset device offset of first character of string
   * \return host copy of device string that was read
   */
  std::string ReadString(DevBaseOffset str_offset);

  /*!
   * \brief sets up runtime metadata for `func` and copies arguments for on-device execution
   * \param func address of the function to be executed
   * \param args args to the packed function
   */
  void PushToExecQueue(DevBaseOffset func, const TVMArgs& args);

  /*!
   * \brief loads binary onto device
   * \param binary_path path to binary object file
   * \param patch_dylib_pointers whether runtime API function pointer patching is needed
   * \return info about loaded binary
   */
  BinaryInfo LoadBinary(const std::string& binary_path, bool patch_dylib_pointers = true);

  /*!
  * \brief read value of symbol from device memory
  * \param symbol_map symbol map to read location of symbol from
  * \param symbol name of symbol being read from
  * \return value at symbol in memory
  */
  template <typename T>
  T DevSymbolRead(const SymbolMap& symbol_map, const std::string& symbol);

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

 private:
  /*! \brief low-level device pointer */
  std::shared_ptr<LowLevelDevice> low_level_device_;
  /*! \brief prefix for binary names in target compiler toolchain */
  std::string toolchain_prefix_;
  /*! \brief array of memory allocators for each on-device section */
  std::shared_ptr<MicroSectionAllocator>
      section_allocators_[static_cast<size_t>(SectionKind::kNumKinds)];
  /*! \brief total number of bytes of usable device memory for this session */
  size_t memory_size_;
  /*! \brief uTVM runtime binary info */
  BinaryInfo runtime_bin_info_;
  /*! \brief path to uTVM runtime source code */
  std::string runtime_binary_path_;
  /*! \brief offset of the runtime entry function */
  DevBaseOffset utvm_main_symbol_;
  /*! \brief offset of the runtime exit breakpoint */
  DevBaseOffset utvm_done_symbol_;

  /*!
   * \brief patches a function pointer in this module to an implementation
   * \param func_name name of the function pointer being patched
   */
  void PatchImplHole(const SymbolMap& symbol_map, const std::string& func_name);

  /*!
   * \brief sets the runtime binary path
   * \param path to runtime binary
   */
  void SetRuntimeBinaryPath(std::string path);

  /*!
   * \brief appends arguments to the host-side buffer of `encoder`
   * \param encoder encoder being used to append `args`
   * \param args args to be appended
   * \return device address of the allocated args
   */
  std::tuple<DevPtr, DevPtr> EncoderAppend(TargetDataLayoutEncoder* encoder, const TVMArgs& args);

  /*!
   * \brief appends a `TVMArray` to the host-side buffer of `encoder`
   * \param encoder encoder being used to append `arr`
   * \param arr TVMArray to be appended
   * \return device address of the allocated `TVMArray`
   */
  DevPtr EncoderAppend(TargetDataLayoutEncoder* encoder, const TVMArray& arr);

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
   * \brief returns the symbol map for the uTVM runtime
   * \return reference to symbol map
   */
  const SymbolMap& runtime_symbol_map() {
    return runtime_bin_info_.symbol_map;
  }

  /*!
    * \brief Push a new session context onto the thread-local stack.
    *  The session on top of the stack is used as the current global session.
    */
  static void EnterWithScope(std::shared_ptr<MicroSession> session);
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
  void* data;
  /*! \brief shared ptr to session where this data is valid */
  std::shared_ptr<MicroSession> session;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_MICRO_SESSION_H_
