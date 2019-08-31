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
 * \file openocd_low_level_device.cc
 */
#include <sstream>
#include <iomanip>

#include "micro_common.h"
#include "low_level_device.h"
#include "tcl_socket.h"

namespace tvm {
namespace runtime {

/*!
 * \brief OpenOCD low-level device for uTVM micro devices connected over JTAG
 */
class OpenOCDLowLevelDevice final : public LowLevelDevice {
 public:
  /*!
   * \brief constructor to initialize connection to openocd device
   * \param base_addr base address of the device
   * \param server_addr address of the OpenOCD server to connect to
   * \param port port of the OpenOCD server to connect to
   */
  explicit OpenOCDLowLevelDevice(std::uintptr_t base_addr,
                                 const std::string& server_addr,
                                 int port) : socket_() {
      socket_.Connect(tvm::common::SockAddr(server_addr.c_str(), port));
      socket_.cmd_builder() << "reset halt";
      socket_.SendCommand();
      base_addr_ = base_addr;
      CHECK(base_addr_ % 8 == 0) << "base address not aligned to 8 bytes";
  }

  void Read(DevBaseOffset offset, void* buf, size_t num_bytes) {
    if (num_bytes == 0) {
      return;
    }

    // TODO(weberlo): Refactor between read and write.
    // Check if we need to chunk this write request.
    if (num_bytes > kMemTransferLimit) {
      DevBaseOffset curr_offset = offset;
      char* curr_buf_ptr = reinterpret_cast<char*>(buf);
      while (num_bytes != 0) {
        size_t amount_to_read;
        if (num_bytes > kMemTransferLimit) {
          amount_to_read = kMemTransferLimit;
        } else {
          amount_to_read = num_bytes;
        }
        Read(offset, reinterpret_cast<void*>(curr_buf_ptr), amount_to_read);
        offset += amount_to_read;
        curr_buf_ptr += amount_to_read;
        num_bytes -= amount_to_read;
      }
      return;
    }
    {
      socket_.cmd_builder() << "array unset output";
      socket_.SendCommand();

      DevPtr addr = DevPtr(base_addr_ + offset.value());
      socket_.cmd_builder()
        << "mem2array output"
        << " " << std::dec << kWordSize
        << " " << addr.cast_to<void*>()
        // Round up any request sizes under a byte, since OpenOCD doesn't support
        // sub-byte-sized transfers.
        << " " << std::dec << (num_bytes < 8 ? 8 : num_bytes);
      socket_.SendCommand();
    }

    {
      socket_.cmd_builder() << "ocd_echo $output";
      socket_.SendCommand();
      const std::string& reply = socket_.last_reply();

      std::istringstream values(reply);
      char* char_buf = reinterpret_cast<char*>(buf);
      ssize_t req_bytes_remaining = num_bytes;
      uint32_t index;
      uint32_t val;
      while (req_bytes_remaining > 0) {
        // The response from this command pairs indices with the contents of the
        // memory at that index.
        values >> index;
        CHECK(index < num_bytes)
          << "index " << index <<
          " out of bounds (length " << num_bytes << ")";
        // Read the value into `curr_val`, instead of reading directly into
        // `buf_iter`, because otherwise it's interpreted as the ASCII value and
        // not the integral value.
        values >> val;
        char_buf[index] = static_cast<uint8_t>(val);
        req_bytes_remaining--;
      }
      if (num_bytes >= 8) {
        uint32_t check_index;
        values >> check_index;
        CHECK(check_index != index) << "more data in response than requested";
      }
    }
  }

  void Write(DevBaseOffset offset, const void* buf, size_t num_bytes) {
    if (num_bytes == 0) {
      return;
    }

    // Check if we need to chunk this write request.
    if (num_bytes > kMemTransferLimit) {
      DevBaseOffset curr_offset = offset;
      const char* curr_buf_ptr = reinterpret_cast<const char*>(buf);
      while (num_bytes != 0) {
        size_t amount_to_write;
        if (num_bytes > kMemTransferLimit) {
          amount_to_write = kMemTransferLimit;
        } else {
          amount_to_write = num_bytes;
        }
        Write(offset, reinterpret_cast<const void*>(curr_buf_ptr), amount_to_write);
        offset += amount_to_write;
        curr_buf_ptr += amount_to_write;
        num_bytes -= amount_to_write;
      }
      return;
    }

    // Clear `input` array.
    socket_.cmd_builder() << "array unset input";
    socket_.SendCommand();
    // Build a command to set the value of `input`.
    {
      std::ostringstream& cmd_builder = socket_.cmd_builder();
      cmd_builder << "array set input {";
      const char* char_buf = reinterpret_cast<const char*>(buf);
      for (size_t i = 0; i < num_bytes; i++) {
        // In a Tcl `array set` commmand, we need to pair the array indices with
        // their values.
        cmd_builder << i << " ";
        // Need to cast to uint, so the number representation of `buf[i]` is
        // printed, and not the ASCII representation.
        cmd_builder << static_cast<uint32_t>(char_buf[i]) << " ";
      }
      cmd_builder << "}";
      socket_.SendCommand();
    }
    {
      DevPtr addr = DevPtr(base_addr_ + offset.value());
      socket_.cmd_builder()
        << "array2mem input"
        << " " << std::dec << kWordSize
        << " " << addr.cast_to<void*>()
        << " " << std::dec << num_bytes;
      socket_.SendCommand();
    }
  }

  void Execute(DevBaseOffset func_offset, DevBaseOffset breakpoint) {
    socket_.cmd_builder() << "halt 0";
    socket_.SendCommand();

    // Set up the stack pointer.
    DevPtr stack_end = stack_top() - 8;
    socket_.cmd_builder() << "reg sp " << stack_end.cast_to<void*>();
    socket_.SendCommand();

    // Set a breakpoint at the beginning of `UTVMDone`.
    socket_.cmd_builder() << "bp " << ToDevPtr(breakpoint).cast_to<void*>() << " 2";
    socket_.SendCommand();

    DevPtr func_addr = DevPtr(base_addr_ + func_offset.value());
    socket_.cmd_builder() << "resume " << func_addr.cast_to<void*>();
    socket_.SendCommand();

    socket_.cmd_builder() << "wait_halt " << kWaitTime;
    socket_.SendCommand();

    socket_.cmd_builder() << "halt 0";
    socket_.SendCommand();

    // Remove the breakpoint.
    socket_.cmd_builder() << "rbp " << ToDevPtr(breakpoint).cast_to<void*>();
    socket_.SendCommand();
  }

  void SetStackTop(DevBaseOffset stack_top) {
    stack_top_ = DevPtr(base_addr_ + stack_top.value());
  }

  std::uintptr_t base_addr() const final {
    return base_addr_;
  }

  DevPtr stack_top() const {
    CHECK(stack_top_ != nullptr) << "stack top was never initialized";
    return stack_top_;
  }

  const char* device_type() const final {
    return "openocd";
  }

 private:
  /*! \brief base address of the micro device memory region */
  std::uintptr_t base_addr_;
  /*! \brief top of the stack section */
  DevPtr stack_top_;
  /*! \brief socket used to communicate with the device through Tcl */
  TclSocket socket_;

  /*! \brief number of bytes in a word on the target device (64-bit) */
  static const constexpr ssize_t kWordSize = 8;
  // NOTE: OpenOCD will call any request larger than this constant an "absurd
  // request".
  /*! \brief maximum number of bytes allowed in a single memory transfer */
  static const constexpr ssize_t kMemTransferLimit = 64000;
  /*! \brief number of milliseconds to wait for function execution to halt */
  static const constexpr int kWaitTime = 10000;
};

const std::shared_ptr<LowLevelDevice> OpenOCDLowLevelDeviceCreate(std::uintptr_t base_addr,
                                                                  const std::string& server_addr,
                                                                  int port) {
  std::shared_ptr<LowLevelDevice> lld =
      std::make_shared<OpenOCDLowLevelDevice>(base_addr, server_addr, port);
  return lld;
}

}  // namespace runtime
}  // namespace tvm
