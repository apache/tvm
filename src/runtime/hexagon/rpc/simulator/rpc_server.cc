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

#include <HAP_farf.h>
#include <dlfcn.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#include "../../../library_module.h"
#include "../../../minrpc/minrpc_server.h"
#include "../../hexagon_common.h"
#include "../../profiler/prof_utils.h"
#include "hexagon_sim_proto.h"
#include "tvm/runtime/packed_func.h"
#include "tvm/runtime/registry.h"

namespace tvm {
namespace runtime {
namespace hexagon {

class stringbuf_with_remote_access : public std::stringbuf {
  constexpr static size_t zero_count = 1024;
  constexpr static char zeros[zero_count] = {};

 public:
  char* reserve_for_remote_write(size_t size) {
    // Reserve memory in the put area by adding zeros to it. The put
    // area will automatically resize itself as needed. This is needed
    // for the simulator to be able to write to program's memory.
    size_t remaining = size;
    while (remaining >= zero_count) {
      sputn(zeros, zero_count);
      remaining -= zero_count;
    }
    sputn(zeros, remaining);
    return pptr() - size;
  }

  // Get area is the storage area with the data that will be read from
  // the buffer. From the buffer's point of view, this is the area with
  // the outgoing data.
  char* get_area() { return gptr(); }

  // Adjust the buffer state after the data has been read by the remote
  // end. This means discarding given amount of bytes from the get area.
  size_t acknowledge_remote_read(size_t size) {
    size_t remaining = size;
    do {
      auto bump = std::min<size_t>(in_avail(), remaining);
      setg(eback(), gptr() + bump, egptr());
      remaining -= bump;
      // The setg will cause in_avail to become 0. At this point calling
      // bumpc will either recaulate the get area pointers (if there are
      // more characters left), or it will return eof().
      if (remaining == 0 || sbumpc() == traits_type::eof()) {
        break;
      }
      remaining--;
    } while (remaining > 0);

    // This will return 0 on success, non-zero if underflow occurred.
    return size - remaining;
  }
};

const char stringbuf_with_remote_access::zeros[zero_count];

class SimulatorIOHandler {
 public:
  SimulatorIOHandler() = default;

  void MessageStart(size_t message_size_bytes) {}
  void MessageDone() {}

  // Store data from the input buffer into 'buf'.
  ssize_t PosixRead(uint8_t* buf, size_t read_len_bytes) {
    auto char_buf = reinterpret_cast<decltype(inp_buffer_)::char_type*>(buf);
    return static_cast<ssize_t>(inp_buffer_.sgetn(char_buf, read_len_bytes));
  }

  // Append 'write_len_bytes' starting at 'buf' to the output buffer.
  ssize_t PosixWrite(const uint8_t* buf, size_t write_len_bytes) {
    auto char_buf = reinterpret_cast<const decltype(out_buffer_)::char_type*>(buf);
    return static_cast<ssize_t>(out_buffer_.sputn(char_buf, write_len_bytes));
  }

  void Close() {}
  void Exit(int code) { exit(code); }

  char* PrepareToReceiveFromRemote(size_t nbytes) {
    // Reserve space in the incoming buffer.
    return inp_buffer_.reserve_for_remote_write(nbytes);
  }
  bool CompleteReceiveFromRemote(size_t nbytes) {
    // Nothing to do.
    return true;
  }
  char* PrepareToSendToRemote(size_t nbytes) {
    // Return the pointer to the data coming out of the buffer.
    return out_buffer_.get_area();
  }
  bool CompleteSendToRemote(size_t nbytes) {
    // Flush the read data from the outgoing buffer.
    return out_buffer_.acknowledge_remote_read(nbytes) == 0;
  }

 private:
  stringbuf_with_remote_access inp_buffer_;  // Data received from remote.
  stringbuf_with_remote_access out_buffer_;  // Data to be sent to remote.
};

// Internal allocator that redirects alloc to TVM's C API.
template <typename TIOHandler>
class SimulatorPageAllocator {
 public:
  using ArenaPageHeader = tvm::support::ArenaPageHeader;

  explicit SimulatorPageAllocator(TIOHandler* io) : io_(io) {}

  ArenaPageHeader* allocate(size_t min_size) {
    size_t npages = ((min_size + kPageSize - 1) / kPageSize);
    void* data;

    if (posix_memalign(&data, kPageAlign, npages * kPageSize) != 0) {
      io_->Exit(static_cast<int>(RPCServerStatus::kAllocError));
    }

    ArenaPageHeader* header = static_cast<ArenaPageHeader*>(data);
    header->size = npages * kPageSize;
    header->offset = sizeof(ArenaPageHeader);
    return header;
  }

  void deallocate(ArenaPageHeader* page) { free(page); }

  static const constexpr int kPageSize = 2 << 10;
  static const constexpr int kPageAlign = kPageSize;

 private:
  TIOHandler* io_;
};

class SimulatorRPCServer : public MinRPCServer<SimulatorIOHandler, SimulatorPageAllocator> {
  using Base = MinRPCServer<SimulatorIOHandler, SimulatorPageAllocator>;

 public:
  SimulatorRPCServer() : Base(&io_) {}

  char* PrepareToReceive(size_t nbytes) {
    // Reserve receiving area.
    return io_.PrepareToReceiveFromRemote(nbytes);
  }
  bool CompleteReceive(size_t nbytes) {
    // Interpret the received message.
    return io_.CompleteReceiveFromRemote(nbytes) && ProcessOnePacket();
  }
  char* PrepareToSend(size_t nbytes) {
    // Identify the beginning of the outgoing data.
    return io_.PrepareToSendToRemote(nbytes);
  }
  bool CompleteSend(size_t nbytes) {
    // Flush the read data.
    return io_.CompleteSendToRemote(nbytes);
  }

 private:
  SimulatorIOHandler io_;
};

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

// Handling communication with the simulator.
//
// Simulator can read and write the memory of the process, but the process has
// no way to find out that it's running in a simulation, nor can it actively
// communicate with the simulator. The RPC server must be entirely passive,
// allowing the simulator to perform data transfers.

extern "C" {
// The names of these symbols will "pollute" the global namespace in the final
// binary, so make them unique (i.e. "more likely to be unique"). These names
// will be referenced in the host's code (the code controlling the simulator)
// as well, so to avoid repetition use human-friendly macros.
int DISPATCH_FUNCTION_NAME(void*) __attribute__((noinline));
alignas(8) volatile Message MESSAGE_BUFFER_NAME;
}

inline uint32_t va(const volatile void* p) {
  static_assert(sizeof(p) == sizeof(uint32_t), "Pointers must be 32-bit long");
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
}

// NOLINTNEXTLINE(runtime/references)
__attribute__((__unused__)) static std::string to_string(const volatile Message& m) {
  std::stringstream out;
  out << "{code=";
  switch (m.code) {
    case Message::kNone:
      out << "kNone";
      break;
    case Message::kAck:
      out << "kAck";
      break;
    case Message::kTerminate:
      out << "kTerminate";
      break;
    case Message::kReceiveStart:
      out << "kReceiveStart";
      break;
    case Message::kReceiveEnd:
      out << "kReceiveEnd";
      break;
    case Message::kSendStart:
      out << "kSendStart";
      break;
    case Message::kSendEnd:
      out << "kSendEnd";
      break;
    default:
      out << "<unknown>(" << m.code << ")";
      break;
  }
  out << ", len:" << m.len << ", va:" << std::hex << m.va << std::dec << '}';
  return out.str();
}

static inline void setmsg(volatile Message* m, uint32_t code, uint32_t len, uint32_t va) {
  m->code = code;
  m->len = len;
  m->va = va;
}

int DISPATCH_FUNCTION_NAME(void* serverp) {
  static bool terminate = false;
  if (terminate) {
    return 1;
  }

  Message msg;
  setmsg(&msg, MESSAGE_BUFFER_NAME.code, MESSAGE_BUFFER_NAME.len, MESSAGE_BUFFER_NAME.va);

  auto& server = *reinterpret_cast<tvm::runtime::hexagon::SimulatorRPCServer*>(serverp);

  switch (msg.code) {
    case Message::kReceiveStart:
      assert(msg.va == Message::null_va);
      setmsg(&MESSAGE_BUFFER_NAME, Message::kAck, msg.len, va(server.PrepareToReceive(msg.len)));
      break;
    case Message::kReceiveEnd:
      server.CompleteReceive(msg.len);
      setmsg(&MESSAGE_BUFFER_NAME, Message::kAck, 0u, Message::null_va);
      break;
    case Message::kSendStart:
      assert(msg.va == Message::null_va);
      setmsg(&MESSAGE_BUFFER_NAME, Message::kAck, msg.len, va(server.PrepareToSend(msg.len)));
      break;
    case Message::kSendEnd:
      server.CompleteSend(msg.len);
      setmsg(&MESSAGE_BUFFER_NAME, Message::kAck, 0u, Message::null_va);
      break;
    case Message::kTerminate:
      // Don't exit immediately, send response to simulator first.
      terminate = true;
      setmsg(&MESSAGE_BUFFER_NAME, Message::kAck, 0u, Message::null_va);
      break;
  }

  return 0;
}

int main(int argc, char* argv[]) {
  // Load C++RT and ourselves as "global" to make all the symbols defined
  // there be visible to any subsequent libraries loaded via dlopen.
  void* cxx_abi = dlopen("libc++abi.so", RTLD_GLOBAL);
  ICHECK(cxx_abi != nullptr);
  void* cxx = dlopen("libc++.so", RTLD_GLOBAL);
  ICHECK(cxx != nullptr);
  void* self = dlopen(argv[0], RTLD_GLOBAL);
  ICHECK(self != nullptr);

  const auto* api = tvm::runtime::Registry::Get("device_api.hexagon");
  ICHECK(api != nullptr);
  tvm::runtime::Registry::Register("device_api.cpu", true).set_body(*api);

  tvm::runtime::hexagon::SimulatorRPCServer server;

  // Hand-encode user-instruction:
  // r17:16 = userinsn(r17:16, r17:16, #0)
  // 1100 1111 00010000 11010000 00010000 : cf10d010

  asm volatile("r17:16 = combine(%0,%1); .long 0xcf10d010"
               :  // No outputs
               : "r"(&MESSAGE_BUFFER_NAME), "r"(&DISPATCH_FUNCTION_NAME)
               : "r16", "r17");

  while (!DISPATCH_FUNCTION_NAME(&server)) {
    // nothing
  }

  dlclose(self);
  dlclose(cxx);
  dlclose(cxx_abi);
  return 0;
}

// Workaround for missing functions in 8.5.08
extern "C" {
__attribute__((weak)) void _Get_eh_data() {}
__attribute__((weak)) void _Parse_fde_instr() {}
}

TVM_REGISTER_GLOBAL("tvm.hexagon.load_module")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      std::string soname = args[0];
      tvm::ObjectPtr<tvm::runtime::Library> n = tvm::runtime::CreateDSOLibraryObject(soname);
      *rv = CreateModuleFromLibrary(n);
    });

TVM_REGISTER_GLOBAL("tvm.hexagon.get_profile_output")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      std::string profiling_mode = args[0];
      std::string out_file = args[1];
      if (profiling_mode.compare("lwp") == 0) {
        *rv = WriteLWPOutput(out_file);
      } else {
        HEXAGON_PRINT(ERROR, "ERROR: Unsupported profiling mode: %s", profiling_mode.c_str());
        *rv = false;
      }
    });

void SaveBinaryToFile(const std::string& file_name, const std::string& data) {
  std::ofstream fs(file_name, std::ios::out | std::ios::binary);
  ICHECK(!fs.fail()) << "Cannot open " << file_name;
  fs.write(&data[0], data.length());
}

TVM_REGISTER_GLOBAL("tvm.rpc.server.upload")
    .set_body([](tvm::runtime::TVMArgs args, tvm::runtime::TVMRetValue* rv) {
      std::string file_name = args[0];
      std::string data = args[1];
      SaveBinaryToFile(file_name, data);
    });
