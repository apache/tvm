/*!
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
 * This Contribution is being provided by Qualcomm Technologies, Inc.,
 * a Delaware corporation, or its subsidiary Qualcomm Innovation Center, Inc.,
 * a California corporation, under certain additional terms and conditions
 * pursuant to Section 5 of the Apache 2.0 license.  In this regard, with
 * respect to this Contribution, the term "Work" in Section 1 of the
 * Apache 2.0 license means only the specific subdirectory within the TVM repo
 * (currently at https://github.com/dmlc/tvm) to which this Contribution is
 * made.
 * In any case, this submission is "Not a Contribution" with respect to its
 * permitted use with any of the "vta" and "verilog" subdirectories in the TVM
 * repo.
 * Qualcomm Technologies, Inc. and Qualcomm Innovation Center, Inc. retain
 * copyright of their respective Contributions.
 */
#include <HexagonWrapper.h>
#include <dmlc/logging.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Process.h>

#include <algorithm>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../hexagon_module.h"
#include "hexagon_sim_proto.h"

namespace tvm {
namespace runtime {
namespace hexagon {

static_assert(sizeof(HEX_VA_t) == sizeof(uint32_t),
              "Hexagon VA must be uint32");

template <typename T>
struct unalign {
  using type = T __attribute__((aligned(1)));
};

template <unsigned N>
struct uint {
  using type = void;
};

template <>
struct uint<8> {
  using type = uint64_t;
};
template <>
struct uint<4> {
  using type = uint32_t;
};
template <>
struct uint<2> {
  using type = uint16_t;
};
template <>
struct uint<1> {
  using type = uint8_t;
};

class HexagonSimulator : public tvm::runtime::hexagon::Device {
 public:
  explicit HexagonSimulator(bool enable_queuing);
  ~HexagonSimulator() final {}
  void* Alloc(unsigned size, unsigned align) final;
  void Free(void* ptr) final;
  void CopyDeviceToDevice(void* dst, const void* src, unsigned len) final;
  void CopyDeviceToHost(void* host_dst, const void* src, unsigned len) final;
  void CopyHostToDevice(void* dst, const void* host_src, unsigned len) final;
  void* Load(const std::string& data, const std::string& fmt) final;
  void Unload(void* mod) final;
  void* Resolve(const std::string& sym) final;
  void Call(void* func, uint32_t* scalar, unsigned sc_num, uint32_t* stack,
            unsigned st_num) final;

 private:
  static HEX_VA_t p2va(const void* p);
  static void* va2p(HEX_VA_t va);

  void CopyFromV(void* host_dst, HEX_VA_t src, unsigned len);
  void CopyToV(HEX_VA_t dst, const void* host_src, unsigned len);

  template <unsigned N>
  void CopyNToV(HEX_VA_t dst, const void* host_src);
  template <unsigned N>
  void CopyNFromV(void* host_dst, HEX_VA_t src);

  // NOLINTNEXTLINE(runtime/references)
  void SendMsg(Message& m, const void* data, bool show_dbg);

  std::unique_ptr<HexagonWrapper> sim_;
  HEX_VA_t dispatch_v_, message_buffer_v_;
  bool task_queuing_;
};

inline HEX_VA_t HexagonSimulator::p2va(const void* p) {
  uintptr_t u = reinterpret_cast<uintptr_t>(p);
  HEX_VA_t va = static_cast<HEX_VA_t>(u);
  CHECK_EQ(static_cast<uintptr_t>(va), u);
  return va;
}

inline void* HexagonSimulator::va2p(HEX_VA_t va) {
  return reinterpret_cast<void*>(static_cast<uintptr_t>(va));
}

template <unsigned N, unsigned A>
constexpr bool is_multiple_of() {
  return (N / A) * A == N;
}

std::unique_ptr<Device> CreateHexagonSimulator() {
  // C++11 does not have std::make_unique.
  return llvm::make_unique<HexagonSimulator>(/*enable_queuing=*/true);
}

template <unsigned N>
void HexagonSimulator::CopyNToV(HEX_VA_t dst, const void* host_src) {
  using src_uint_t = typename unalign<typename uint<N>::type>::type;
  auto* ps = reinterpret_cast<const src_uint_t*>(host_src);
  CHECK_EQ(sim_->WriteVirtual(dst, -1u, N, *ps), HEX_STAT_SUCCESS);
}

template <unsigned N>
void HexagonSimulator::CopyNFromV(void* host_dst, HEX_VA_t src) {
  typename uint<N>::type v;
  CHECK_EQ(sim_->ReadVirtual(src, -1u, N, &v), HEX_STAT_SUCCESS);

  using dst_uint_t = typename unalign<typename uint<N>::type>::type;
  auto* pd = reinterpret_cast<dst_uint_t*>(host_dst);
  *pd = v;
}

void HexagonSimulator::CopyToV(HEX_VA_t dst, const void* host_src,
                               unsigned len) {
  const uint8_t* src = static_cast<const uint8_t*>(host_src);

  while (len >= 8) {
    CopyNToV<8>(dst, src);
    dst += 8;
    src += 8;
    len -= 8;
  }
  if (len >= 4) {
    CopyNToV<4>(dst, src);
    dst += 4;
    src += 4;
    len -= 4;
  }
  if (len >= 2) {
    CopyNToV<2>(dst, src);
    dst += 2;
    src += 2;
    len -= 2;
  }
  if (len >= 1) {
    CopyNToV<1>(dst, src);
    dst++;
    src++;
    len--;
  }
  CHECK_EQ(len, 0);
}

void HexagonSimulator::CopyFromV(void* host_dst, HEX_VA_t src, unsigned len) {
  uint8_t* dst = static_cast<uint8_t*>(host_dst);

  while (len >= 8) {
    CopyNFromV<8>(dst, src);
    dst += 8;
    src += 8;
    len -= 8;
  }
  if (len >= 4) {
    CopyNFromV<4>(dst, src);
    dst += 4;
    src += 4;
    len -= 4;
  }
  if (len >= 2) {
    CopyNFromV<2>(dst, src);
    dst += 2;
    src += 2;
    len -= 2;
  }
  if (len >= 1) {
    CopyNFromV<1>(dst, src);
    dst++;
    src++;
    len--;
  }
  CHECK_EQ(len, 0);
}

void HexagonSimulator::SendMsg(Message& m, const void* data, bool show_dbg) {
  auto run = [this](bool report_cycles) {
    HEXAPI_CoreState core = HEX_CORE_RESET;
    HEX_4u_t result;
    HEX_8u_t cycles0, cycles1;
    if (report_cycles)
      CHECK_EQ(sim_->GetSimulatedCycleCount(&cycles0), HEX_STAT_SUCCESS);
    core = sim_->Run(&result);
    CHECK_EQ(core, HEX_CORE_BREAKPOINT);
    if (report_cycles) {
      CHECK_EQ(sim_->GetSimulatedCycleCount(&cycles1), HEX_STAT_SUCCESS);
      LOG(INFO) << "host: execution took " << (cycles1 - cycles0) << " cycles";
    }
  };

  // Send the message request.
  Message r = {kMsgReq, m.len, 0u};
  CopyToV(message_buffer_v_, &r, sizeof(r));
  run(false);

  // Receive the acknowledgement with the address for the payload.
  CopyFromV(&r, message_buffer_v_, sizeof(r));
  CHECK_EQ(r.code, kMsgAck);
  CHECK_GE(r.len, m.len);

  // Send the actual message.
  m.va = r.va;
  CopyToV(message_buffer_v_, &m, sizeof(m));
  if (m.len > 0) CopyToV(r.va, data, m.len);
  run(show_dbg);

  // Receive the return data.
  CopyFromV(&m, message_buffer_v_, sizeof(m));
  CHECK_EQ(m.code, kNone);
}

HexagonSimulator::HexagonSimulator(bool enable_queuing)
    : sim_(new HexagonWrapper(HEX_CPU_V66)), task_queuing_(enable_queuing) {
  HEXAPI_Status status = HEX_STAT_SUCCESS;

  // Locate the sim_dev binary in PATH, or in the current working directory.
  llvm::StringRef sim_dev = "sim_dev";
  llvm::Optional<std::string> path_sim_dev =
      llvm::sys::Process::FindInEnvPath("PATH", sim_dev);
  if (!path_sim_dev) {
    if (!llvm::sys::fs::exists(sim_dev)) {
      LOG(ERROR) << "Cannot find sim_dev in PATH.";
      exit(1);
    }
    path_sim_dev = sim_dev;
  }

  status = sim_->ConfigureExecutableBinary(path_sim_dev->c_str());
  if (status != HEX_STAT_SUCCESS)
    LOG(FATAL) << "HexagonSimulator: ConfigureExecutableBinary failed "
                  "with code="
               << static_cast<int>(status);

  status = sim_->EndOfConfiguration();
  if (status != HEX_STAT_SUCCESS)
    LOG(FATAL) << "HexagonSimulator: EndOfConfiguration failed with "
                  "code="
               << static_cast<int>(status);

  status = sim_->LoadExecutableBinary();
  if (status != HEX_STAT_SUCCESS)
    LOG(FATAL) << "HexagonSimulator: LoadExecutableBinary failed with "
                  "code="
               << static_cast<int>(status);

  status = sim_->ReadSymbolValue("dispatch", &dispatch_v_);
  if (status != HEX_STAT_SUCCESS)
    LOG(FATAL) << "HexagonSimulator: ReadSymbolValue(\"dispatch\") "
                  "failed with code="
               << static_cast<int>(status);

  status = sim_->ReadSymbolValue("message_buffer", &message_buffer_v_);
  if (status != HEX_STAT_SUCCESS)
    LOG(FATAL) << "HexagonSimulator: ReadSymbolValue(\"message_buffer\") "
                  "failed with code="
               << static_cast<int>(status);

  status = sim_->SetBreakpoint(dispatch_v_);
  if (status != HEX_STAT_SUCCESS)
    LOG(FATAL) << "HexagonSimulator: SetBreakpoint failed with "
                  "code="
               << static_cast<int>(status);

  HEXAPI_CoreState core = HEX_CORE_RESET;

  HEX_4u_t result;
  core = sim_->Run(&result);
  if (core != HEX_CORE_BREAKPOINT)
    LOG(FATAL) << "HexagonSimulator: Run not stopped on breakpoint, "
                  "code="
               << static_cast<int>(core);
}

void* HexagonSimulator::Alloc(unsigned size, unsigned align) {
  LOG(INFO) << "HexagonSimulator::Alloc(size=" << size << ", align=" << align
            << ')';
  Message m = {kAlloc, sizeof(MsgAlloc), 0u};
  MsgAlloc ma = {size, align};
  SendMsg(m, &ma, true);

  CHECK_EQ(sizeof(MsgPointer), m.len);
  MsgPointer mp;
  CopyFromV(&mp, m.va, m.len);

  LOG(INFO) << "HexagonSimulator::Alloc -> " << std::hex << mp.va << std::dec;
  CHECK_NE(mp.va, 0);
  return va2p(mp.va);
}

void HexagonSimulator::Free(void* ptr) {
  LOG(INFO) << "HexagonSimulator::Free(ptr=" << std::hex << ptr << std::dec
            << ')';
  if (task_queuing_) {
    Message mf = {kFlush, 0, 0};
    SendMsg(mf, 0, true);
  }
  Message m = {kFree, sizeof(MsgPointer), 0u};
  MsgPointer mp = {p2va(ptr)};
  SendMsg(m, &mp, true);
}

void HexagonSimulator::CopyDeviceToDevice(void* dst, const void* src,
                                          unsigned len) {
  LOG(INFO) << "HexagonSimulator::CopyDeviceToDevice(dst=" << std::hex << dst
            << ", src=" << src << ", len=" << std::dec << len << ')';
  CHECK(dst != nullptr && src != nullptr);
  Message m = {kCopy, sizeof(MsgCopy), 0u};
  MsgCopy mc = {p2va(dst), p2va(src), len};
  SendMsg(m, &mc, true);
}

void HexagonSimulator::CopyDeviceToHost(void* host_dst, const void* src,
                                        unsigned len) {
  LOG(INFO) << "HexagonSimulator::CopyDeviceToHost(host_dst=" << host_dst
            << ", src=" << src << ", len=" << len << ')';
  if (task_queuing_) {
    Message mf = {kFlush, 0, 0};
    SendMsg(mf, 0, true);
  }
  CopyFromV(host_dst, p2va(src), len);
}

void HexagonSimulator::CopyHostToDevice(void* dst, const void* host_src,
                                        unsigned len) {
  LOG(INFO) << "HexagonSimulator::CopyHostToDevice(dst=" << dst
            << ", host_src=" << host_src << ", len=" << len << ')';
  CopyToV(p2va(dst), host_src, len);
}

void* HexagonSimulator::Load(const std::string& data, const std::string& fmt) {
  // Load the shared library.
  Message m = {kLoad, static_cast<uint32_t>(data.size() + 1), 0u};
  SendMsg(m, data.c_str(), false);

  CHECK_EQ(sizeof(MsgPointer), m.len);
  MsgPointer mp;
  CopyFromV(&mp, m.va, sizeof(mp));

  return va2p(mp.va);
}

void HexagonSimulator::Unload(void* mod) {
  CHECK(mod);
  Message m = {kUnload, sizeof(MsgPointer), 0u};
  MsgPointer mp = {p2va(mod)};
  SendMsg(m, &mp, false);
}

void* HexagonSimulator::Resolve(const std::string& sym) {
  LOG(INFO) << "HexagonSimulator::Resolve(sym=" << sym << ')';
  Message m = {kResolve, static_cast<uint32_t>(sym.size() + 1), 0u};
  SendMsg(m, sym.c_str(), true);

  CHECK_EQ(sizeof(MsgPointer), m.len);
  MsgPointer mp;
  CopyFromV(&mp, m.va, sizeof(mp));

  LOG(INFO) << "HexagonSimulator::Resolve -> " << std::hex << mp.va
            << std::dec;
  return va2p(mp.va);
}

void HexagonSimulator::Call(void* func, uint32_t* scalar, unsigned sc_num,
                            uint32_t* stack, unsigned st_num) {
  LOG(INFO) << "HexagonSimulator::Call(func=" << std::hex << func
            << ", scalar=" << scalar << ", sc_num=" << std::dec
            << sc_num
            // NOLINTNEXTLINE(build/include_what_you_use)
            << ", stack=" << std::hex << stack << ", st_num=" << std::dec
            << st_num;

  std::vector<uint32_t> data;

  // Copy the MsgCall contents into the data vector as a sequence of uints.
  MsgCall me = {p2va(func), sc_num, st_num};

  CHECK((is_multiple_of<sizeof(MsgCall), sizeof(uint32_t)>()));
  for (unsigned i = 0, e = sizeof(me) / sizeof(uint32_t); i != e; ++i)
    data.push_back(reinterpret_cast<uint32_t*>(&me)[i]);

  // Append the scalar (register) arguments.
  for (unsigned i = 0; i != sc_num; ++i) data.push_back(scalar[i]);
  // Append the stack contents.
  for (unsigned i = 0; i != st_num; ++i) data.push_back(stack[i]);

  std::ostringstream log_data;
  log_data << "data: {" << std::hex;
  for (unsigned i = 0, e = static_cast<uint32_t>(data.size()); i != e; ++i)
    log_data << ' ' << reinterpret_cast<uint32_t*>(data.data())[i];
  log_data << std::dec << " }" << std::flush;
  LOG(INFO) << log_data.str();

  Message m = {kCall, static_cast<uint32_t>(data.size() * sizeof(uint32_t)),
               0u};
  SendMsg(m, data.data(), true);

  if (!task_queuing_) {
    Message mf = {kFlush, 0, 0};
    SendMsg(mf, 0, true);
  }

  std::vector<uint8_t> rv(m.len);
  CopyFromV(rv.data(), m.va, m.len);

  std::ostringstream log_rv;
  log_rv << "HexagonSimulator::Call -> {" << std::hex;
  for (unsigned i = 0, e = std::min<unsigned>(rv.size(), 4u); i != e; ++i)
    log_rv << ' ' << std::setw(2) << std::setfill('0') << uint32_t(rv[i]);
  if (rv.size() > 4) log_rv << "...";
  log_rv << std::dec << " }";
  LOG(INFO) << log_rv.str();
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
