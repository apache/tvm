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

#include <dmlc/logging.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Process.h>

#include <algorithm>
#include <deque>
#include <iomanip>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "../hexagon_module.h"
#include "HexagonWrapper.h"
#include "hexagon_sim_proto.h"

namespace tvm {
namespace runtime {
namespace hexagon {

static_assert(sizeof(HEX_VA_t) == sizeof(uint32_t),
              "Hexagon VA must be uint32");

template <typename T>
struct unalign {
  using type = struct { T value; } __attribute__((aligned(1), packed));
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

using string_list = std::deque<std::string>;

namespace detail {

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args... args) {
  return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}
template <typename T>
std::unique_ptr<T> make_unique(size_t size) {
  using U = typename std::remove_extent<T>::type;
  return std::unique_ptr<T>(new U[size]());
}

// Converter class to translate vector<string> to char**. This relieves the
// user from memory reallocation and copying.
struct non_const_str {
  non_const_str() {}
  explicit non_const_str(const std::string& str)
      : non_const_str(std::vector<std::string>{str}) {}
  explicit non_const_str(const std::vector<std::string>& vec) {
    for (const std::string& s : vec) {
      auto c = detail::make_unique<char[]>(s.size() + 1);
      std::strncpy(c.get(), s.c_str(), s.size() + 1);
      storage_.push_back(std::move(c));
      pointers_.push_back(storage_.back().get());
    }
  }
  non_const_str(non_const_str&& ncs) { *this = std::move(ncs); }
  non_const_str& operator=(non_const_str&& ncs) {
    if (this != &ncs) {
      for (auto& s : ncs.storage_) storage_.push_back(std::move(s));
      for (auto& s : storage_) pointers_.push_back(s.get());
    }
    return *this;
  }
  size_t size() const { return pointers_.size(); }
  operator char*() {
    CHECK_EQ(pointers_.size(), 1);
    return pointers_[0];
  }
  operator char* *() { return pointers_.data(); }

 private:
  std::vector<char*> pointers_;
  std::vector<std::unique_ptr<char[]>> storage_;
};

using MaybeString = llvm::Optional<std::string>;

MaybeString front(const string_list& deq) {
  return !deq.empty() ? MaybeString(deq.front()) : MaybeString();
}

MaybeString pop_front(string_list& deq) {  // NOLINT(*)
  if (deq.empty()) return MaybeString();
  std::string f = deq.front();
  deq.pop_front();
  return MaybeString(f);
}

llvm::Optional<int64_t> to_int(const MaybeString& str) {
  auto none = llvm::Optional<int64_t>();
  if (str.hasValue()) {
    try {
      size_t pos;
      int64_t val = std::stoll(*str, &pos, 0);
      return pos == str->size() ? llvm::Optional<int64_t>(val) : none;
    } catch (std::invalid_argument) {
    }
  }
  return none;
}

llvm::Optional<uint64_t> to_uint(const MaybeString& str) {
  auto none = llvm::Optional<uint64_t>();
  if (str.hasValue()) {
    try {
      size_t pos;
      uint64_t val = std::stoull(*str, &pos, 0);
      return pos == str->size() ? llvm::Optional<uint64_t>(val) : none;
    } catch (std::invalid_argument) {
    }
  }
  return none;
}

llvm::Optional<float> to_float(const MaybeString& str) {
  auto none = llvm::Optional<float>();
  if (str.hasValue()) {
    try {
      size_t pos;
      float val = std::stof(*str, &pos);
      return pos == str->size() ? llvm::Optional<float>(val) : none;
    } catch (std::invalid_argument) {
    }
  }
  return none;
}

llvm::Optional<bool> to_bool(const MaybeString& str) {
  auto none = llvm::Optional<bool>();
  if (auto num = to_int(str)) {
    if (*num == 0) return false;
    if (*num == 1) return true;
    return none;
  }
  if (str) {
    if (*str == "true" || *str == "TRUE") return true;
    if (*str == "false" || *str == "FALSE") return false;
  }
  return none;
}

template <typename T>
using MaybeRange = llvm::Optional<std::pair<T, T>>;

template <typename T, llvm::Optional<T> Parse(const MaybeString&)>
MaybeRange<T> to_range(const MaybeString& str) {
  auto none = MaybeRange<T>();
  if (str && !str->empty()) {
    auto n = str->find('-', 1);
    if (n != std::string::npos) {
      auto begin = Parse(str->substr(0, n));
      auto end = Parse(str->substr(n + 1, str->size() - n - 1));
      if (begin && end) {
        return std::make_pair(*begin, *end);
      }
    }
  }
  return none;
}

}  // namespace detail

class HexagonSimulator final : public tvm::runtime::hexagon::Device {
 public:
  explicit HexagonSimulator(bool enable_queuing);
  ~HexagonSimulator() final {}
  void* Alloc(unsigned size, unsigned align) final;
  void Free(void* ptr) final;
  void* AllocVtcm(unsigned size, unsigned align) final;
  void FreeVtcm(void* ptr) final;
  void CopyDeviceToDevice(void* dst, const void* src, unsigned len) final;
  void CopyDeviceToHost(void* host_dst, const void* src, unsigned len) final;
  void CopyHostToDevice(void* dst, const void* host_src, unsigned len) final;
  void* Load(const std::string& data, const std::string& fmt) final;
  void Unload(void* mod) final;
  void* Resolve(const std::string& sym) final;
  void Call(void* func, uint32_t* scalar, unsigned sc_num, uint32_t* stack,
            unsigned st_num) final;

  static std::string to_string(HEXAPI_Status status);

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

  std::string arch_;
  std::unique_ptr<HexagonWrapper> sim_;
  HEX_VA_t dispatch_v_, message_buffer_v_;
  bool task_queuing_;

  // Sim configuration routines.
  bool Configure(string_list& opts);  // NOLINT(*)

  bool HandleAHBBusPenalty(string_list& rest);      // NOLINT(*)
  bool HandleAHBBusRatio(string_list& rest);        // NOLINT(*)
  bool HandleAHBHighAddr(string_list& rest);        // NOLINT(*)
  bool HandleAHBLowAddr(string_list& rest);         // NOLINT(*)
  bool HandleAXI2BusPenalty(string_list& rest);     // NOLINT(*)
  bool HandleAXI2BusRatio(string_list& rest);       // NOLINT(*)
  bool HandleAXI2HighAddr(string_list& rest);       // NOLINT(*)
  bool HandleAXI2LowAddr(string_list& rest);        // NOLINT(*)
  bool HandleBuildTag(string_list& rest);           // NOLINT(*)
  bool HandleBusPenalty(string_list& rest);         // NOLINT(*)
  bool HandleBusRatio(string_list& rest);           // NOLINT(*)
  bool HandleBusTrace(string_list& rest);           // NOLINT(*)
  bool HandleBypassIdle(string_list& rest);         // NOLINT(*)
  bool HandleConnectionTimeout(string_list& rest);  // NOLINT(*)
  bool HandleCoprocTrace(string_list& rest);        // NOLINT(*)
  bool HandleCoreDump(string_list& rest);           // NOLINT(*)
  bool HandleCosimFile(string_list& rest);          // NOLINT(*)
  bool HandleDCacheTrace(string_list& rest);        // NOLINT(*)
  bool HandleDSPClock(string_list& rest);           // NOLINT(*)
  bool HandleETMCFGBase(string_list& rest);         // NOLINT(*)
  bool HandleGDBServ(string_list& rest);            // NOLINT(*)
  bool HandleHVXLength(string_list& rest);          // NOLINT(*)
  bool HandleICacheTrace(string_list& rest);        // NOLINT(*)
  bool HandleL2CacheTrace(string_list& rest);       // NOLINT(*)
  bool HandleL2CFGBase(string_list& rest);          // NOLINT(*)
  bool HandleL2TCMBase(string_list& rest);          // NOLINT(*)
  bool HandleMemFillRand(string_list& rest);        // NOLINT(*)
  bool HandleMemFill(string_list& rest);            // NOLINT(*)
  bool HandleMemTrace(string_list& rest);           // NOLINT(*)
  bool HandleNullPtr(string_list& rest);            // NOLINT(*)
  bool HandlePacketAnalyze(string_list& rest);      // NOLINT(*)
  bool HandlePCFilter(string_list& rest);           // NOLINT(*)
  bool HandlePCTraceMin(string_list& rest);         // NOLINT(*)
  bool HandlePCTraceNano(string_list& rest);        // NOLINT(*)
  bool HandlePCTrace(string_list& rest);            // NOLINT(*)
  bool HandlePMUStatsFile(string_list& rest);       // NOLINT(*)
  bool HandleProfile(string_list& rest);            // NOLINT(*)
  bool HandleProfileTimeZero(string_list& rest);    // NOLINT(*)
  bool HandleQuiet(string_list& rest);              // NOLINT(*)
  bool HandleReconnect(string_list& rest);          // NOLINT(*)
  bool HandleRTOS(string_list& rest);               // NOLINT(*)
  bool HandleSimErr(string_list& rest);             // NOLINT(*)
  bool HandleSimIn(string_list& rest);              // NOLINT(*)
  bool HandleSimOut(string_list& rest);             // NOLINT(*)
  bool HandleStackStart(string_list& rest);         // NOLINT(*)
  bool HandleStallTrace(string_list& rest);         // NOLINT(*)
  bool HandleStatsFile(string_list& rest);          // NOLINT(*)
  bool HandleSubsystemBase(string_list& rest);      // NOLINT(*)
  bool HandleSymFile(string_list& rest);            // NOLINT(*)
  bool HandleTCM(string_list& rest);                // NOLINT(*)
  bool HandleTCMHighAddr(string_list& rest);        // NOLINT(*)
  bool HandleTCMLowAddr(string_list& rest);         // NOLINT(*)
  bool HandleTimeFilterNS(string_list& rest);       // NOLINT(*)
  bool HandleTiming(string_list& rest);             // NOLINT(*)
  bool HandleUArchTrace(string_list& rest);         // NOLINT(*)
  bool HandleUseFS(string_list& rest);              // NOLINT(*)
  bool HandleV2PTranslation(string_list& rest);     // NOLINT(*)
  bool HandleVerbose(string_list& rest);            // NOLINT(*)

  using MaybeUInt64 = llvm::Optional<uint64_t>;
  using MaybeUIntRange = std::pair<MaybeUInt64, MaybeUInt64>;

  bool should_parse_next(const string_list& rest);
  llvm::Optional<HEXAPI_Interval> to_interval(const detail::MaybeString& str);
  llvm::Optional<HEXAPI_TimingMode> to_timingmode(
      const detail::MaybeString& str);
  llvm::Optional<HEXAPI_VerboseMode> to_verbosemode(
      const detail::MaybeString& str);
  llvm::Optional<HEXAPI_Nullptr> to_nullptr(const detail::MaybeString& str);

  MaybeUIntRange ahb_, axi2_;
  llvm::Optional<uint32_t> debug_port_;
  detail::non_const_str sim_dev_args_;

  using OptionHandler = bool (HexagonSimulator::*)(string_list&);
  static std::map<std::string, OptionHandler> opt_map_;
};

decltype(HexagonSimulator::opt_map_) HexagonSimulator::opt_map_ = {
    {"--ahbbuspenalty", &HexagonSimulator::HandleAHBBusPenalty},
    {"--ahbbusratio", &HexagonSimulator::HandleAHBBusRatio},
    {"--ahb:highaddr", &HexagonSimulator::HandleAHBHighAddr},
    {"--ahb:lowaddr", &HexagonSimulator::HandleAHBLowAddr},
    {"--axi2buspenalty", &HexagonSimulator::HandleAXI2BusPenalty},
    {"--axi2busratio", &HexagonSimulator::HandleAXI2BusRatio},
    {"--axi2:highaddr", &HexagonSimulator::HandleAXI2HighAddr},
    {"--axi2:lowaddr", &HexagonSimulator::HandleAXI2LowAddr},
    {"-b", &HexagonSimulator::HandleBusTrace},
    {"--build_tag", &HexagonSimulator::HandleBuildTag},
    {"--buspenalty", &HexagonSimulator::HandleBusPenalty},
    {"--busratio", &HexagonSimulator::HandleBusRatio},
    {"--bustrace", &HexagonSimulator::HandleBusTrace},
    {"--bypass_idle", &HexagonSimulator::HandleBypassIdle},
    {"--connection_timeout", &HexagonSimulator::HandleConnectionTimeout},
    {"--coproctrace", &HexagonSimulator::HandleCoprocTrace},
    {"--coredump", &HexagonSimulator::HandleCoreDump},
    {"--cosim_file", &HexagonSimulator::HandleCosimFile},
    {"--dcachetrace", &HexagonSimulator::HandleDCacheTrace},
    {"--dsp_clock", &HexagonSimulator::HandleDSPClock},
    {"-E", &HexagonSimulator::HandleSimErr},
    {"--etm_base", &HexagonSimulator::HandleETMCFGBase},
    {"--etmcfg_base", &HexagonSimulator::HandleETMCFGBase},
    {"--gdbserv", &HexagonSimulator::HandleGDBServ},
    {"-G", &HexagonSimulator::HandleGDBServ},
    {"--hvx_length", &HexagonSimulator::HandleHVXLength},
    {"--icachetrace", &HexagonSimulator::HandleICacheTrace},
    {"-I", &HexagonSimulator::HandleSimIn},
    {"--l2cachetrace", &HexagonSimulator::HandleL2CacheTrace},
    {"--l2cfg_base", &HexagonSimulator::HandleL2CFGBase},
    {"--l2tcm_base", &HexagonSimulator::HandleL2TCMBase},
    {"--memfill", &HexagonSimulator::HandleMemFill},
    {"--memfill_rand", &HexagonSimulator::HandleMemFillRand},
    {"--memtrace", &HexagonSimulator::HandleMemTrace},
    {"-m", &HexagonSimulator::HandleMemTrace},
    {"--nullptr", &HexagonSimulator::HandleNullPtr},
    {"-O", &HexagonSimulator::HandleSimOut},
    {"--packet_analyze", &HexagonSimulator::HandlePacketAnalyze},
    {"--pcfilter", &HexagonSimulator::HandlePCFilter},
    {"--pctrace", &HexagonSimulator::HandlePCTrace},
    {"--pctrace_min", &HexagonSimulator::HandlePCTraceMin},
    {"--pctrace_nano", &HexagonSimulator::HandlePCTraceNano},
    {"-p", &HexagonSimulator::HandleProfile},
    {"--pmu_statsfile", &HexagonSimulator::HandlePMUStatsFile},
    {"--profile", &HexagonSimulator::HandleProfile},
    {"--profile_timezero", &HexagonSimulator::HandleProfileTimeZero},
    {"-q", &HexagonSimulator::HandleQuiet},
    {"--quiet", &HexagonSimulator::HandleQuiet},
    {"--reconnect", &HexagonSimulator::HandleReconnect},
    {"--rtos", &HexagonSimulator::HandleRTOS},
    {"-S", &HexagonSimulator::HandleStatsFile},
    {"--sim_err", &HexagonSimulator::HandleSimErr},
    {"--sim_in", &HexagonSimulator::HandleSimIn},
    {"--sim_out", &HexagonSimulator::HandleSimOut},
    {"--stackstart", &HexagonSimulator::HandleStackStart},
    {"--stalltrace", &HexagonSimulator::HandleStallTrace},
    {"--statsfile", &HexagonSimulator::HandleStatsFile},
    {"--subsystem_base", &HexagonSimulator::HandleSubsystemBase},
    {"--symfile", &HexagonSimulator::HandleSymFile},
    {"--tcm", &HexagonSimulator::HandleTCM},
    {"--tcm:highaddr", &HexagonSimulator::HandleTCMHighAddr},
    {"--tcm:lowaddr", &HexagonSimulator::HandleTCMLowAddr},
    {"-t", &HexagonSimulator::HandlePCTrace},
    {"--timefilter_ns", &HexagonSimulator::HandleTimeFilterNS},
    {"--timing", &HexagonSimulator::HandleTiming},
    {"--uarchtrace", &HexagonSimulator::HandleUArchTrace},
    {"-u", &HexagonSimulator::HandlePCTraceMin},
    {"--usefs", &HexagonSimulator::HandleUseFS},
    {"--v2p_translation", &HexagonSimulator::HandleV2PTranslation},
    {"--verbose", &HexagonSimulator::HandleVerbose},
};

#define CHECKED_CALL(func, ...)                            \
  do {                                                     \
    HEXAPI_Status s = sim_->func(__VA_ARGS__);             \
    CHECK_EQ(s, HEX_STAT_SUCCESS)                          \
        << "HexagonSimulator: " #func " failed with code " \
        << HexagonSimulator::to_string(s);                 \
  } while (false)

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

std::shared_ptr<Device> CreateHexagonSimulator() {
  return std::make_shared<HexagonSimulator>(/*enable_queuing=*/true);
}

template <unsigned N>
void HexagonSimulator::CopyNToV(HEX_VA_t dst, const void* host_src) {
  using src_uint_t = typename unalign<typename uint<N>::type>::type;
  auto* ps = reinterpret_cast<const src_uint_t*>(host_src);
  CHECK_EQ(sim_->WriteVirtual(dst, -1u, N, ps->value), HEX_STAT_SUCCESS);
}

template <unsigned N>
void HexagonSimulator::CopyNFromV(void* host_dst, HEX_VA_t src) {
  typename uint<N>::type v;
  CHECK_EQ(sim_->ReadVirtual(src, -1u, N, &v), HEX_STAT_SUCCESS);

  using dst_uint_t = typename unalign<typename uint<N>::type>::type;
  auto* pd = reinterpret_cast<dst_uint_t*>(host_dst);
  pd->value = v;
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
    if (report_cycles) {
      CHECK_EQ(sim_->GetSimulatedCycleCount(&cycles0), HEX_STAT_SUCCESS);
    }

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

HexagonSimulator::HexagonSimulator(bool enable_queuing) {
  task_queuing_ = enable_queuing;

  // The simulator argument string is in the form:
  //   <cpu_ver> <optional_arguments>
  // The optional arguments are seperated with spaces:
  // Ex: --hvx_length 128 --memfill 0 --timing -m output.txt
  const char* sim_args_env = std::getenv("HEXAGON_SIM_ARGS");
  if (sim_args_env == nullptr) sim_args_env = "";
  auto sim_args_iss = std::istringstream(std::string(sim_args_env));
  using iterator = std::istream_iterator<std::string>;
  auto sim_args = string_list(iterator(sim_args_iss), iterator());

  std::string target_str =
      !sim_args.empty() ? *detail::pop_front(sim_args) : std::string("v66");

  arch_ = target_str;
  sim_ =
      detail::make_unique<HexagonWrapper>(detail::non_const_str(target_str));
  LOG(INFO) << "HexagonSimulator: Core version: " << arch_;

  // Locate the sim_dev binary in PATH, or in the current working directory.
  llvm::StringRef sim_dev = "sim_dev";
  detail::MaybeString path_sim_dev =
      llvm::sys::Process::FindInEnvPath("PATH", sim_dev);
  if (!path_sim_dev) {
    if (!llvm::sys::fs::exists(sim_dev)) {
      LOG(FATAL) << "Cannot find sim_dev in PATH.";
    }
    path_sim_dev = sim_dev.str();
  }

  CHECKED_CALL(ConfigureExecutableBinary, path_sim_dev->c_str());

  std::vector<std::string> app_args = {*path_sim_dev};
  if (char* ev = getenv("ADSP_LIBRARY_PATH")) {
    app_args.push_back("-L");
    app_args.push_back(ev);
  }
  sim_dev_args_ = detail::non_const_str(app_args);
  CHECKED_CALL(ConfigureAppCommandLine, sim_dev_args_.size(), sim_dev_args_);

  Configure(sim_args);

  CHECKED_CALL(EndOfConfiguration);
  CHECKED_CALL(LoadExecutableBinary);
  CHECKED_CALL(ReadSymbolValue, "dispatch", &dispatch_v_);
  CHECKED_CALL(ReadSymbolValue, "message_buffer", &message_buffer_v_);
  CHECKED_CALL(SetBreakpoint, dispatch_v_);

  HEXAPI_CoreState core = HEX_CORE_RESET;

  HEX_4u_t result;
  core = sim_->Run(&result);
  if (core != HEX_CORE_BREAKPOINT) {
    LOG(FATAL) << "HexagonSimulator: Run not stopped on breakpoint, "
                  "code="
               << static_cast<int>(core);
  }

  // At this point the simulator has executed the executable's initialization
  // code that could have written to the SSR register.
  // Enable UPCYCLE register.
  HEX_4u_t thread_num;
  CHECKED_CALL(GetCurrentHWThreadNum, &thread_num);
  HEX_4u_t thread_ssr;
  CHECKED_CALL(ReadThreadRegister, thread_num, TH_REG_SSR, &thread_ssr);
  thread_ssr |= (1 << 23);
  CHECKED_CALL(WriteThreadRegister, thread_num, TH_REG_SSR, thread_ssr);
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

void* HexagonSimulator::AllocVtcm(unsigned size, unsigned align) {
  LOG(INFO) << "HexagonSimulator::AllocVtcm(size=" << size
            << ", align=" << align << ')';
  Message m = {kAllocVtcm, sizeof(MsgAlloc), 0u};
  MsgAlloc ma = {size, align};
  SendMsg(m, &ma, true);

  CHECK_EQ(sizeof(MsgPointer), m.len);
  MsgPointer mp;
  CopyFromV(&mp, m.va, m.len);

  LOG(INFO) << "HexagonSimulator::AllocVtcm -> " << std::hex << mp.va
            << std::dec;
  CHECK_NE(mp.va, 0);
  return va2p(mp.va);
}

void HexagonSimulator::FreeVtcm(void* ptr) {}

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
  for (unsigned i = 0, e = static_cast<uint32_t>(data.size()); i != e; ++i) {
    log_data << ' ' << reinterpret_cast<uint32_t*>(data.data())[i];
  }
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
  for (unsigned i = 0, e = std::min<unsigned>(rv.size(), 4u); i != e; ++i) {
    log_rv << ' ' << std::setw(2) << std::setfill('0')
           << static_cast<uint32_t>(rv[i]);
  }
  if (rv.size() > 4) log_rv << "...";
  log_rv << std::dec << " }";
  LOG(INFO) << log_rv.str();
}

bool HexagonSimulator::Configure(string_list& opts) {
  while (!opts.empty()) {
    std::string key = *detail::pop_front(opts);
    auto f = opt_map_.find(key);
    if (f == opt_map_.end()) {
      LOG(FATAL) << "Unrecognized simulator option: " << key;
      // unreachable
    }
    CHECK((this->*f->second)(opts)) << "error handling option: " << key;
  }

  // Check AHB.
  if (ahb_.first.hasValue() && ahb_.second.hasValue()) {
    CHECKED_CALL(ConfigureAHB, *ahb_.first, *ahb_.second);
  } else {
    CHECK(!ahb_.first.hasValue() && !ahb_.second.hasValue())
        << "HexagonSimulator: please specify both low and high addresses "
           "for AHB";
  }

  // Check AXI2.
  if (axi2_.first.hasValue() && axi2_.second.hasValue()) {
    CHECKED_CALL(ConfigureAXI2, *axi2_.first, *axi2_.second);
  } else {
    CHECK(!axi2_.first.hasValue() && !axi2_.second.hasValue())
        << "HexagonSimulator: please specify both low and high addresses "
           "for AXI2";
  }

  return true;
}

bool HexagonSimulator::HandleAHBBusPenalty(string_list& rest) {
  auto penalty = detail::to_uint(detail::pop_front(rest));
  auto interval = to_interval(detail::pop_front(rest));
  if (penalty && interval) {
    CHECKED_CALL(ConfigureAHBBusPenalty, *penalty, *interval);
  }
  return static_cast<bool>(penalty) && static_cast<bool>(interval);
}

bool HexagonSimulator::HandleAHBBusRatio(string_list& rest) {
  auto ratio = detail::to_float(detail::pop_front(rest));
  if (ratio) {
    CHECKED_CALL(ConfigureAHBBusRatio, *ratio);
  }
  return static_cast<bool>(ratio);
}

bool HexagonSimulator::HandleAHBHighAddr(string_list& rest) {
  auto addr = detail::to_uint(detail::pop_front(rest));
  CHECK(addr) << "HexagonSimulator: invalid value for AHB high adddress";
  if (addr) {
    ahb_.second = *addr;
  }
  return static_cast<bool>(addr);
}

bool HexagonSimulator::HandleAHBLowAddr(string_list& rest) {
  auto addr = detail::to_uint(detail::pop_front(rest));
  CHECK(addr) << "HexagonSimulator: invalid value for AHB low adddress";
  if (addr) {
    ahb_.first = *addr;
  }
  return static_cast<bool>(addr);
}

bool HexagonSimulator::HandleAXI2BusPenalty(string_list& rest) {
  auto penalty = detail::to_uint(detail::pop_front(rest));
  auto interval = to_interval(detail::pop_front(rest));
  if (penalty && interval) {
    CHECKED_CALL(ConfigureAXI2BusPenalty, *penalty, *interval);
  }
  return static_cast<bool>(penalty) && static_cast<bool>(interval);
}

bool HexagonSimulator::HandleAXI2BusRatio(string_list& rest) {
  auto ratio = detail::to_float(detail::pop_front(rest));
  if (ratio) {
    CHECKED_CALL(ConfigureAXI2BusRatio, *ratio);
  }
  return static_cast<bool>(ratio);
}

bool HexagonSimulator::HandleAXI2HighAddr(string_list& rest) {
  auto addr = detail::to_uint(detail::pop_front(rest));
  CHECK(addr) << "HexagonSimulator: invalid value for AXI2 high adddress";
  if (addr) {
    axi2_.second = *addr;
  }
  return static_cast<bool>(addr);
}

bool HexagonSimulator::HandleAXI2LowAddr(string_list& rest) {
  auto addr = detail::to_uint(detail::pop_front(rest));
  CHECK(addr) << "HexagonSimulator: invalid value for AXI2 low adddress";
  if (addr) {
    axi2_.first = *addr;
  }
  return static_cast<bool>(addr);
}

bool HexagonSimulator::HandleBuildTag(string_list& rest) {
  sim_->PrintBuildTag();
  return true;
}

bool HexagonSimulator::HandleBusPenalty(string_list& rest) {
  auto penalty = detail::to_uint(detail::pop_front(rest));
  auto interval = to_interval(detail::pop_front(rest));
  if (penalty && interval) {
    CHECKED_CALL(ConfigureBusPenalty, *penalty, *interval);
  }
  return static_cast<bool>(penalty) && static_cast<bool>(interval);
}

bool HexagonSimulator::HandleBusRatio(string_list& rest) {
  auto ratio = detail::to_float(detail::pop_front(rest));
  if (ratio) {
    CHECKED_CALL(ConfigureBusRatio, *ratio);
  }
  return static_cast<bool>(ratio);
}

bool HexagonSimulator::HandleBusTrace(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_BUS, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandleBypassIdle(string_list& rest) {
  CHECKED_CALL(ConfigureBypassIdle, true);
  return true;
}

bool HexagonSimulator::HandleConnectionTimeout(string_list& rest) {
  auto time = detail::to_int(detail::pop_front(rest));
  if (time) {
    CHECKED_CALL(ConfigureConnectionTimeout, *time);
  }
  return static_cast<bool>(time);
}

bool HexagonSimulator::HandleCoprocTrace(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_COPROC, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandleCoreDump(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigureCoreDump, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandleCosimFile(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigureCosim, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandleDCacheTrace(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_DCACHE, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandleDSPClock(string_list& rest) {
  auto freq = detail::to_uint(detail::pop_front(rest));
  if (freq) {
    CHECKED_CALL(ConfigureCoreFrequency, *freq);
  }
  return static_cast<bool>(freq);
}

bool HexagonSimulator::HandleETMCFGBase(string_list& rest) {
  auto base = detail::to_uint(detail::pop_front(rest));
  if (base) {
    CHECKED_CALL(ConfigureEtmcfgBase, *base);
  }
  return static_cast<bool>(base);
}

bool HexagonSimulator::HandleGDBServ(string_list& rest) {
  auto port = detail::to_uint(detail::pop_front(rest));
  if (port) {
    CHECKED_CALL(ConfigureRemoteDebug, *port);
    debug_port_ = *port;
  }
  return static_cast<bool>(port);
}

bool HexagonSimulator::HandleHVXLength(string_list& rest) {
  auto len = detail::to_int(detail::pop_front(rest));
  if (len) {
    CHECKED_CALL(ConfigureHVXLength, *len);
  }
  return static_cast<bool>(len);
}

bool HexagonSimulator::HandleICacheTrace(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_ICACHE, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandleL2CacheTrace(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_L2CACHE, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandleL2CFGBase(string_list& rest) {
  auto base = detail::to_uint(detail::pop_front(rest));
  if (base) {
    CHECKED_CALL(ConfigureL2cfgBase, *base);
  }
  return static_cast<bool>(base);
}

bool HexagonSimulator::HandleL2TCMBase(string_list& rest) {
  auto base = detail::to_uint(detail::pop_front(rest));
  if (base) {
    CHECKED_CALL(ConfigureL2tcmBase, *base);
  }
  return static_cast<bool>(base);
}

bool HexagonSimulator::HandleMemFillRand(string_list& rest) {
  auto seed = detail::to_uint(detail::pop_front(rest));
  if (seed) {
    CHECKED_CALL(ConfigureMemFillRandom, *seed);
  }
  return static_cast<bool>(seed);
}

bool HexagonSimulator::HandleMemFill(string_list& rest) {
  auto val = detail::to_uint(detail::pop_front(rest));
  if (val) {
    CHECKED_CALL(ConfigureMemFill, *val);
  }
  return static_cast<bool>(val);
}

bool HexagonSimulator::HandleMemTrace(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_MEM, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandleNullPtr(string_list& rest) {
  auto behavior = to_nullptr(detail::pop_front(rest));
  if (behavior) {
    CHECKED_CALL(ConfigureNULLPointerBehavior, *behavior);
  }
  return static_cast<bool>(behavior);
}

bool HexagonSimulator::HandlePacketAnalyze(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigurePacketAnalysis, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandlePCFilter(string_list& rest) {
  auto range =
      detail::to_range<uint64_t, detail::to_uint>(detail::pop_front(rest));
  if (range) {
    CHECKED_CALL(ConfigurePCRangeFilter, range->first, range->second);
  }
  return static_cast<bool>(range);
}

bool HexagonSimulator::HandlePCTraceMin(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_PC_MIN, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandlePCTraceNano(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_PC_NANO, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandlePCTrace(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_PC, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandlePMUStatsFile(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigurePmuStatisticsFile, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandleProfile(string_list& rest) {
  auto path = detail::pop_front(rest);
  if (path) {
    CHECKED_CALL(ConfigureGProf, path->c_str());
  }
  return static_cast<bool>(path);
}

bool HexagonSimulator::HandleProfileTimeZero(string_list& rest) {
  auto timezero = detail::to_bool(detail::pop_front(rest));
  if (timezero) {
    CHECKED_CALL(ConfigureProfileMode, *timezero);
  }
  return static_cast<bool>(timezero);
}

bool HexagonSimulator::HandleQuiet(string_list& rest) {
  sim_->VerboseMode(HEX_QUIET);
  return true;
}

bool HexagonSimulator::HandleReconnect(string_list& rest) {
  if (!debug_port_) {
    LOG(FATAL) << "Reconnect error: --reconnect must be specified "
                  "AFTER --gdbserv <port_num>";
  }
  CHECKED_CALL(ConfigureRemoteDebug, *debug_port_, true);
  return true;
}

bool HexagonSimulator::HandleRTOS(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigureOSAwareness, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandleSimErr(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigureSimStderr, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandleSimIn(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigureSimStdin, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandleSimOut(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigureSimStdout, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandleStackStart(string_list& rest) {
  auto base = detail::to_uint(detail::pop_front(rest));
  auto size = detail::to_uint(detail::pop_front(rest));
  if (base && size) {
    CHECKED_CALL(ConfigureStackInfo, *base, *size);
  }
  return static_cast<bool>(base) && static_cast<bool>(size);
}

bool HexagonSimulator::HandleStallTrace(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_STALL, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandleStatsFile(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigureStatisticsFile, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandleSubsystemBase(string_list& rest) {
  auto base = detail::to_uint(detail::pop_front(rest));
  if (base) {
    CHECKED_CALL(ConfigureSubsystemBase, *base);
  }
  return static_cast<bool>(base);
}

bool HexagonSimulator::HandleSymFile(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(AddSymbolFile, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandleTCM(string_list& rest) {
  CHECKED_CALL(ConfigureTimingMode, HEX_TIMING);
  return true;
}

bool HexagonSimulator::HandleTCMHighAddr(string_list& rest) {
  // This option takes an argument, but (the option) is ignored.
  auto addr = detail::to_uint(detail::pop_front(rest));
  return static_cast<bool>(addr);
}

bool HexagonSimulator::HandleTCMLowAddr(string_list& rest) {
  auto addr = detail::to_uint(detail::pop_front(rest));
  if (addr) {
    CHECKED_CALL(ConfigureTCM, *addr);
  }
  return static_cast<bool>(addr);
}

bool HexagonSimulator::HandleTimeFilterNS(string_list& rest) {
  auto range =
      detail::to_range<uint64_t, detail::to_uint>(detail::pop_front(rest));
  if (range) {
    CHECKED_CALL(ConfigureTimeRangeFilter, range->first, HEX_NANOSEC,
                 range->second, HEX_NANOSEC);
  }
  return static_cast<bool>(range);
}

bool HexagonSimulator::HandleTiming(string_list& rest) {
  HEXAPI_TimingMode timing_mode = HEX_TIMING;
  // The argument to --timing is optional.
  if (should_parse_next(rest)) {
    if (auto mode = to_timingmode(detail::pop_front(rest))) {
      timing_mode = *mode;
    } else {
      return false;
    }
  }
  CHECKED_CALL(ConfigureTimingMode, timing_mode);
  return true;
}

bool HexagonSimulator::HandleUArchTrace(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_UARCH, file->c_str());
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandleUseFS(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigureARFilesystem, detail::non_const_str(*file));
  }
  return static_cast<bool>(file);
}

bool HexagonSimulator::HandleV2PTranslation(string_list& rest) {
  auto enable = detail::to_bool(detail::pop_front(rest));
  if (enable) {
    CHECKED_CALL(EnableVirtualToPhysicalTranslation, *enable);
  }
  return static_cast<bool>(enable);
}

bool HexagonSimulator::HandleVerbose(string_list& rest) {
  auto mode = to_verbosemode(detail::pop_front(rest));
  if (mode) {
    sim_->VerboseMode(*mode);
  }
  return static_cast<bool>(mode);
}

bool HexagonSimulator::should_parse_next(const string_list& rest) {
  if (auto str = detail::front(rest)) {
    return str->empty() || str->front() != '-';
  }
  return false;
}

llvm::Optional<HEXAPI_Interval> HexagonSimulator::to_interval(
    const detail::MaybeString& str) {
  auto none = llvm::Optional<HEXAPI_Interval>();
  if (!str) return none;

  if (auto val = detail::to_int(*str)) {
    switch (*val) {
      case HEX_MILLISEC:
      case HEX_MICROSEC:
      case HEX_NANOSEC:
      case HEX_PICOSEC:
      case HEX_PCYCLE:
        return static_cast<HEXAPI_Interval>(*val);
    }
  }

  return llvm::StringSwitch<llvm::Optional<HEXAPI_Interval>>(*str)
      .Case("MILLISEC", HEX_MILLISEC)
      .Case("MICROSEC", HEX_MICROSEC)
      .Case("NANOSEC", HEX_NANOSEC)
      .Case("PICOSEC", HEX_PICOSEC)
      .Case("PCYCLE", HEX_PCYCLE)
      .Default(none);
}

llvm::Optional<HEXAPI_TimingMode> HexagonSimulator::to_timingmode(
    const detail::MaybeString& str) {
  auto none = llvm::Optional<HEXAPI_TimingMode>();
  if (!str) return none;

  if (auto val = detail::to_int(*str)) {
    switch (*val) {
      case HEX_NOTIMING:
      case HEX_TIMING_NODBC:
      case HEX_TIMING:
      case HEX_TIMING_COHERENCY:
        return static_cast<HEXAPI_TimingMode>(*val);
    }
  }

  return llvm::StringSwitch<llvm::Optional<HEXAPI_TimingMode>>(*str)
      .Case("NOTIMING", HEX_NOTIMING)
      .Case("TIMING_NODBC", HEX_TIMING_NODBC)
      .Case("TIMING", HEX_TIMING)
      .Case("TIMING_COHERENCY", HEX_TIMING_COHERENCY)
      .Default(none);
}

llvm::Optional<HEXAPI_VerboseMode> HexagonSimulator::to_verbosemode(
    const detail::MaybeString& str) {
  auto none = llvm::Optional<HEXAPI_VerboseMode>();
  if (!str) return none;

  if (auto val = detail::to_int(*str)) {
    switch (*val) {
      case HEX_SILENT:
      case HEX_QUIET:
      case HEX_NORMAL:
      case HEX_VERBOSE:
      case HEX_REALLY_VERBOSE:
        return static_cast<HEXAPI_VerboseMode>(*val);
    }
  }

  return llvm::StringSwitch<llvm::Optional<HEXAPI_VerboseMode>>(*str)
      .Case("SILENT", HEX_SILENT)
      .Case("QUIET", HEX_QUIET)
      .Case("NORMAL", HEX_NORMAL)
      .Case("VERBOSE", HEX_VERBOSE)
      .Case("REALLY_VERBOSE", HEX_REALLY_VERBOSE)
      .Default(none);
}

llvm::Optional<HEXAPI_Nullptr> HexagonSimulator::to_nullptr(
    const detail::MaybeString& str) {
  auto none = llvm::Optional<HEXAPI_Nullptr>();
  if (!str) return none;

  if (auto val = detail::to_int(*str)) {
    switch (*val) {
      case HEX_NULLPTR_IGNORE:
      case HEX_NULLPTR_WARN:
      case HEX_NULLPTR_FATAL:
      case HEX_NULLPTR_PCZERO:
        return static_cast<HEXAPI_Nullptr>(*val);
    }
  }

  return llvm::StringSwitch<llvm::Optional<HEXAPI_Nullptr>>(*str)
      .Case("IGNORE", HEX_NULLPTR_IGNORE)
      .Case("WARN", HEX_NULLPTR_WARN)
      .Case("FATAL", HEX_NULLPTR_FATAL)
      .Case("PCZERO", HEX_NULLPTR_PCZERO)
      .Default(none);
}

std::string HexagonSimulator::to_string(HEXAPI_Status status) {
  switch (status) {
    case HEX_STAT_ERROR:
      return "ERROR";
    case HEX_STAT_SUCCESS:
      return "SUCCESS";
    case HEX_STAT_CANNOT_CONFIG:
      return "CANNOT_CONFIG";
    case HEX_STAT_INVALID_ARGS:
      return "INVALID_ARGS";
    case HEX_STAT_RANGE_ERROR:
      return "RANGE_ERROR";
    case HEX_STAT_FILE_ACCESS_ERROR:
      return "FILE_ACCESS_ERROR";
    case HEX_STAT_DEVICE_NOT_FOUND:
      return "DEVICE_NOT_FOUND";
    case HEX_STAT_MEM_ACCESS_ERROR:
      return "MEM_ACCESS_ERROR";
    case HEX_STAT_CANNOT_TRANSLATE:
      return "CANNOT_TRANSLATE";
    case HEX_STAT_NO_ACTIVE_THREADS:
      return "NO_ACTIVE_THREADS";
    case HEX_STAT_LOAD_ELF_ERROR:
      return "LOAD_ELF_ERROR";
    case HEX_STAT_CORE_RESET:
      return "CORE_RESET";
    default:
      return "unknown";
  }
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
