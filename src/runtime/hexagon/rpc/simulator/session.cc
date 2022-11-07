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

#include <HexagonWrapper.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
// POSIX includes
#include <dirent.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <deque>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "../../../rpc/rpc_channel.h"
#include "../../../rpc/rpc_endpoint.h"
#include "../../../rpc/rpc_session.h"
#include "hexagon_sim_proto.h"

#define CHECKED_CALL(func, ...)                                               \
  do {                                                                        \
    HEXAPI_Status s = sim_->func(__VA_ARGS__);                                \
    ICHECK_EQ(s, HEX_STAT_SUCCESS)                                            \
        << self_name_ << ": " #func " failed with code " << Status_{s}.str(); \
  } while (false)

namespace tvm {
namespace runtime {
namespace hexagon {

using string_list = std::deque<std::string>;

namespace detail {

// Replacement for llvm::StringSwitch.
template <typename T>
class StringSwitch {
 public:
  explicit StringSwitch(const std::string& key) : key(key) {}
  operator T() const {
    auto f = map.find(key);
    if (f != map.end()) {
      return f->second;
    }
    ICHECK(static_cast<bool>(def_val)) << "default value not set";
    return *def_val;
  }
  StringSwitch& Case(const std::string& key, T val) {
    map.insert(std::make_pair(key, val));
    return *this;
  }
  StringSwitch& Default(T val) {
    ICHECK(!static_cast<bool>(def_val)) << "default value already set";
    def_val = val;
    return *this;
  }

 private:
  const std::string key;
  std::map<std::string, T> map;
  std::optional<T> def_val;
};

using MaybeString = std::optional<std::string>;

MaybeString front(const string_list& deq) {
  return !deq.empty() ? MaybeString(deq.front()) : MaybeString();
}

MaybeString pop_front(string_list& deq) {  // NOLINT(*)
  if (deq.empty()) return MaybeString();
  std::string f = deq.front();
  deq.pop_front();
  return MaybeString(f);
}

// Functions used when parsing the argument string.

std::optional<int64_t> to_int(const MaybeString& str) {
  if (str.has_value()) {
    try {
      size_t pos;
      int64_t val = std::stoll(*str, &pos, 0);
      return pos == str->size() ? std::optional<int64_t>(val) : std::nullopt;
    } catch (std::invalid_argument&) {
    }
  }
  return std::nullopt;
}

std::optional<uint64_t> to_uint(const MaybeString& str) {
  if (str.has_value()) {
    try {
      size_t pos;
      uint64_t val = std::stoull(*str, &pos, 0);
      return pos == str->size() ? std::optional<uint64_t>(val) : std::nullopt;
    } catch (std::invalid_argument&) {
    }
  }
  return std::nullopt;
}

std::optional<float> to_float(const MaybeString& str) {
  if (str.has_value()) {
    try {
      size_t pos;
      float val = std::stof(*str, &pos);
      return pos == str->size() ? std::optional<float>(val) : std::nullopt;
    } catch (std::invalid_argument&) {
    }
  }
  return std::nullopt;
}

std::optional<bool> to_bool(const MaybeString& str) {
  if (auto num = to_int(str)) {
    if (*num == 0) return false;
    if (*num == 1) return true;
    return std::nullopt;
  }
  if (str) {
    if (*str == "true" || *str == "TRUE") return true;
    if (*str == "false" || *str == "FALSE") return false;
  }
  return std::nullopt;
}

template <typename T>
using MaybeRange = std::optional<std::pair<T, T>>;

template <typename T, std::optional<T> Parse(const MaybeString&)>
MaybeRange<T> to_range(const MaybeString& str) {
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
  return std::nullopt;
}

}  // namespace detail

class SimulatorRPCChannel final : public RPCChannel {
 public:
  SimulatorRPCChannel(int stack_size, std::string args);
  ~SimulatorRPCChannel() final;
  size_t Send(const void* data, size_t size) final;
  size_t Recv(void* data, size_t size) final;

 private:
  struct Status_ {
    HEXAPI_Status s;
    std::string str() const;
  };
  struct Core_ {
    HEXAPI_CoreState s;
    std::string str() const;
  };
  struct Cpu_ {
    HEXAPI_Cpu c;
    std::string str() const;
  };
  struct SDKInfo_ {
    SDKInfo_(const std::string& sdk_root, const std::string& cpu);
    std::string root;
    std::string qurt_root;  // sdk_root/rtos/qurt/computevNN.
    std::string runelf;     // Path to runelf.pbn.
    std::string runmain;    // Path to run_main_on_hexagon.
  };

  struct Message_ {
    Message msg;
    std::string str() const;
  };

  Message SendMsg(Message msg);
  Message SendMsg(uint32_t code, uint32_t len, uint32_t va);
  void ReadFromProcess(void* host_dst, HEX_VA_t src, size_t len);
  void WriteToProcess(HEX_VA_t dst, const void* host_src, size_t len);

  static HEX_8u_t PassVirtAddrCallback(void* handle, int threadno, HEX_8u_t RssV, HEX_8u_t RttV,
                                       HEX_8u_t RxxV, HEX_1u_t imm);

  std::optional<HEXAPI_Cpu> GetCPU(const detail::MaybeString& cpu_str);

  // File name templates for mkstemps.
#define SUFFIX ".cfg"
  static constexpr int template_length_ = sizeof("temp-xxxx-XXXXXX" SUFFIX);  // == strlen() + 1
  char osam_file_[template_length_] = "temp-osam-XXXXXX" SUFFIX;
  char cosim_file_[template_length_] = "temp-q6ss-XXXXXX" SUFFIX;
  const int suffix_len_ = strlen(SUFFIX);
#undef SUFFIX

  static const constexpr char* self_name_ = "SimulatorRPCChannel";
  static const constexpr char* default_cpu_ = "v68";
  std::string cpu_;

  HEX_VA_t dispatch_v_, message_buffer_v_;
  std::unique_ptr<HexagonWrapper> sim_;

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

  using MaybeUInt64 = std::optional<uint64_t>;
  using MaybeUIntRange = std::pair<MaybeUInt64, MaybeUInt64>;

  bool should_parse_next(const string_list& rest);
  std::optional<HEXAPI_Interval> to_interval(const detail::MaybeString& str);
  std::optional<HEXAPI_TimingMode> to_timingmode(const detail::MaybeString& str);
  std::optional<HEXAPI_VerboseMode> to_verbosemode(const detail::MaybeString& str);
  std::optional<HEXAPI_Nullptr> to_nullptr(const detail::MaybeString& str);

  MaybeUIntRange ahb_, axi2_;
  std::optional<uint32_t> debug_port_;

  using OptionHandler = bool (SimulatorRPCChannel::*)(string_list&);
  static std::map<std::string, OptionHandler> opt_map_;
};

const constexpr char* SimulatorRPCChannel::self_name_;
const constexpr char* SimulatorRPCChannel::default_cpu_;

decltype(SimulatorRPCChannel::opt_map_) SimulatorRPCChannel::opt_map_ = {
    {"--ahbbuspenalty", &SimulatorRPCChannel::HandleAHBBusPenalty},
    {"--ahbbusratio", &SimulatorRPCChannel::HandleAHBBusRatio},
    {"--ahb:highaddr", &SimulatorRPCChannel::HandleAHBHighAddr},
    {"--ahb:lowaddr", &SimulatorRPCChannel::HandleAHBLowAddr},
    {"--axi2buspenalty", &SimulatorRPCChannel::HandleAXI2BusPenalty},
    {"--axi2busratio", &SimulatorRPCChannel::HandleAXI2BusRatio},
    {"--axi2:highaddr", &SimulatorRPCChannel::HandleAXI2HighAddr},
    {"--axi2:lowaddr", &SimulatorRPCChannel::HandleAXI2LowAddr},
    {"-b", &SimulatorRPCChannel::HandleBusTrace},
    {"--build_tag", &SimulatorRPCChannel::HandleBuildTag},
    {"--buspenalty", &SimulatorRPCChannel::HandleBusPenalty},
    {"--busratio", &SimulatorRPCChannel::HandleBusRatio},
    {"--bustrace", &SimulatorRPCChannel::HandleBusTrace},
    {"--bypass_idle", &SimulatorRPCChannel::HandleBypassIdle},
    {"--connection_timeout", &SimulatorRPCChannel::HandleConnectionTimeout},
    {"--coproctrace", &SimulatorRPCChannel::HandleCoprocTrace},
    {"--coredump", &SimulatorRPCChannel::HandleCoreDump},
    {"--cosim_file", &SimulatorRPCChannel::HandleCosimFile},
    {"--dcachetrace", &SimulatorRPCChannel::HandleDCacheTrace},
    {"--dsp_clock", &SimulatorRPCChannel::HandleDSPClock},
    {"-E", &SimulatorRPCChannel::HandleSimErr},
    {"--etm_base", &SimulatorRPCChannel::HandleETMCFGBase},
    {"--etmcfg_base", &SimulatorRPCChannel::HandleETMCFGBase},
    {"--gdbserv", &SimulatorRPCChannel::HandleGDBServ},
    {"-G", &SimulatorRPCChannel::HandleGDBServ},
    {"--hvx_length", &SimulatorRPCChannel::HandleHVXLength},
    {"--icachetrace", &SimulatorRPCChannel::HandleICacheTrace},
    {"-I", &SimulatorRPCChannel::HandleSimIn},
    {"--l2cachetrace", &SimulatorRPCChannel::HandleL2CacheTrace},
    {"--l2cfg_base", &SimulatorRPCChannel::HandleL2CFGBase},
    {"--l2tcm_base", &SimulatorRPCChannel::HandleL2TCMBase},
    {"--memfill", &SimulatorRPCChannel::HandleMemFill},
    {"--memfill_rand", &SimulatorRPCChannel::HandleMemFillRand},
    {"--memtrace", &SimulatorRPCChannel::HandleMemTrace},
    {"-m", &SimulatorRPCChannel::HandleMemTrace},
    {"--nullptr", &SimulatorRPCChannel::HandleNullPtr},
    {"-O", &SimulatorRPCChannel::HandleSimOut},
    {"--packet_analyze", &SimulatorRPCChannel::HandlePacketAnalyze},
    {"--pcfilter", &SimulatorRPCChannel::HandlePCFilter},
    {"--pctrace", &SimulatorRPCChannel::HandlePCTrace},
    {"--pctrace_min", &SimulatorRPCChannel::HandlePCTraceMin},
    {"--pctrace_nano", &SimulatorRPCChannel::HandlePCTraceNano},
    {"-p", &SimulatorRPCChannel::HandleProfile},
    {"--pmu_statsfile", &SimulatorRPCChannel::HandlePMUStatsFile},
    {"--profile", &SimulatorRPCChannel::HandleProfile},
    {"--profile_timezero", &SimulatorRPCChannel::HandleProfileTimeZero},
    {"-q", &SimulatorRPCChannel::HandleQuiet},
    {"--quiet", &SimulatorRPCChannel::HandleQuiet},
    {"--reconnect", &SimulatorRPCChannel::HandleReconnect},
    {"--rtos", &SimulatorRPCChannel::HandleRTOS},
    {"-S", &SimulatorRPCChannel::HandleStatsFile},
    {"--sim_err", &SimulatorRPCChannel::HandleSimErr},
    {"--sim_in", &SimulatorRPCChannel::HandleSimIn},
    {"--sim_out", &SimulatorRPCChannel::HandleSimOut},
    {"--stackstart", &SimulatorRPCChannel::HandleStackStart},
    {"--stalltrace", &SimulatorRPCChannel::HandleStallTrace},
    {"--statsfile", &SimulatorRPCChannel::HandleStatsFile},
    {"--subsystem_base", &SimulatorRPCChannel::HandleSubsystemBase},
    {"--symfile", &SimulatorRPCChannel::HandleSymFile},
    {"--tcm", &SimulatorRPCChannel::HandleTCM},
    {"--tcm:highaddr", &SimulatorRPCChannel::HandleTCMHighAddr},
    {"--tcm:lowaddr", &SimulatorRPCChannel::HandleTCMLowAddr},
    {"-t", &SimulatorRPCChannel::HandlePCTrace},
    {"--timefilter_ns", &SimulatorRPCChannel::HandleTimeFilterNS},
    {"--timing", &SimulatorRPCChannel::HandleTiming},
    {"--uarchtrace", &SimulatorRPCChannel::HandleUArchTrace},
    {"-u", &SimulatorRPCChannel::HandlePCTraceMin},
    {"--usefs", &SimulatorRPCChannel::HandleUseFS},
    {"--v2p_translation", &SimulatorRPCChannel::HandleV2PTranslation},
    {"--verbose", &SimulatorRPCChannel::HandleVerbose},
};

std::string SimulatorRPCChannel::Status_::str() const {
  switch (s) {
    case HEX_STAT_ERROR:
      return "HEX_STAT_ERROR";
    case HEX_STAT_SUCCESS:
      return "HEX_STAT_SUCCESS";
    case HEX_STAT_CANNOT_CONFIG:
      return "HEX_STAT_CANNOT_CONFIG";
    case HEX_STAT_INVALID_ARGS:
      return "HEX_STAT_INVALID_ARGS";
    case HEX_STAT_RANGE_ERROR:
      return "HEX_STAT_RANGE_ERROR";
    case HEX_STAT_FILE_ACCESS_ERROR:
      return "HEX_STAT_FILE_ACCESS_ERROR";
    case HEX_STAT_DEVICE_NOT_FOUND:
      return "HEX_STAT_DEVICE_NOT_FOUND";
    case HEX_STAT_MEM_ACCESS_ERROR:
      return "HEX_STAT_MEM_ACCESS_ERROR";
    case HEX_STAT_CANNOT_TRANSLATE:
      return "HEX_STAT_CANNOT_TRANSLATE";
    case HEX_STAT_NO_ACTIVE_THREADS:
      return "HEX_STAT_NO_ACTIVE_THREADS";
    case HEX_STAT_LOAD_ELF_ERROR:
      return "HEX_STAT_LOAD_ELF_ERROR";
    case HEX_STAT_CORE_RESET:
      return "HEX_STAT_CORE_RESET";
    default:
      break;
  }
  return std::to_string(static_cast<int>(s));
}

std::string SimulatorRPCChannel::Core_::str() const {
  switch (s) {
    case HEX_CORE_SUCCESS:
      return "HEX_CORE_SUCCESS";
    case HEX_CORE_FINISHED:
      return "HEX_CORE_FINISHED";
    case HEX_CORE_RESET:
      return "HEX_CORE_RESET";
    case HEX_CORE_BREAKPOINT:
      return "HEX_CORE_BREAKPOINT";
    case HEX_CORE_ASYNCHRONOUS_BREAK:
      return "HEX_CORE_ASYNCHRONOUS_BREAK";
    case HEX_CORE_ERROR:
      return "HEX_CORE_ERROR";
    default:
      break;
  }
  return std::to_string(static_cast<int>(s));
}

std::string SimulatorRPCChannel::Cpu_::str() const {
  switch (c) {
    case HEX_CPU_V65:
      return "v65";
    case HEX_CPU_V66:
      return "v66";
    case HEX_CPU_V68:
      return "v68";
    case HEX_CPU_V69:
      return "v69";
    default:
      break;
  }
  return default_cpu_;
}

// LOG(FATAL) always throws an exception or terminates the
// process, but the compiler doesn't know that.
#if (__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#endif

#if (__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreturn-type"
#endif

std::string SimulatorRPCChannel::Message_::str() const {
  switch (msg.code) {
    case Message::kNone:
      return "kNone";
    case Message::kAck:
      return "kAck";
    case Message::kTerminate:
      return "kTerminate";
    case Message::kReceiveStart:
      return "kReceiveStart";
    case Message::kReceiveEnd:
      return "kReceiveEnd";
    case Message::kSendStart:
      return "kSendStart";
    case Message::kSendEnd:
      return "kSendEnd";
    default:
      LOG(FATAL) << "Internal error: Unrecognized code value: " << msg.code;
      break;
  }
}

#if (__GNUC__)
#pragma GCC diagnostic pop
#endif

#if (__clang__)
#pragma GCC diagnostic pop
#endif

SimulatorRPCChannel::SDKInfo_::SDKInfo_(const std::string& sdk_root, const std::string& cpu)
    : root(sdk_root) {
  // For v69 chips, still look for v68 in the directory names.
  std::string check_cpu = cpu == "v69" ? "v68" : cpu;

  qurt_root = root + "/rtos/qurt/compute" + check_cpu;
  runelf = qurt_root + "/sdksim_bin/runelf.pbn";

  // The "run_main_on_hexagon_sim" binary lives in a subdirectory that looks
  // like "[...]on_hexagon/ship/hexagon_toolv84_v68/run_main_on_hexagon_sim".
  // We need to get the right "hexagon_toolv..." component, based mostly on
  // the cpu version.
  std::vector<std::string> dir_names;

  DIR* dir = opendir((root + "/libs/run_main_on_hexagon/ship").c_str());
  ICHECK(dir != nullptr) << "Cannot read directory " << root + "/libs/run_main_on_hexagon/ship";
  while (dirent* d = readdir(dir)) {
    if (d->d_type != DT_DIR) continue;

    std::string name = d->d_name;
    // Note: The first substr is always safe, and the second only executes
    // when "name" is at least 13 characters long.
    if (name.substr(0, 13) == "hexagon_toolv" && name.substr(name.size() - 3, 3) == check_cpu) {
      dir_names.push_back(name);
    }
  }
  closedir(dir);
  ICHECK(!dir_names.empty());

  auto max_it = std::max_element(dir_names.begin(), dir_names.end());
  runmain = root + "/libs/run_main_on_hexagon/ship/" + *max_it + "/run_main_on_hexagon_sim";
}

HEX_8u_t SimulatorRPCChannel::PassVirtAddrCallback(void* handle, int threadno, HEX_8u_t RssV,
                                                   HEX_8u_t RttV, HEX_8u_t RxxV, HEX_1u_t imm) {
  // Rssv = combine(&message_buffer, &dispatch)
  auto* rpc = reinterpret_cast<SimulatorRPCChannel*>(handle);
  rpc->dispatch_v_ = RssV & ~0u;  // ~0u is uint32_t
  rpc->message_buffer_v_ = RssV >> 32;

  LOG(INFO) << "dispatch:" << reinterpret_cast<void*>(rpc->dispatch_v_)
            << ", message buffer:" << reinterpret_cast<void*>(rpc->message_buffer_v_);
  HEXAPI_Status s = rpc->sim_->SetBreakpoint(rpc->dispatch_v_);
  ICHECK_EQ(s, HEX_STAT_SUCCESS) << self_name_ << ": SetBreakpoint failed with code "
                                 << Status_{s}.str();
  return RssV;
}

std::optional<HEXAPI_Cpu> SimulatorRPCChannel::GetCPU(const detail::MaybeString& cpu_str) {
  if (!cpu_str) return std::nullopt;
  return detail::StringSwitch<std::optional<HEXAPI_Cpu>>(*cpu_str)
      .Case("v65", HEX_CPU_V65)
      .Case("v66", HEX_CPU_V66)
      .Case("v68", HEX_CPU_V68)
      .Case("v69", HEX_CPU_V69)
      .Default(std::nullopt);
}

SimulatorRPCChannel::SimulatorRPCChannel(int stack_size, std::string args) {
  const char* sdk_root_env = std::getenv("HEXAGON_SDK_ROOT");
  ICHECK(sdk_root_env != nullptr) << "Please set HEXAGON_SDK_ROOT";
  const char* toolchain_env = std::getenv("HEXAGON_TOOLCHAIN");
  ICHECK(toolchain_env != nullptr) << "Please set HEXAGON_TOOLCHAIN";

  std::string sdk_root(sdk_root_env);
  std::string toolchain(toolchain_env);

  auto sim_args_iss = std::istringstream(args);
  using iterator = std::istream_iterator<std::string>;
  auto sim_args = string_list(iterator(sim_args_iss), iterator());

  detail::MaybeString target_str = detail::pop_front(sim_args);
  auto maybe_cpu = GetCPU(target_str);
  if (!maybe_cpu) {
    if (!target_str || target_str->empty()) {
      LOG(INFO) << "CPU not given, defaulting to " << default_cpu_;
      maybe_cpu = GetCPU(std::string(default_cpu_));
    } else {
      LOG(FATAL) << "Invalid CPU name " << *target_str;
    }
  }
  cpu_ = Cpu_{*maybe_cpu}.str();
  sim_ = std::make_unique<HexagonWrapper>(*maybe_cpu, (toolchain + "/lib/iss").c_str());
  SDKInfo_ sdk(sdk_root, cpu_);

  // Prepare the osam.cfg file.
  int fd_osam = mkstemps(osam_file_, suffix_len_);
  ICHECK_GE(fd_osam, 0);
  std::string osam_str = sdk.qurt_root + "/debugger/lnx64/qurt_model.so";
  ICHECK_EQ(write(fd_osam, osam_str.c_str(), osam_str.size()), osam_str.size());
  close(fd_osam);
  // Prepare the q6ss.cfg file.
  int fd_cosim = mkstemps(cosim_file_, suffix_len_);
  ICHECK_GE(fd_cosim, 0);
  std::string cosim_str =
      toolchain +
      "/lib/iss/qtimer.so --csr_base=0xFC900000 --irq_p=1 --freq=19200000 --cnttid=1\n" +
      toolchain + "/lib/iss/l2vic.so 32 0xFC910000";
  ICHECK_EQ(write(fd_cosim, cosim_str.c_str(), cosim_str.size()), cosim_str.size());
  close(fd_cosim);

  CHECKED_CALL(ConfigureL2tcmBase, 0xD800);
  CHECKED_CALL(ConfigureARFilesystem, &std::string(".")[0]);
  CHECKED_CALL(ConfigureOSAwareness, osam_file_);
  CHECKED_CALL(ConfigureCosim, cosim_file_);
  CHECKED_CALL(ConfigureExecutableBinary, sdk.runelf.c_str());

  std::string stack_arg =
      stack_size > 0 ? std::string(" -stack_size=") + std::to_string(stack_size) : "";
  std::string cmdline = sdk.runelf + " " + sdk.runmain + stack_arg + " -- libhexagon_rpc_sim.so";
  char* parg = &cmdline[0];
  CHECKED_CALL(ConfigureAppCommandLine, 1, &parg);

  // Configure the simulator.
  Configure(sim_args);

  CHECKED_CALL(EndOfConfiguration);
  CHECKED_CALL(AddUserDefinedInstCallback, this, &PassVirtAddrCallback);

  // Start the initial run, until the callback is executed.
  HEX_4u_t result;
  HEXAPI_CoreState core = sim_->Run(&result);
  if (core != HEX_CORE_BREAKPOINT) {
    LOG(FATAL) << self_name_ << ": Run not stopped on breakpoint, code=" << Core_{core}.str();
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

SimulatorRPCChannel::~SimulatorRPCChannel() {
  SendMsg(Message::kTerminate, 0, Message::null_va);

  HEX_4u_t result;
  HEXAPI_CoreState core = sim_->Run(&result);
  ICHECK_EQ(core, HEX_CORE_FINISHED);

  unlink(osam_file_);
  unlink(cosim_file_);
}

size_t SimulatorRPCChannel::Send(const void* data, size_t size) {
  ICHECK(size <= std::numeric_limits<uint32_t>::max());

  Message reply_start =
      SendMsg(Message::kReceiveStart, static_cast<uint32_t>(size), Message::null_va);
  ICHECK_EQ(reply_start.code, Message::kAck);
  ICHECK_GE(reply_start.len, size);
  ICHECK_NE(reply_start.va, Message::null_va);

  WriteToProcess(reply_start.va, data, size);

  Message reply_end = SendMsg(Message::kReceiveEnd, static_cast<uint32_t>(size), reply_start.va);
  ICHECK_EQ(reply_end.code, Message::kAck);
  return size;
}

size_t SimulatorRPCChannel::Recv(void* data, size_t size) {
  ICHECK(size <= std::numeric_limits<uint32_t>::max());

  Message reply_start = SendMsg(Message::kSendStart, static_cast<uint32_t>(size), Message::null_va);
  ICHECK_EQ(reply_start.code, Message::kAck);
  ICHECK_GE(reply_start.len, size);
  ICHECK_NE(reply_start.va, Message::null_va);

  ReadFromProcess(data, reply_start.va, size);

  Message reply_end = SendMsg(Message::kSendEnd, static_cast<uint32_t>(size), reply_start.va);
  ICHECK_EQ(reply_end.code, Message::kAck);
  return size;
}

Message SimulatorRPCChannel::SendMsg(Message msg) {
  auto run = [this]() {
    HEXAPI_CoreState core = HEX_CORE_RESET;
    HEX_4u_t result;

    core = sim_->Run(&result);
    Core_ core_ = {core};
    ICHECK_EQ(core, HEX_CORE_BREAKPOINT)
        << "Expecting HEX_CORE_BREAKPOINT, received: " << core_.str();
  };

  WriteToProcess(message_buffer_v_, &msg, sizeof msg);
  run();

  Message ret = {0};
  ReadFromProcess(&ret, message_buffer_v_, sizeof ret);
  return ret;
}

Message SimulatorRPCChannel::SendMsg(uint32_t code, uint32_t len, uint32_t va) {
  Message m;
  m.code = code;
  m.len = len;
  m.va = va;
  return SendMsg(m);
}

void SimulatorRPCChannel::ReadFromProcess(void* host_dst, HEX_VA_t src, size_t len) {
  if (len == 0) return;

  auto* dst = reinterpret_cast<char*>(host_dst);

  while (len > 0) {
    uint32_t src_align = 1u << __builtin_ctz(src);  // Max pow-of-2 dividing src.
    uint32_t len_align = 1u << __builtin_ctz(len);  // Max pow-of-2 dividing len.
    // The transfer behaves as a bus transaction, so it has to be of a size
    // that is a power of 2 not exceeding 8, and the remote address has to
    // be aligned to at least the transfer size.
    auto read_size = std::min<size_t>({src_align, len_align, 8});
    CHECKED_CALL(ReadVirtual, src, /*asid*/ -1u, read_size, dst);

    src += read_size;
    dst += read_size;
    len -= read_size;
  }
}

void SimulatorRPCChannel::WriteToProcess(HEX_VA_t dst, const void* host_src, size_t len) {
  if (len == 0) return;

  auto* src = reinterpret_cast<const char*>(host_src);

  while (len > 0) {
    uint32_t dst_align = 1u << __builtin_ctz(dst);  // Max pow-of-2 dividing dst.
    uint32_t len_align = 1u << __builtin_ctz(len);  // Max pow-of-2 dividing len.
    // The transfer behaves as a bus transaction, so it has to be of a size
    // that is a power of 2 not exceeding 8, and the remote address has to
    // be aligned to at least the transfer size.
    auto write_size = std::min<size_t>({dst_align, len_align, 8});
    HEX_8u_t val = 0;
    memcpy(&val, src, write_size);
    CHECKED_CALL(WriteVirtual, dst, /*asid*/ -1u, write_size, val);

    src += write_size;
    dst += write_size;
    len -= write_size;
  }
}

// Configuration functions

bool SimulatorRPCChannel::Configure(string_list& opts) {
  while (!opts.empty()) {
    std::string key = *detail::pop_front(opts);
    auto f = opt_map_.find(key);
    if (f == opt_map_.end()) {
      LOG(FATAL) << "Unrecognized simulator option: " << key;
      // unreachable
    }
    ICHECK((this->*f->second)(opts)) << "error handling option: " << key;
  }

  // Check AHB.
  if (ahb_.first.has_value() && ahb_.second.has_value()) {
    CHECKED_CALL(ConfigureAHB, *ahb_.first, *ahb_.second);
  } else {
    ICHECK(!ahb_.first.has_value() && !ahb_.second.has_value())
        << self_name_ << ": please specify both low and high addresses for AHB";
  }

  // Check AXI2.
  if (axi2_.first.has_value() && axi2_.second.has_value()) {
    CHECKED_CALL(ConfigureAXI2, *axi2_.first, *axi2_.second);
  } else {
    ICHECK(!axi2_.first.has_value() && !axi2_.second.has_value())
        << self_name_ << ": please specify both low and high addresses for AXI2";
  }

  return true;
}

bool SimulatorRPCChannel::HandleAHBBusPenalty(string_list& rest) {
  auto penalty = detail::to_uint(detail::pop_front(rest));
  auto interval = to_interval(detail::pop_front(rest));
  if (penalty && interval) {
    CHECKED_CALL(ConfigureAHBBusPenalty, *penalty, *interval);
  }
  return static_cast<bool>(penalty) && static_cast<bool>(interval);
}

bool SimulatorRPCChannel::HandleAHBBusRatio(string_list& rest) {
  auto ratio = detail::to_float(detail::pop_front(rest));
  if (ratio) {
    CHECKED_CALL(ConfigureAHBBusRatio, *ratio);
  }
  return static_cast<bool>(ratio);
}

bool SimulatorRPCChannel::HandleAHBHighAddr(string_list& rest) {
  auto addr = detail::to_uint(detail::pop_front(rest));
  ICHECK(addr) << self_name_ << ": invalid value for AHB high adddress";
  if (addr) {
    ahb_.second = *addr;
  }
  return static_cast<bool>(addr);
}

bool SimulatorRPCChannel::HandleAHBLowAddr(string_list& rest) {
  auto addr = detail::to_uint(detail::pop_front(rest));
  ICHECK(addr) << self_name_ << ": invalid value for AHB low adddress";
  if (addr) {
    ahb_.first = *addr;
  }
  return static_cast<bool>(addr);
}

bool SimulatorRPCChannel::HandleAXI2BusPenalty(string_list& rest) {
  auto penalty = detail::to_uint(detail::pop_front(rest));
  auto interval = to_interval(detail::pop_front(rest));
  if (penalty && interval) {
    CHECKED_CALL(ConfigureAXI2BusPenalty, *penalty, *interval);
  }
  return static_cast<bool>(penalty) && static_cast<bool>(interval);
}

bool SimulatorRPCChannel::HandleAXI2BusRatio(string_list& rest) {
  auto ratio = detail::to_float(detail::pop_front(rest));
  if (ratio) {
    CHECKED_CALL(ConfigureAXI2BusRatio, *ratio);
  }
  return static_cast<bool>(ratio);
}

bool SimulatorRPCChannel::HandleAXI2HighAddr(string_list& rest) {
  auto addr = detail::to_uint(detail::pop_front(rest));
  ICHECK(addr) << self_name_ << ": invalid value for AXI2 high adddress";
  if (addr) {
    axi2_.second = *addr;
  }
  return static_cast<bool>(addr);
}

bool SimulatorRPCChannel::HandleAXI2LowAddr(string_list& rest) {
  auto addr = detail::to_uint(detail::pop_front(rest));
  ICHECK(addr) << self_name_ << ": invalid value for AXI2 low adddress";
  if (addr) {
    axi2_.first = *addr;
  }
  return static_cast<bool>(addr);
}

bool SimulatorRPCChannel::HandleBuildTag(string_list& rest) {
  sim_->PrintBuildTag();
  return true;
}

bool SimulatorRPCChannel::HandleBusPenalty(string_list& rest) {
  auto penalty = detail::to_uint(detail::pop_front(rest));
  auto interval = to_interval(detail::pop_front(rest));
  if (penalty && interval) {
    CHECKED_CALL(ConfigureBusPenalty, *penalty, *interval);
  }
  return static_cast<bool>(penalty) && static_cast<bool>(interval);
}

bool SimulatorRPCChannel::HandleBusRatio(string_list& rest) {
  auto ratio = detail::to_float(detail::pop_front(rest));
  if (ratio) {
    CHECKED_CALL(ConfigureBusRatio, *ratio);
  }
  return static_cast<bool>(ratio);
}

bool SimulatorRPCChannel::HandleBusTrace(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_BUS, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandleBypassIdle(string_list& rest) {
  CHECKED_CALL(ConfigureBypassIdle, true);
  return true;
}

bool SimulatorRPCChannel::HandleConnectionTimeout(string_list& rest) {
  auto time = detail::to_int(detail::pop_front(rest));
  if (time) {
    CHECKED_CALL(ConfigureConnectionTimeout, *time);
  }
  return static_cast<bool>(time);
}

bool SimulatorRPCChannel::HandleCoprocTrace(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_COPROC, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandleCoreDump(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigureCoreDump, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandleCosimFile(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigureCosim, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandleDCacheTrace(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_DCACHE, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandleDSPClock(string_list& rest) {
  auto freq = detail::to_uint(detail::pop_front(rest));
  if (freq) {
    CHECKED_CALL(ConfigureCoreFrequency, *freq);
  }
  return static_cast<bool>(freq);
}

bool SimulatorRPCChannel::HandleETMCFGBase(string_list& rest) {
  auto base = detail::to_uint(detail::pop_front(rest));
  if (base) {
    CHECKED_CALL(ConfigureEtmcfgBase, *base);
  }
  return static_cast<bool>(base);
}

bool SimulatorRPCChannel::HandleGDBServ(string_list& rest) {
  auto port = detail::to_uint(detail::pop_front(rest));
  if (port) {
    CHECKED_CALL(ConfigureRemoteDebug, *port);
    debug_port_ = *port;
  }
  return static_cast<bool>(port);
}

bool SimulatorRPCChannel::HandleHVXLength(string_list& rest) {
  auto len = detail::to_int(detail::pop_front(rest));
  if (len) {
    CHECKED_CALL(ConfigureHVXLength, *len);
  }
  return static_cast<bool>(len);
}

bool SimulatorRPCChannel::HandleICacheTrace(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_ICACHE, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandleL2CacheTrace(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_L2CACHE, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandleL2CFGBase(string_list& rest) {
  auto base = detail::to_uint(detail::pop_front(rest));
  if (base) {
    CHECKED_CALL(ConfigureL2cfgBase, *base);
  }
  return static_cast<bool>(base);
}

bool SimulatorRPCChannel::HandleL2TCMBase(string_list& rest) {
  auto base = detail::to_uint(detail::pop_front(rest));
  if (base) {
    CHECKED_CALL(ConfigureL2tcmBase, *base);
  }
  return static_cast<bool>(base);
}

bool SimulatorRPCChannel::HandleMemFillRand(string_list& rest) {
  auto seed = detail::to_uint(detail::pop_front(rest));
  if (seed) {
    CHECKED_CALL(ConfigureMemFillRandom, *seed);
  }
  return static_cast<bool>(seed);
}

bool SimulatorRPCChannel::HandleMemFill(string_list& rest) {
  auto val = detail::to_uint(detail::pop_front(rest));
  if (val) {
    CHECKED_CALL(ConfigureMemFill, *val);
  }
  return static_cast<bool>(val);
}

bool SimulatorRPCChannel::HandleMemTrace(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_MEM, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandleNullPtr(string_list& rest) {
  auto behavior = to_nullptr(detail::pop_front(rest));
  if (behavior) {
    CHECKED_CALL(ConfigureNULLPointerBehavior, *behavior);
  }
  return static_cast<bool>(behavior);
}

bool SimulatorRPCChannel::HandlePacketAnalyze(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigurePacketAnalysis, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandlePCFilter(string_list& rest) {
  auto range = detail::to_range<uint64_t, detail::to_uint>(detail::pop_front(rest));
  if (range) {
    CHECKED_CALL(ConfigurePCRangeFilter, range->first, range->second);
  }
  return static_cast<bool>(range);
}

bool SimulatorRPCChannel::HandlePCTraceMin(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_PC_MIN, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandlePCTraceNano(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_PC_NANO, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandlePCTrace(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_PC, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandlePMUStatsFile(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigurePmuStatisticsFile, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandleProfile(string_list& rest) {
  auto path = detail::pop_front(rest);
  if (path) {
    CHECKED_CALL(ConfigureGProf, path->c_str());
  }
  return static_cast<bool>(path);
}

bool SimulatorRPCChannel::HandleProfileTimeZero(string_list& rest) {
  auto timezero = detail::to_bool(detail::pop_front(rest));
  if (timezero) {
    CHECKED_CALL(ConfigureProfileMode, *timezero);
  }
  return static_cast<bool>(timezero);
}

bool SimulatorRPCChannel::HandleQuiet(string_list& rest) {
  sim_->VerboseMode(HEX_QUIET);
  return true;
}

bool SimulatorRPCChannel::HandleReconnect(string_list& rest) {
  if (!debug_port_) {
    LOG(FATAL) << "Reconnect error: --reconnect must be specified "
                  "AFTER --gdbserv <port_num>";
  }
  CHECKED_CALL(ConfigureRemoteDebug, *debug_port_, true);
  return true;
}

bool SimulatorRPCChannel::HandleRTOS(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigureOSAwareness, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandleSimErr(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigureSimStderr, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandleSimIn(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigureSimStdin, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandleSimOut(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigureSimStdout, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandleStackStart(string_list& rest) {
  auto base = detail::to_uint(detail::pop_front(rest));
  auto size = detail::to_uint(detail::pop_front(rest));
  if (base && size) {
    CHECKED_CALL(ConfigureStackInfo, *base, *size);
  }
  return static_cast<bool>(base) && static_cast<bool>(size);
}

bool SimulatorRPCChannel::HandleStallTrace(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_STALL, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandleStatsFile(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigureStatisticsFile, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandleSubsystemBase(string_list& rest) {
  auto base = detail::to_uint(detail::pop_front(rest));
  if (base) {
    CHECKED_CALL(ConfigureSubsystemBase, *base);
  }
  return static_cast<bool>(base);
}

bool SimulatorRPCChannel::HandleSymFile(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(AddSymbolFile, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandleTCM(string_list& rest) {
  CHECKED_CALL(ConfigureTimingMode, HEX_TIMING);
  return true;
}

bool SimulatorRPCChannel::HandleTCMHighAddr(string_list& rest) {
  // This option takes an argument, but (the option) is ignored.
  auto addr = detail::to_uint(detail::pop_front(rest));
  return static_cast<bool>(addr);
}

bool SimulatorRPCChannel::HandleTCMLowAddr(string_list& rest) {
  auto addr = detail::to_uint(detail::pop_front(rest));
  if (addr) {
    CHECKED_CALL(ConfigureTCM, *addr);
  }
  return static_cast<bool>(addr);
}

bool SimulatorRPCChannel::HandleTimeFilterNS(string_list& rest) {
  auto range = detail::to_range<uint64_t, detail::to_uint>(detail::pop_front(rest));
  if (range) {
    CHECKED_CALL(ConfigureTimeRangeFilter, range->first, HEX_NANOSEC, range->second, HEX_NANOSEC);
  }
  return static_cast<bool>(range);
}

bool SimulatorRPCChannel::HandleTiming(string_list& rest) {
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

bool SimulatorRPCChannel::HandleUArchTrace(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(SetTracing, HEX_TRACE_UARCH, file->c_str());
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandleUseFS(string_list& rest) {
  auto file = detail::pop_front(rest);
  if (file) {
    CHECKED_CALL(ConfigureARFilesystem, &(*file)[0]);
  }
  return static_cast<bool>(file);
}

bool SimulatorRPCChannel::HandleV2PTranslation(string_list& rest) {
  auto enable = detail::to_bool(detail::pop_front(rest));
  if (enable) {
    CHECKED_CALL(EnableVirtualToPhysicalTranslation, *enable);
  }
  return static_cast<bool>(enable);
}

bool SimulatorRPCChannel::HandleVerbose(string_list& rest) {
  auto mode = to_verbosemode(detail::pop_front(rest));
  if (mode) {
    sim_->VerboseMode(*mode);
  }
  return static_cast<bool>(mode);
}

bool SimulatorRPCChannel::should_parse_next(const string_list& rest) {
  if (auto str = detail::front(rest)) {
    return str->empty() || str->front() != '-';
  }
  return false;
}

std::optional<HEXAPI_Interval> SimulatorRPCChannel::to_interval(const detail::MaybeString& str) {
  if (!str) return std::nullopt;

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

  return detail::StringSwitch<std::optional<HEXAPI_Interval>>(*str)
      .Case("MILLISEC", HEX_MILLISEC)
      .Case("MICROSEC", HEX_MICROSEC)
      .Case("NANOSEC", HEX_NANOSEC)
      .Case("PICOSEC", HEX_PICOSEC)
      .Case("PCYCLE", HEX_PCYCLE)
      .Default(std::nullopt);
}

std::optional<HEXAPI_TimingMode> SimulatorRPCChannel::to_timingmode(
    const detail::MaybeString& str) {
  if (!str) return std::nullopt;

  if (auto val = detail::to_int(*str)) {
    switch (*val) {
      case HEX_NOTIMING:
      case HEX_TIMING_NODBC:
      case HEX_TIMING:
      case HEX_TIMING_COHERENCY:
        return static_cast<HEXAPI_TimingMode>(*val);
    }
  }

  return detail::StringSwitch<std::optional<HEXAPI_TimingMode>>(*str)
      .Case("NOTIMING", HEX_NOTIMING)
      .Case("TIMING_NODBC", HEX_TIMING_NODBC)
      .Case("TIMING", HEX_TIMING)
      .Case("TIMING_COHERENCY", HEX_TIMING_COHERENCY)
      .Default(std::nullopt);
}

std::optional<HEXAPI_VerboseMode> SimulatorRPCChannel::to_verbosemode(
    const detail::MaybeString& str) {
  if (!str) return std::nullopt;

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

  return detail::StringSwitch<std::optional<HEXAPI_VerboseMode>>(*str)
      .Case("SILENT", HEX_SILENT)
      .Case("QUIET", HEX_QUIET)
      .Case("NORMAL", HEX_NORMAL)
      .Case("VERBOSE", HEX_VERBOSE)
      .Case("REALLY_VERBOSE", HEX_REALLY_VERBOSE)
      .Default(std::nullopt);
}

std::optional<HEXAPI_Nullptr> SimulatorRPCChannel::to_nullptr(const detail::MaybeString& str) {
  if (!str) return std::nullopt;

  if (auto val = detail::to_int(*str)) {
    switch (*val) {
      case HEX_NULLPTR_IGNORE:
      case HEX_NULLPTR_WARN:
      case HEX_NULLPTR_FATAL:
      case HEX_NULLPTR_PCZERO:
        return static_cast<HEXAPI_Nullptr>(*val);
    }
  }

  return detail::StringSwitch<std::optional<HEXAPI_Nullptr>>(*str)
      .Case("IGNORE", HEX_NULLPTR_IGNORE)
      .Case("WARN", HEX_NULLPTR_WARN)
      .Case("FATAL", HEX_NULLPTR_FATAL)
      .Case("PCZERO", HEX_NULLPTR_PCZERO)
      .Default(std::nullopt);
}

TVM_REGISTER_GLOBAL("tvm.contrib.hexagon.create_hexagon_session")
    .set_body([](TVMArgs args, TVMRetValue* rv) {
      ICHECK(args.size() >= 4) << args.size() << " is less than 4";

      std::string session_name = args[0];
      int stack_size = args[1];
      std::string sim_args = args[2];
      auto channel = std::make_unique<SimulatorRPCChannel>(stack_size, sim_args);
      std::shared_ptr<RPCEndpoint> endpoint =
          RPCEndpoint::Create(std::move(channel), session_name, "", nullptr);
      std::shared_ptr<RPCSession> session = CreateClientSession(endpoint);
      *rv = CreateRPCSessionModule(session);
    });

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm
