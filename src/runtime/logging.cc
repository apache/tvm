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
#include <tvm/runtime/logging.h>

#include <string>

#if (TVM_LOG_CUSTOMIZE == 0)
namespace tvm {
namespace runtime {
namespace detail {

const char* ::tvm::runtime::detail::LogMessage::level_strings_[] = {
    ": Debug: ",    // TVM_LOG_LEVEL_DEBUG
    ": ",           // TVM_LOG_LEVEL_INFO
    ": Warning: ",  // TVM_LOG_LEVEL_WARNING
    ": Error: ",    // TVM_LOG_LEVEL_ERROR
};

namespace {
constexpr const char* kSrcPrefix = "/src/";
// Note: Better would be std::char_traits<const char>::length(kSrcPrefix) but it is not
// a constexpr on all compilation targets.
constexpr const size_t kSrcPrefixLength = 5;
constexpr const char* kDefaultKeyword = "DEFAULT";
}  // namespace

namespace {
/*! \brief Convert __FILE__ to a vlog_level_map_ key, which strips any prefix ending iwth src/ */
std::string FileToVLogMapKey(const std::string& filename) {
  // Canonicalize the filename.
  // TODO(mbs): Not Windows friendly.
  size_t last_src = filename.rfind(kSrcPrefix, std::string::npos, kSrcPrefixLength);
  if (last_src == std::string::npos) {
    std::string no_slash_src{kSrcPrefix + 1};
    if (filename.substr(0, no_slash_src.size()) == no_slash_src) {
      return filename.substr(no_slash_src.size());
    }
  }
  // Strip anything before the /src/ prefix, on the assumption that will yield the
  // TVM project relative filename. If no such prefix fallback to filename without
  // canonicalization.
  return (last_src == std::string::npos) ? filename : filename.substr(last_src + kSrcPrefixLength);
}
}  // namespace

/* static */
TvmLogDebugSettings TvmLogDebugSettings::ParseSpec(const char* opt_spec) {
  TvmLogDebugSettings settings;
  if (opt_spec == nullptr) {
    // DLOG and VLOG disabled.
    return settings;
  }
  std::string spec(opt_spec);
  if (spec.empty() || spec == "0") {
    // DLOG and VLOG disabled.
    return settings;
  }
  settings.dlog_enabled_ = true;
  if (spec == "1") {
    // Legacy specification for enabling just DLOG.
    return settings;
  }
  std::istringstream spec_stream(spec);
  auto tell_pos = [&](const std::string& last_read) {
    int pos = spec_stream.tellg();
    if (pos == -1) {
      LOG(INFO) << "override pos: " << last_read;
      // when pos == -1, failbit was set due to std::getline reaching EOF without seeing delimiter.
      pos = spec.size() - last_read.size();
    }
    return pos;
  };
  while (spec_stream) {
    std::string name;
    if (!std::getline(spec_stream, name, '=')) {
      // Reached end.
      break;
    }
    if (name.empty()) {
      TVM_FFI_THROW(InternalError)
          << "TVM_LOG_DEBUG ill-formed at position " << tell_pos(name) << ": empty filename";
    }

    name = FileToVLogMapKey(name);

    std::string level;
    if (!std::getline(spec_stream, level, ',')) {
      TVM_FFI_THROW(InternalError) << "TVM_LOG_DEBUG ill-formed at position " << tell_pos(level)
                                   << ": expecting \"=<level>\" after \"" << name << "\"";
      return settings;
    }
    if (level.empty()) {
      TVM_FFI_THROW(InternalError) << "TVM_LOG_DEBUG ill-formed at position " << tell_pos(level)
                                   << ": empty level after \"" << name << "\"";
      return settings;
    }
    // Parse level, default to 0 if ill-formed which we don't detect.
    char* end_of_level = nullptr;
    int level_val = static_cast<int>(strtol(level.c_str(), &end_of_level, 10));
    if (end_of_level != level.c_str() + level.size()) {
      TVM_FFI_THROW(InternalError) << "TVM_LOG_DEBUG ill-formed at position " << tell_pos(level)
                                   << ": invalid level: \"" << level << "\"";
      return settings;
    }
    LOG(INFO) << "TVM_LOG_DEBUG enables VLOG statements in '" << name << "' up to level " << level;
    settings.vlog_level_map_.emplace(name, level_val);
  }
  return settings;
}

bool TvmLogDebugSettings::VerboseEnabledImpl(const std::string& filename, int level) const {
  // Check for exact match.
  auto itr = vlog_level_map_.find(FileToVLogMapKey(filename));
  if (itr != vlog_level_map_.end()) {
    return level <= itr->second;
  }
  // Check for default.
  itr = vlog_level_map_.find(kDefaultKeyword);
  if (itr != vlog_level_map_.end()) {
    return level <= itr->second;
  }
  return false;
}

LogFatal::Entry& LogFatal::GetEntry() {
  static thread_local LogFatal::Entry result;
  return result;
}

std::string VLogContext::str() const {
  std::stringstream result;
  for (const auto* entry : context_stack_) {
    TVM_FFI_ICHECK_NOTNULL(entry);
    result << entry->str();
    result << ": ";
  }
  return result.str();
}

}  // namespace detail
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_LOG_CUSTOMIZE
