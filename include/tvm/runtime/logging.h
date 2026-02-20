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
 * \file tvm/runtime/logging.h
 * \brief logging utilities
 *
 * We use the following facilities from tvm/ffi/error.h for
 * error handling and checking:
 *
 *  - TVM_FFI_THROW(ErrorKind) << "msg";
 *  - TVM_FFI_CHECK(cond, ErrorKind) << "msg";
 *  - TVM_FFI_CHECK_EQ(x, y, ErrorKind) << "msg";
 *  - TVM_FFI_ICHECK(x) << "msg";  // InternalError
 *  - TVM_FFI_ICHECK_EQ(x, y) << "msg";
 *  - TVM_FFI_DCHECK(x) << "msg";  // Debug-only InternalError
 *
 * LOG(INFO), LOG(WARNING), LOG(ERROR) are kept for logging.
 * LOG(FATAL) is kept for completeness, it throws InternalError.
 */
#ifndef TVM_RUNTIME_LOGGING_H_
#define TVM_RUNTIME_LOGGING_H_

#include <tvm/ffi/error.h>
#include <tvm/runtime/base.h>

#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

/*!
 * \brief Macro helper to force a function not to be inlined.
 * It is only used in places that we know not inlining is good,
 * e.g. some logging functions.
 */
#if defined(_MSC_VER)
#define TVM_NO_INLINE __declspec(noinline)
#else
#define TVM_NO_INLINE __attribute__((noinline))
#endif

/*!
 * \brief Macro helper to force a function to be inlined.
 * It is only used in places that we know inline is important,
 * e.g. some template expansion cases.
 */
#ifdef _MSC_VER
#define TVM_ALWAYS_INLINE __forceinline
#else
#define TVM_ALWAYS_INLINE inline __attribute__((always_inline))
#endif

/*!
 * \brief Macro helper for exception throwing.
 */
#define TVM_THROW_EXCEPTION noexcept(false)

/*!
 * \brief Whether or not enable backtrace logging during a
 *        fatal error.
 *
 * \note TVM won't depend on LIBBACKTRACE or other exec_info
 *       library when this option is disabled.
 */
#ifndef TVM_LOG_STACK_TRACE
#define TVM_LOG_STACK_TRACE 1
#endif

/*!
 * \brief Whether or not use libbacktrace library
 *        for getting backtrace information
 */
#ifndef TVM_USE_LIBBACKTRACE
#define TVM_USE_LIBBACKTRACE 0
#endif

/*!
 * \brief Whether or not customize the logging output.
 *  If log customize is enabled, the user must implement
 *  tvm::runtime::detail::LogFatalImpl and tvm::runtime::detail::LogMessageImpl.
 */
#ifndef TVM_LOG_CUSTOMIZE
#define TVM_LOG_CUSTOMIZE 0
#endif

namespace tvm {
namespace runtime {

using ffi::EnvErrorAlreadySet;
using ffi::Error;

/*!
 * \brief Error type for errors from LOG(FATAL). This error
 * contains a backtrace of where it occurred.
 *
 * \note LOG(FATAL) always throws InternalError. For typed errors,
 * use TVM_FFI_THROW(ErrorKind) instead.
 */
class InternalError : public Error {
 public:
  InternalError(std::string file, int lineno, std::string message)
      : Error("InternalError", std::move(message), TVMFFIBacktrace(file.c_str(), lineno, "", 0)) {}
};

/*! \brief Internal implementation */
namespace detail {
// Provide support for customized logging.
#if TVM_LOG_CUSTOMIZE
/*!
 * \brief Custom implementations of LogFatal.
 *
 * \sa TVM_LOG_CUSTOMIZE
 */
[[noreturn]] TVM_DLL void LogFatalImpl(const std::string& file, int lineno,
                                       const std::string& message);

/*!
 * \brief Custom implementations of LogMessage.
 *
 * \sa TVM_LOG_CUSTOMIZE
 */
TVM_DLL void LogMessageImpl(const std::string& file, int lineno, int level,
                            const std::string& message);

/*!
 * \brief Class to accumulate an error message and throw it. Do not use
 * directly, instead use LOG(FATAL).
 */
class LogFatal {
 public:
  LogFatal(const std::string& file, int lineno) : file_(file), lineno_(lineno) {}
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4722)
#endif
  [[noreturn]] ~LogFatal() TVM_THROW_EXCEPTION { LogFatalImpl(file_, lineno_, stream_.str()); }
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  std::ostringstream& stream() { return stream_; }

 private:
  std::ostringstream stream_;
  std::string file_;
  int lineno_;
};

/*!
 * \brief Class to accumulate an log message. Do not use directly, instead use
 * LOG(INFO), LOG(WARNING), LOG(ERROR).
 */
class LogMessage {
 public:
  LogMessage(const std::string& file, int lineno, int level)
      : file_(file), lineno_(lineno), level_(level) {}
  ~LogMessage() { LogMessageImpl(file_, lineno_, level_, stream_.str()); }
  std::ostringstream& stream() { return stream_; }

 private:
  std::string file_;
  int lineno_;
  int level_;
  std::ostringstream stream_;
};

#else

/*!
 * \brief Class to accumulate an error message and throw it. Do not use
 * directly, instead use LOG(FATAL).
 * \note The `LogFatal` class is designed to be an empty class to reduce stack size usage.
 * To play this trick, we use the thread-local storage to store its internal data.
 */
class LogFatal {
 public:
  TVM_NO_INLINE LogFatal(const char* file, int lineno) { GetEntry().Init(file, lineno); }
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4722)
#endif
  [[noreturn]] ~LogFatal() TVM_THROW_EXCEPTION {
    GetEntry().Finalize();
    throw;
  }
#ifdef _MSC_VER
#pragma warning(pop)
#endif
  std::ostringstream& stream() { return GetEntry().stream_; }

 private:
  struct Entry {
    void Init(const char* file, int lineno) {
      this->stream_.str("");
      this->file_ = file;
      this->lineno_ = lineno;
    }
    [[noreturn]] TVM_NO_INLINE Error Finalize() TVM_THROW_EXCEPTION {
      InternalError error(file_, lineno_, stream_.str());
      throw error;
    }
    std::ostringstream stream_;
    std::string file_;
    int lineno_;
  };

  TVM_DLL TVM_NO_INLINE static Entry& GetEntry();
};

/*!
 * \brief Class to accumulate an log message. Do not use directly, instead use
 * LOG(INFO), LOG(WARNING), LOG(ERROR).
 */
class LogMessage {
 public:
  LogMessage(const std::string& file, int lineno, int level) {
    std::time_t t = std::time(nullptr);
    stream_ << "[" << std::put_time(std::localtime(&t), "%H:%M:%S") << "] " << file << ":" << lineno
            << level_strings_[level];
  }
  TVM_NO_INLINE ~LogMessage() { std::cerr << stream_.str() << std::endl; }
  std::ostringstream& stream() { return stream_; }

 private:
  std::ostringstream stream_;
  TVM_DLL static const char* level_strings_[];
};

#endif

// This class is used to explicitly ignore values in the conditional
// logging macros.  This avoids compiler warnings like "value computed
// is not used" and "statement has no effect".
class LogMessageVoidify {
 public:
  LogMessageVoidify() {}
  // This has to be an operator with a precedence lower than << but
  // higher than "?:". See its usage.
  void operator&(std::ostream&) {}
};

/*! \brief Captures the state of the \p TVM_LOG_DEBUG environment flag. */
class TvmLogDebugSettings {
 public:
  /*!
   * \brief Parses the \p TVM_LOG_DEBUG environment flag as per the specification given by
   * \p DebugLoggingEnabled and \p VerboseLoggingEnabled, and caches the result.
   */
  inline static const TvmLogDebugSettings& FromFlag() {
    // Parse and cache the verbosity level map.
    static const auto* settings =
        new TvmLogDebugSettings(TvmLogDebugSettings::ParseSpec(std::getenv("TVM_LOG_DEBUG")));
    return *settings;
  }

  /*!
   * \brief Parses \p opt_spec as per specification for \p TVM_LOG_DEBUG given by
   * \p DebugLoggingEnabled and \p VerboseLoggingEnabled. Throws if specification is ill-formed.
   */
  static TvmLogDebugSettings ParseSpec(const char* opt_spec);

  /*!
   * \brief Implements \p VerboseLoggingEnabled below w.r.t. the already parsed \p TVM_LOG_DEBUG
   * environment variable.
   */
  inline bool VerboseEnabled(const char* opt_filename, int level) const {
    if (opt_filename == nullptr || level < 0 || vlog_level_map_.empty()) {
      return false;
    }
    return VerboseEnabledImpl(opt_filename, level);
  }

  /*! \brief Returns true if \p DLOG statements should be executed. */
  bool dlog_enabled() const { return dlog_enabled_; }

 private:
  // Slow path for VerboseEnabled.
  bool VerboseEnabledImpl(const std::string& filename, int level) const;

  /*! \brief If true, DLOG statements are enabled. */
  bool dlog_enabled_ = false;
  /*!
   * \brief A map from canonicalized filenames to the maximum VLOG verbosity level for that file.
   * May also contain the 'wildcard' entry \p "DEFAULT" representing the level for all other files.
   */
  std::unordered_map<std::string, int> vlog_level_map_;
};

/*!
 * \brief Returns true if a DLOG statement is enabled by the \p TVM_LOG_DEBUG environment
 * variable. Requires:
 * \code
 *   TVM_LOG_DEBUG=1
 * \endcode
 * or a valid setting as described by \p VerboseLoggingEnabled below.
 */
inline bool DebugLoggingEnabled() {
  static int state = 0;
  if (state == 0) {
    state = TvmLogDebugSettings::FromFlag().dlog_enabled() ? 1 : -1;
  }
  return state == 1;
}

/*!
 * \brief Returns true if a VLOG statement in \p filename is enabled by the \p TVM_LOG_DEBUG
 * environment variable for logging at verbosity \p level. Levels should be non-negative.
 *
 * Filenames are canonicalized to be w.r.t. the src/ dir of the TVM tree. (VLOG's should not
 * appear under include/).
 *
 * To enable file \p ir/bar.cc for level 0 only set:
 * \code
 * TVM_LOG_DEBUG="ir/bar.cc=0"
 * \endcode
 *
 * To enable all files up to level 3 but disable \p ir/bar.cc set:
 * \code
 * TVM_LOG_DEBUG="DEFAULT=2,ir/bar.cc=-1"
 * \endcode
 *
 * Any of these settings will also enable DLOG statements.
 */
inline bool VerboseLoggingEnabled(const char* opt_filename, int level) {
  return TvmLogDebugSettings::FromFlag().VerboseEnabled(opt_filename, level);
}

/*!
 * \brief A stack of VLOG context messages.
 *
 * For use by \p VLOG_CONTEXT macro only.
 */
class VLogContext {
 public:
  void Push(std::stringstream* stream) { context_stack_.push_back(stream); }
  void Pop() {
    if (!context_stack_.empty()) {
      context_stack_.pop_back();
    }
  }

  std::string str() const;

 private:
  std::vector<std::stringstream*> context_stack_;
};

/*! \brief Get thread local \p VLogContext for tracking a stack of VLOG context messages. */
inline VLogContext* ThreadLocalVLogContext() {
  static thread_local VLogContext inst;
  return &inst;
}

/*!
 * \brief A RAII class to push/pos a VLOG context message onto the thread-local stack.
 *
 * For use by \p VLOG_CONTEXT macro only.
 */
class VLogContextEntry {
 public:
  VLogContextEntry() { ThreadLocalVLogContext()->Push(&sstream_); }
  ~VLogContextEntry() { ThreadLocalVLogContext()->Pop(); }
  std::ostream& stream() { return sstream_; }

 private:
  std::stringstream sstream_;
};

}  // namespace detail

#define TVM_LOG_LEVEL_DEBUG 0
#define TVM_LOG_LEVEL_INFO 1
#define TVM_LOG_LEVEL_WARNING 2
#define TVM_LOG_LEVEL_ERROR 3
#define TVM_LOG_LEVEL_FATAL 4
#define LOG(level) LOG_##level
#define LOG_DEBUG \
  ::tvm::runtime::detail::LogMessage(__FILE__, __LINE__, TVM_LOG_LEVEL_DEBUG).stream()
#define LOG_FATAL ::tvm::runtime::detail::LogFatal(__FILE__, __LINE__).stream()
#define LOG_INFO ::tvm::runtime::detail::LogMessage(__FILE__, __LINE__, TVM_LOG_LEVEL_INFO).stream()
#define LOG_ERROR \
  ::tvm::runtime::detail::LogMessage(__FILE__, __LINE__, TVM_LOG_LEVEL_ERROR).stream()
#define LOG_WARNING \
  ::tvm::runtime::detail::LogMessage(__FILE__, __LINE__, TVM_LOG_LEVEL_WARNING).stream()

#define LOG_IF(severity, condition) \
  !(condition) ? (void)0 : ::tvm::runtime::detail::LogMessageVoidify() & LOG(severity)

#if TVM_LOG_DEBUG

#define LOG_DFATAL LOG_FATAL
#define DFATAL FATAL
#define DLOG(severity) LOG_IF(severity, ::tvm::runtime::detail::DebugLoggingEnabled())
#define DLOG_IF(severity, condition) \
  LOG_IF(severity, ::tvm::runtime::detail::DebugLoggingEnabled() && (condition))

/*!
 * \brief If the \p TVM_LOG_DEBUG build flag is enabled, push a context message onto an internal
 * stack. All VLOG messages will include this stack in their prefix to help with debugging. E.g.:
 * \code
 *   VLOG_CONTEXT << "my context";
 *   VLOG(1) << "my log message";
 * \endcode
 * Thread safe. No-op with no execution overhead if the \p TVM_LOG_DEBUG build flag is not enabled.
 */
#define VLOG_CONTEXT                                    \
  ::tvm::runtime::detail::VLogContextEntry vlog_entry_; \
  vlog_entry_.stream()

#else

#define LOG_DFATAL LOG_ERROR
#define DFATAL ERROR
#define DLOG(severity) true ? (void)0 : ::tvm::runtime::detail::LogMessageVoidify() & LOG(severity)
#define DLOG_IF(severity, condition) \
  (true || !(condition)) ? (void)0 : ::tvm::runtime::detail::LogMessageVoidify() & LOG(severity)
#define VLOG_CONTEXT true ? (void)0 : ::tvm::runtime::detail::LogMessageVoidify() & LOG(INFO)

#endif

/*!
 * \brief If the \p TVM_LOG_DEBUG build flag is enabled, and the containing file has been enabled
 * at \p level or greater in the \p TVM_LOG_DEBUG environment variable, then log a message at
 * \p INFO severity.
 *
 * See \p VerboseLoggingEnabled for the format of the \p TVM_LOG_DEBUG environment variable.
 * Thread safe. No-op with no execution overhead if the \p TVM_LOG_DEBUG build flag is not enabled.
 * No-op with some execution overhead if the \p TVM_LOG_DEBUG build flag is enabled but the
 * containing file is not enabled.
 */
#define VLOG(level)                                                               \
  DLOG_IF(INFO, ::tvm::runtime::detail::VerboseLoggingEnabled(__FILE__, (level))) \
      << ::tvm::runtime::detail::ThreadLocalVLogContext()->str()

}  // namespace runtime
// Re-export error types
using runtime::Error;
using runtime::InternalError;

}  // namespace tvm
#endif  // TVM_RUNTIME_LOGGING_H_
