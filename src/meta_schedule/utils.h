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
#ifndef TVM_META_SCHEDULE_UTILS_H_
#define TVM_META_SCHEDULE_UTILS_H_

#include <dmlc/memory_io.h>
#include <tvm/arith/analyzer.h>
#include <tvm/meta_schedule/apply_history_best.h>
#include <tvm/meta_schedule/arg_info.h>
#include <tvm/meta_schedule/builder.h>
#include <tvm/meta_schedule/cost_model.h>
#include <tvm/meta_schedule/database.h>
#include <tvm/meta_schedule/extracted_task.h>
#include <tvm/meta_schedule/feature_extractor.h>
#include <tvm/meta_schedule/measure_callback.h>
#include <tvm/meta_schedule/profiler.h>
#include <tvm/meta_schedule/runner.h>
#include <tvm/meta_schedule/schedule_rule.h>
#include <tvm/meta_schedule/search_strategy.h>
#include <tvm/meta_schedule/space_generator.h>
#include <tvm/meta_schedule/task_scheduler.h>
#include <tvm/meta_schedule/tune_context.h>
#include <tvm/node/node.h>
#include <tvm/node/serialization.h>
#include <tvm/runtime/container/optional.h>
#include <tvm/support/parallel_for.h>
#include <tvm/tir/schedule/schedule.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <string>
#include <vector>

#include "../printer/text_printer.h"
#include "../support/array.h"
#include "../support/base64.h"
#include "../support/nd_int_set.h"
#include "../support/table_printer.h"
#include "../support/utils.h"
#include "../tir/schedule/primitive.h"
#include "../tir/schedule/utils.h"

#define TVM_PY_LOG(logging_level, logging_func)                          \
  ::tvm::meta_schedule::PyLogMessage(__FILE__, __LINE__, logging_func,   \
                                     PyLogMessage::Level::logging_level) \
      .stream()

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Class to accumulate an log message on the python side. Do not use directly, instead use
 * TVM_PY_LOG(DEBUG), TVM_PY_LOG(INFO), TVM_PY_LOG(WARNING), TVM_PY_ERROR(ERROR).
 */
class PyLogMessage {
 public:
  enum class Level : int32_t {
    DEBUG = 10,
    INFO = 20,
    WARNING = 30,
    ERROR = 40,
    // FATAL not included
  };

  PyLogMessage(const std::string& file, int lineno, PackedFunc logging_func, Level logging_level) {
    this->logging_func = logging_func;
    this->logging_level = logging_level;
  }
  TVM_NO_INLINE ~PyLogMessage() {
    if (this->logging_func.defined()) {
      logging_func(static_cast<int>(logging_level), stream_.str());
    } else {
      if (logging_level == Level::INFO) {
        LOG(INFO) << stream_.str();
      } else if (logging_level == Level::WARNING) {
        LOG(WARNING) << stream_.str();
      } else if (logging_level == Level::ERROR) {
        LOG(ERROR) << stream_.str();
      } else if (logging_level == Level::DEBUG) {
        DLOG(INFO) << stream_.str();
      } else {
        LOG(FATAL) << stream_.str();
      }
    }
  }
  std::ostringstream& stream() { return stream_; }

 private:
  std::ostringstream stream_;
  PackedFunc logging_func;
  Level logging_level;
};

/*! \brief The type of the random state */
using TRandState = support::LinearCongruentialEngine::TRandState;

/*!
 * \brief Get the base64 encoded result of a string.
 * \param str The string to encode.
 * \return The base64 encoded string.
 */
inline std::string Base64Encode(std::string str) {
  std::string result;
  dmlc::MemoryStringStream m_stream(&result);
  support::Base64OutStream b64stream(&m_stream);
  static_cast<dmlc::Stream*>(&b64stream)->Write(str);
  b64stream.Finish();
  return result;
}

/*!
 * \brief Get the base64 decoded result of a string.
 * \param str The string to decode.
 * \return The base64 decoded string.
 */
inline std::string Base64Decode(std::string str) {
  std::string result;
  dmlc::MemoryStringStream m_stream(&str);
  support::Base64InStream b64stream(&m_stream);
  b64stream.InitPosition();
  static_cast<dmlc::Stream*>(&b64stream)->Read(&result);
  return result;
}

/*!
 * \brief Parses a json string into a json object.
 * \param json_str The json string.
 * \return The json object
 */
ObjectRef JSONLoads(std::string json_str);

/*!
 * \brief Dumps a json object into a json string.
 * \param json_obj The json object.
 * \return The json string
 */
std::string JSONDumps(ObjectRef json_obj);

/*!
 * \brief Converts a structural hash code to string
 * \param hash_code The hash code
 * \return The string representation of the hash code
 */
inline String SHash2Str(Workload::THashCode hash_code) { return std::to_string(hash_code); }

/*!
 * \brief Converts an TVM object to the hex string representation of its structural hash.
 * \param obj The TVM object.
 * \return The hex string representation of the hash code.
 */
inline String SHash2Hex(const ObjectRef& obj) {
  std::ostringstream os;
  size_t hash_code = 0;
  if (obj.defined()) {
    hash_code = StructuralHash()(obj);
  }
  os << "0x" << std::setw(16) << std::setfill('0') << std::hex << hash_code;
  return os.str();
}

/*!
 * \brief Fork a random state into another, i.e. PRNG splitting.
 * The given random state is also mutated.
 * \param rand_state The random state to be forked
 * \return The forked random state
 */
inline support::LinearCongruentialEngine::TRandState ForkSeed(
    support::LinearCongruentialEngine::TRandState* rand_state) {
  return support::LinearCongruentialEngine(rand_state).ForkSeed();
}

/*!
 * \brief Fork a random state into another ones, i.e. PRNG splitting.
 *  The given random state is also mutated.
 * \param rand_state The random state to be forked
 * \param n The number of forks
 * \return The forked random states
 */
inline std::vector<support::LinearCongruentialEngine::TRandState> ForkSeed(
    support::LinearCongruentialEngine::TRandState* rand_state, int n) {
  std::vector<support::LinearCongruentialEngine::TRandState> results;
  results.reserve(n);
  for (int i = 0; i < n; ++i) {
    results.push_back(support::LinearCongruentialEngine(rand_state).ForkSeed());
  }
  return results;
}

/*!
 * \brief Get deep copy of an IRModule.
 * \param mod The IRModule to make a deep copy.
 * \return The deep copy of the IRModule.
 */
inline IRModule DeepCopyIRModule(IRModule mod) {
  return Downcast<IRModule>(LoadJSON(SaveJSON(mod)));
}

/*!
 * \brief Concatenate strings
 * \param strs The strings to concatenate
 * \param delim The delimiter
 * \return The concatenated string
 */
inline std::string Concat(const Array<String>& strs, const std::string& delim) {
  if (strs.empty()) {
    return "";
  }
  std::ostringstream os;
  os << strs[0];
  for (int i = 1, n = strs.size(); i < n; ++i) {
    os << delim << strs[i];
  }
  return os.str();
}

/*!
 * \brief Get the BlockRV from a block StmtSRef
 * \param sch The schedule
 * \param block_sref The block StmtSRef
 * \param global_var_name The global variable name
 * \return The BlockRV
 */
inline tir::BlockRV GetRVFromSRef(const tir::Schedule& sch, const tir::StmtSRef& block_sref,
                                  const String& global_var_name) {
  const tir::BlockNode* block = TVM_SREF_TO_BLOCK(block, block_sref);
  return sch->GetBlock(block->name_hint, global_var_name);
}

/*!
 * \brief A helper data structure that replays a trace and collects failure counts
 * for each postprocessor
 */
struct ThreadedTraceApply {
  /*! \brief Constructor */
  explicit ThreadedTraceApply(const Array<Postproc>& postprocs)
      : n_(postprocs.size()), items_(new Item[n_]) {
    for (int i = 0; i < n_; ++i) {
      items_[i].postproc = postprocs[i];
      items_[i].fail_counter = 0;
    }
  }

  /*! \brief Destructor */
  ~ThreadedTraceApply() { delete[] items_; }

  /*!
   * \brief Apply the trace and postprocessors to an IRModule
   * \param mod The IRModule to be applied
   * \param trace The trace to apply to the IRModule
   * \param rand_state The random seed
   * \return The schedule created, or NullOpt if any postprocessor fails
   */
  Optional<tir::Schedule> Apply(const IRModule& mod, const tir::Trace& trace,
                                TRandState* rand_state) {
    tir::Schedule sch =
        tir::Schedule::Traced(mod,
                              /*rand_state=*/ForkSeed(rand_state),
                              /*debug_mode=*/0,
                              /*error_render_level=*/tir::ScheduleErrorRenderLevel::kNone);

    trace->ApplyToSchedule(sch, /*remove_postproc=*/true);
    sch->EnterPostproc();

    for (int i = 0; i < n_; ++i) {
      Item& item = items_[i];
      try {
        if (!item.postproc->Apply(sch)) {
          ++item.fail_counter;
          return NullOpt;
        }
      } catch (const std::exception& e) {
        // Used in multi-thread, only output to screen but failure summary sent to logging
        LOG(WARNING) << "ThreadedTraceApply::Apply failed with error " << e.what();
        return NullOpt;
      }
    }
    return sch;
  }

  /*! \brief Returns a string summarizing the failures on each postprocessor */
  std::string SummarizeFailures() const {
    std::ostringstream os;
    for (int i = 0; i < n_; ++i) {
      const Item& item = items_[i];
      os << "Postproc #" << i << " [" << item.postproc  //
         << "]: " << item.fail_counter.load() << " failure(s)";
      if (i != n_ - 1) {
        os << "\n";
      }
    }
    return os.str();
  }

 private:
  /*! \brief A helper data structure that stores the fail count for each postprocessor. */
  struct Item {
    /*! \brief The postprocessor. */
    Postproc postproc{nullptr};
    /*! \brief The thread-safe postprocessor failure counter. */
    std::atomic<int> fail_counter{0};
  };

  /*! \brief The number of total postprocessors. */
  int n_;
  /*! \brief The pointer to the list of postprocessor items. */
  Item* items_;
};

/*!
 * \brief Get the number of cores in CPU
 * \param target The target
 * \return The number of cores.
 */
inline int GetTargetNumCores(const Target& target) {
  int num_cores = target->GetAttr<Integer>("num-cores").value_or(-1);
  if (num_cores == -1) {
    static const auto* f_cpu_count = runtime::Registry::Get("meta_schedule.cpu_count");
    ICHECK(f_cpu_count)
        << "ValueError: Cannot find the packed function \"meta_schedule._cpu_count\"";
    num_cores = (*f_cpu_count)(false);
    LOG(FATAL)
        << "Target does not have attribute \"num-cores\", physical core number must be "
           "defined! For example, on the local machine, the target must be \"llvm -num-cores "
        << num_cores << "\"";
  }
  return num_cores;
}

/*!
 * \brief Get the median of the running time from RunnerResult in millisecond
 * \param results The results from RunnerResult
 * \return The median of the running time in millisecond
 */
inline double GetRunMsMedian(const RunnerResult& runner_result) {
  Array<FloatImm> run_secs = runner_result->run_secs.value();
  ICHECK(!run_secs.empty());
  std::vector<double> v;
  v.reserve(run_secs.size());
  std::transform(run_secs.begin(), run_secs.end(), std::back_inserter(v),
                 [](const FloatImm& f) -> double { return f->value; });
  std::sort(v.begin(), v.end());
  int n = v.size();
  if (n % 2 == 0) {
    return (v[n / 2] + v[n / 2 + 1]) * 0.5 * 1000.0;
  } else {
    return v[n / 2] * 1000.0;
  }
}

/*!
 * \brief Convert the given object to an array of floating point numbers
 * \param obj The object to be converted
 * \return The array of floating point numbers
 */
inline Array<FloatImm> AsFloatArray(const ObjectRef& obj) {
  const ArrayNode* arr = obj.as<ArrayNode>();
  ICHECK(arr) << "TypeError: Expect an array, but gets: " << obj->GetTypeKey();
  Array<FloatImm> results;
  results.reserve(arr->size());
  for (const ObjectRef& elem : *arr) {
    if (const auto* int_imm = elem.as<IntImmNode>()) {
      results.push_back(FloatImm(DataType::Float(32), int_imm->value));
    } else if (const auto* float_imm = elem.as<FloatImmNode>()) {
      results.push_back(FloatImm(DataType::Float(32), float_imm->value));
    } else {
      LOG(FATAL) << "TypeError: Expect an array of float or int, but gets: " << elem->GetTypeKey();
    }
  }
  return results;
}

/*!
 * \brief Convert the given object to an array of integers
 * \param obj The object to be converted
 * \return The array of integers
 */
inline Array<Integer> AsIntArray(const ObjectRef& obj) {
  const ArrayNode* arr = obj.as<ArrayNode>();
  ICHECK(arr) << "TypeError: Expect an array, but gets: " << obj->GetTypeKey();
  Array<Integer> results;
  results.reserve(arr->size());
  for (const ObjectRef& elem : *arr) {
    if (const auto* int_imm = elem.as<IntImmNode>()) {
      results.push_back(Integer(int_imm->value));
    } else {
      LOG(FATAL) << "TypeError: Expect an array of integers, but gets: " << elem->GetTypeKey();
    }
  }
  return results;
}

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_UTILS_H_
