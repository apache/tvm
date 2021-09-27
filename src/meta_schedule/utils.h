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
#include <tvm/meta_schedule/arg_info.h>
#include <tvm/meta_schedule/builder.h>
#include <tvm/meta_schedule/database.h>
#include <tvm/meta_schedule/runner.h>
#include <tvm/meta_schedule/search_strategy.h>
#include <tvm/meta_schedule/space_generator.h>
#include <tvm/meta_schedule/tune_context.h>
#include <tvm/node/node.h>
#include <tvm/node/serialization.h>
#include <tvm/support/parallel_for.h>
#include <tvm/tir/schedule/schedule.h>

#include <string>
#include <vector>

#include <vector>

#include "../printer/text_printer.h"
#include "../support/array.h"
#include "../support/base64.h"
#include "../tir/schedule/primitive.h"

namespace tvm {
namespace meta_schedule {

/*!
 * \brief Read lines from a json file.
 * \param path The path to the json file.
 * \param allow_missing Whether to create new file when the given path is not found.
 * \return An array containing lines read from the json file.
 */
inline Array<String> JSONFileReadLines(const String& path, bool allow_missing) {
  std::ifstream is(path);
  if (is.good()) {
    Array<String> results;
    for (std::string str; std::getline(is, str);) {
      results.push_back(str);
    }
    return results;
  }
  CHECK(allow_missing) << "ValueError: File doesn't exist: " << path;
  std::ofstream os(path);
  CHECK(os.good()) << "ValueError: Cannot create new file: " << path;
  return {};
}

/*!
 * \brief Append a line to a json file.
 * \param path The path to the json file.
 * \param line The line to append.
 */
inline void JSONFileAppendLine(const String& path, const std::string& line) {
  std::ofstream os(path, std::ofstream::app);
  CHECK(os.good()) << "ValueError: Cannot open the file to write: " << path;
  os << line << std::endl;
}

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
 * \brief Parse lines of json string into a json object.
 * \param lines The lines of json string.
 * \return Array of json objects parsed.
 * \note The function calls the python-side json parser in runtime registry.
 */
inline Array<ObjectRef> JSONStr2Obj(const Array<String>& lines) {
  static const runtime::PackedFunc* f_to_obj =
      runtime::Registry::Get("meta_schedule.batch_json_str2obj");
  ICHECK(f_to_obj) << "IndexError: Cannot find the packed function "
                      "`meta_schedule.batch_json_str2obj` in the global registry";
  return (*f_to_obj)(lines);
}

/*!
 * \brief Serialize a json object into a json string.
 * \param json_obj The json object to serialize.
 * \return A string containing the serialized json object.
 * \note The function calls the python-side json obj serializer in runtime registry.
 */
inline String JSONObj2Str(const ObjectRef& json_obj) {
  static const runtime::PackedFunc* f_to_str = runtime::Registry::Get("meta_schedule.json_obj2str");
  ICHECK(f_to_str) << "IndexError: Cannot find the packed function "
                      "`meta_schedule.json_obj2str` in the global registry";
  return (*f_to_str)(json_obj);
}

/*!
 * \brief Converts a structural hash code to string
 * \param hash_code The hash code
 * \return The string representation of the hash code
 */
inline String SHash2Str(Workload::THashCode hash_code) { return std::to_string(hash_code); }

/*!
 * \brief Find the entry function of the given IRModule, i.e, functions marked by
 * `tir::attr::kIsEntryFunc`, whose name is `main` or being the only PrimeFunc.
 * \param mod The IRModule to find the entry function.
 * \return The entry function.
 */
inline tir::PrimFunc FindEntryFunc(const IRModule& mod) {
  // Priority 1: PrimFunc marked as `tir::attr::kIsEntryFunc`
  int num_prim_func = 0;
  const tir::PrimFuncNode* main_func = nullptr;
  const tir::PrimFuncNode* last_func = nullptr;
  for (const auto& kv : mod->functions) {
    GlobalVar gv = kv.first;
    BaseFunc base_func = kv.second;
    if (const auto* func = base_func.as<tir::PrimFuncNode>()) {
      last_func = func;
      if (func->HasNonzeroAttr(tir::attr::kIsEntryFunc)) {
        return GetRef<tir::PrimFunc>(func);
      }
      if (gv->name_hint == "main") {
        main_func = func;
      }
      ++num_prim_func;
    }
  }
  // Priority 2: PrimFunc whose name is `main`
  if (main_func != nullptr) {
    return GetRef<tir::PrimFunc>(main_func);
  }
  // Priority 3: The only PrimFunc in the IRModule
  if (num_prim_func == 0) {
    LOG(FATAL) << "ValueError: Cannot find any PrimFunc in the given IRModule: "
               << tir::AsTVMScript(mod);
  }
  if (num_prim_func > 1) {
    LOG(FATAL) << "ValueError: Multiple PrimFuncs exist in the IRModule, but none of them are "
                  "annotated with `kIsEntryFunc`, i.e. `tir.is_entry_func`"
               << tir::AsTVMScript(mod);
  }
  return GetRef<tir::PrimFunc>(last_func);
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
 * The given random state is also mutated.
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

}  // namespace meta_schedule
}  // namespace tvm

#endif  // TVM_META_SCHEDULE_UTILS_H_
