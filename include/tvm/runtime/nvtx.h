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
#ifndef TVM_RUNTIME_NVTX_H_
#define TVM_RUNTIME_NVTX_H_

#include <string>
namespace tvm {
namespace runtime {

/*!
 * \brief A class to create a NVTX range. No-op if TVM is not built against NVTX.
 */
class NVTXScopedRange {
 public:
  /*! \brief Enter an NVTX scoped range */
  explicit NVTXScopedRange(const char* name);
  /*! \brief Enter an NVTX scoped range */
  explicit NVTXScopedRange(const std::string& name) : NVTXScopedRange(name.c_str()) {}
  /*! \brief Exist an NVTX scoped range */
  ~NVTXScopedRange();
  NVTXScopedRange(const NVTXScopedRange& other) = delete;
  NVTXScopedRange(NVTXScopedRange&& other) = delete;
  NVTXScopedRange& operator=(const NVTXScopedRange& other) = delete;
  NVTXScopedRange& operator=(NVTXScopedRange&& other) = delete;
};

#ifdef _MSC_VER
#define TVM_NVTX_FUNC_SCOPE() NVTXScopedRange _nvtx_func_scope_(__FUNCSIG__);
#else
#define TVM_NVTX_FUNC_SCOPE() NVTXScopedRange _nvtx_func_scope_(__PRETTY_FUNCTION__);
#endif

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_NVTX_H_
