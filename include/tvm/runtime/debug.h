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
 * \file tvm/runtime/debug.h
 * \brief Helpers for debugging at runtime.
 */
#ifndef TVM_RUNTIME_DEBUG_H_
#define TVM_RUNTIME_DEBUG_H_

#include <tvm/runtime/container/adt.h>
#include <tvm/runtime/ndarray.h>

#include <ostream>
#include <string>

namespace tvm {
namespace runtime {

/*!
 * \brief Helpers to describe runtime objects in human-friendly form. For \p nd_arrays we show their
 * shapes and dtypes, but also their contents if 'small' and on the \p host_device (mostly so that
 * we can see dynamic shapes as they are computed). For \p adts we show the ADT fields. For
 * \p objects we dispatch to one of the above as appropriate.
 */
void AppendNDArray(std::ostream& os, const NDArray& nd_array, const DLDevice& host_device,
                   bool show_content = true);
void AppendADT(std::ostream& os, const ADT& adt, const DLDevice& host_device,
               bool show_content = true);
void AppendRuntimeObject(std::ostream& os, const ObjectRef& object, const DLDevice& host_device,
                         bool show_content = true);
std::string RuntimeObject2String(const ObjectRef& object, const DLDevice& host_device,
                                 bool show_content = true);

}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_DEBUG_H_
