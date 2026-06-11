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
 * \file src/runtime/contrib/clml/clml_memory_planner.h
 * \brief CLML memory planner header
 */
#ifndef TVM_RUNTIME_CONTRIB_CLML_CLML_MEMORY_PLANNER_H_
#define TVM_RUNTIME_CONTRIB_CLML_CLML_MEMORY_PLANNER_H_

#include "clml_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

void FreeMemory(CachedLayer* layer, int nid);

void ReleaseDDRMemory(cl_mem memptr);

size_t RequestOnChipMemory(CachedLayer* layer, size_t size);

cl_mem RequestDDRMemory(CachedLayer* layer, size_t size);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_CLML_CLML_MEMORY_PLANNER_H_
