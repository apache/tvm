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
 * \file tvm_runtime.h
 * \brief Pack all tvm runtime source files
 */
#include <sys/stat.h>

#include <fstream>

#define DMLC_USE_LOGGING_LIBRARY <tvm/runtime/logging.h>
#define TVM_USE_LIBBACKTRACE 0

#include "../src/runtime/c_runtime_api.cc"
#include "../src/runtime/cpu_device_api.cc"
#include "../src/runtime/dso_library.cc"
#include "../src/runtime/file_utils.cc"
#include "../src/runtime/graph_executor/graph_executor.cc"
#include "../src/runtime/library_module.cc"
#include "../src/runtime/logging.cc"
#include "../src/runtime/memory/memory_manager.cc"
#include "../src/runtime/module.cc"
#include "../src/runtime/ndarray.cc"
#include "../src/runtime/object.cc"
#include "../src/runtime/registry.cc"
#include "../src/runtime/system_library.cc"
#include "../src/runtime/thread_pool.cc"
#include "../src/runtime/threading_backend.cc"
#include "../src/runtime/workspace_pool.cc"

#ifdef TVM_OPENCL_RUNTIME
#include "../src/runtime/opencl/opencl_device_api.cc"
#include "../src/runtime/opencl/opencl_module.cc"
#include "../src/runtime/opencl/opencl_wrapper/opencl_wrapper.cc"
#include "../src/runtime/opencl/texture_pool.cc"
#include "../src/runtime/source_utils.cc"
#endif
