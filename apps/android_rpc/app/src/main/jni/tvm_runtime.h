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

/* Enable custom logging - this will cause TVM to pass every log message
 * through CustomLogMessage instead of LogMessage. By enabling this, we must
 * implement dmlc::CustomLogMessage::Log. We use this to pass TVM log
 * messages to Android logcat.
 */
#define DMLC_LOG_CUSTOMIZE 1

/* Ensure that fatal errors are passed to the logger before throwing
 * in LogMessageFatal
 */
#define DMLC_LOG_BEFORE_THROW 1

#include "../src/runtime/c_runtime_api.cc"
#include "../src/runtime/cpu_device_api.cc"
#include "../src/runtime/dso_library.cc"
#include "../src/runtime/file_util.cc"
#include "../src/runtime/graph/graph_runtime.cc"
#include "../src/runtime/library_module.cc"
#include "../src/runtime/module.cc"
#include "../src/runtime/ndarray.cc"
#include "../src/runtime/object.cc"
#include "../src/runtime/registry.cc"
#include "../src/runtime/rpc/rpc_channel.cc"
#include "../src/runtime/rpc/rpc_endpoint.cc"
#include "../src/runtime/rpc/rpc_event_impl.cc"
#include "../src/runtime/rpc/rpc_local_session.cc"
#include "../src/runtime/rpc/rpc_module.cc"
#include "../src/runtime/rpc/rpc_server_env.cc"
#include "../src/runtime/rpc/rpc_session.cc"
#include "../src/runtime/rpc/rpc_socket_impl.cc"
#include "../src/runtime/system_library.cc"
#include "../src/runtime/thread_pool.cc"
#include "../src/runtime/threading_backend.cc"
#include "../src/runtime/workspace_pool.cc"

#ifdef TVM_OPENCL_RUNTIME
#include "../src/runtime/opencl/opencl_device_api.cc"
#include "../src/runtime/opencl/opencl_module.cc"
#endif

#ifdef TVM_VULKAN_RUNTIME
#include "../src/runtime/vulkan/vulkan.cc"
#endif

#ifdef USE_SORT
#include "../src/runtime/contrib/sort/sort.cc"
#endif

#include <android/log.h>

void dmlc::CustomLogMessage::Log(const std::string& msg) {
  // This is called for every message logged by TVM.
  // We pass the message to logcat.
  __android_log_write(ANDROID_LOG_DEBUG, "TVM_RUNTIME", msg.c_str());
}
