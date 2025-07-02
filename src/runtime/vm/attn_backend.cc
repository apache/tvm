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

/*! \file src/runtime/vm/attn_backend.cc */

#include "attn_backend.h"

namespace tvm {
namespace runtime {
namespace vm {

std::unique_ptr<PagedPrefillFunc> ConvertPagedPrefillFunc(Array<ObjectRef> args,
                                                          AttnKind attn_kind) {
  if (args.empty()) {
    return nullptr;
  }
  String backend_name = Downcast<String>(args[0]);
  if (backend_name == "tir") {
    CHECK_EQ(args.size(), 2);
    ffi::Function attn_func = Downcast<ffi::Function>(args[1]);
    return std::make_unique<TIRPagedPrefillFunc>(std::move(attn_func), attn_kind);
  }
  if (backend_name == "flashinfer") {
    CHECK_EQ(args.size(), 3);
    ffi::Function attn_func = Downcast<ffi::Function>(args[1]);
    ffi::Function plan_func = Downcast<ffi::Function>(args[2]);
    return std::make_unique<FlashInferPagedPrefillFunc>(std::move(attn_func), std::move(plan_func),
                                                        attn_kind);
  }
  LOG(FATAL) << "Cannot reach here";
  throw;
}

std::unique_ptr<RaggedPrefillFunc> ConvertRaggedPrefillFunc(Array<ObjectRef> args,
                                                            AttnKind attn_kind) {
  if (args.empty()) {
    return nullptr;
  }
  String backend_name = Downcast<String>(args[0]);
  if (backend_name == "tir") {
    CHECK_EQ(args.size(), 2);
    ffi::Function attn_func = Downcast<ffi::Function>(args[1]);
    return std::make_unique<TIRRaggedPrefillFunc>(std::move(attn_func), attn_kind);
  }
  if (backend_name == "flashinfer") {
    CHECK_EQ(args.size(), 3);
    ffi::Function attn_func = Downcast<ffi::Function>(args[1]);
    ffi::Function plan_func = Downcast<ffi::Function>(args[2]);
    return std::make_unique<FlashInferRaggedPrefillFunc>(std::move(attn_func), std::move(plan_func),
                                                         attn_kind);
  }
  LOG(FATAL) << "Cannot reach here";
  throw;
}

std::unique_ptr<PagedDecodeFunc> ConvertPagedDecodeFunc(Array<ObjectRef> args, AttnKind attn_kind) {
  if (args.empty()) {
    return nullptr;
  }
  String backend_name = Downcast<String>(args[0]);
  if (backend_name == "tir") {
    CHECK_EQ(args.size(), 2);
    ffi::Function attn_func = Downcast<ffi::Function>(args[1]);
    return std::make_unique<TIRPagedDecodeFunc>(std::move(attn_func), attn_kind);
  }
  if (backend_name == "flashinfer") {
    CHECK_EQ(args.size(), 3);
    ffi::Function attn_func = Downcast<ffi::Function>(args[1]);
    ffi::Function plan_func = Downcast<ffi::Function>(args[2]);
    return std::make_unique<FlashInferPagedDecodeFunc>(std::move(attn_func), std::move(plan_func),
                                                       attn_kind);
  }
  LOG(FATAL) << "Cannot reach here";
  throw;
}

std::unique_ptr<PagedPrefillTreeMaskFunc> ConvertPagedPrefillTreeMaskFunc(Array<ObjectRef> args,
                                                                          AttnKind attn_kind) {
  if (args.empty()) {
    return nullptr;
  }
  String backend_name = Downcast<String>(args[0]);
  if (backend_name == "tir") {
    CHECK_EQ(args.size(), 2);
    ffi::Function attn_func = Downcast<ffi::Function>(args[1]);
    return std::make_unique<TIRPagedPrefillTreeMaskFunc>(std::move(attn_func), attn_kind);
  }
  LOG(FATAL) << "Cannot reach here";
  throw;
}

std::unique_ptr<RaggedPrefillTreeMaskFunc> ConvertRaggedPrefillTreeMaskFunc(Array<ObjectRef> args,
                                                                            AttnKind attn_kind) {
  if (args.empty()) {
    return nullptr;
  }
  String backend_name = Downcast<String>(args[0]);
  if (backend_name == "tir") {
    CHECK_EQ(args.size(), 2);
    ffi::Function attn_func = Downcast<ffi::Function>(args[1]);
    return std::make_unique<TIRRaggedPrefillTreeMaskFunc>(std::move(attn_func), attn_kind);
  }
  LOG(FATAL) << "Cannot reach here";
  throw;
}

}  // namespace vm
}  // namespace runtime
}  // namespace tvm
