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

#include <gtest/gtest.h>
#include <tvm/runtime/logging.h>
#include <tvm/te/operation.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>

TEST(SimplePasses, SideEffect) {
  using namespace tvm;
  auto buf = tirx::decl_buffer({16}, DataType::Float(32));
  auto i = tirx::Var("i", DataType::Int(32));
  TVM_FFI_ICHECK(tirx::SideEffect(tirx::BufferLoad(buf, {i})) == tirx::CallEffectKind::kReadState);
  TVM_FFI_ICHECK(tirx::SideEffect(exp(tirx::Cast(DataType::Float(32), i + 1))) ==
                 tirx::CallEffectKind::kPure);
  TVM_FFI_ICHECK(tirx::SideEffect(tirx::Call(DataType::Handle(), tirx::builtin::tvm_storage_sync(),
                                             {})) == tirx::CallEffectKind::kUpdateState);
}
