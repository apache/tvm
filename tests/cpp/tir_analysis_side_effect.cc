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

#include <dmlc/logging.h>
#include <gtest/gtest.h>
#include <tvm/te/operation.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>

TEST(SimplePasses, SideEffect) {
  using namespace tvm;
  auto buf = tir::decl_buffer({16}, DataType::Float(32));
  auto i = tir::Var("i", DataType::Int(32));
  ICHECK(tir::SideEffect(tir::BufferLoad(buf, {i})) == tir::CallEffectKind::kReadState);
  ICHECK(tir::SideEffect(exp(tir::Cast(DataType::Float(32), i + 1))) == tir::CallEffectKind::kPure);
  ICHECK(tir::SideEffect(tir::Call(DataType::Handle(), tir::builtin::tvm_storage_sync(), {})) ==
         tir::CallEffectKind::kUpdateState);
}
