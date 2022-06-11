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
#include <tvm/parser/parser.h>
#include <tvm/relay/attrs/transform.h>
#include <tvm/relay/dataflow_matcher.h>
#include <tvm/relay/dataflow_pattern.h>
#include <tvm/relay/function.h>

namespace tvm {
namespace relay {
namespace {

TEST(WithFields, GlobalVar) {
  auto tensor_type = relay::TensorType({}, DataType::Bool());
  GlobalVar func_init("dummy_func", tensor_type, {});
  GlobalVar func_cp = WithFields(func_init);
  ICHECK(func_init->name_hint == func_cp->name_hint);
  ICHECK(func_init->span == func_cp->span);
  ICHECK(func_init->checked_type_ == func_cp->checked_type_);
}

TEST(WithFields, Constant) {
  int64_t out_channels = 64;
  Device dev{DLDeviceType::kDLCPU, 0};
  runtime::NDArray multiplier_nda = runtime::NDArray::Empty({out_channels}, DataType::Int(32), dev);
  Constant constant_init(multiplier_nda, {});
  Constant constant_cp = WithFields(constant_init);
  ICHECK(constant_init->checked_type_ == constant_cp->checked_type_);
  ICHECK_EQ(constant_init->data, constant_cp->data);
  ICHECK_EQ(constant_init->span, constant_cp->span);
}

}  // namespace
}  // namespace relay
}  // namespace tvm
