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
#include <nnvm/op.h>

#include <utility>

NNVM_REGISTER_OP(add)
    .describe("add two data together")
    .set_num_inputs(2)
    .set_attr("inplace_pair", std::make_pair(0, 0));

NNVM_REGISTER_OP(add).set_attr<std::string>("nick_name", "plus");

TEST(Op, GetAttr) {
  using namespace nnvm;
  auto add = Op::Get("add");
  auto nick = Op::GetAttr<std::string>("nick_name");

  CHECK_EQ(nick[add], "plus");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
