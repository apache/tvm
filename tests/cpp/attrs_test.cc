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
#include <tvm/attrs.h>
#include <tvm/ir.h>

namespace tvm {
namespace test {
// test example usage docs
struct TestAttrs : public AttrsNode<TestAttrs> {
  int axis;
  std::string name;
  Expr expr;
  double learning_rate;

  TVM_DECLARE_ATTRS(TestAttrs, "attrs.cpptest.TestAttrs") {
    TVM_ATTR_FIELD(axis)
        .set_default(10)
        .set_lower_bound(1)
        .set_upper_bound(10)
        .describe("axis field");
    TVM_ATTR_FIELD(name)
        .describe("name of the field");
    TVM_ATTR_FIELD(expr)
        .describe("expression field")
        .set_default(make_const(Int(32), 1));
    TVM_ATTR_FIELD(learning_rate)
        .describe("learning_rate")
        .set_default(0.1);
  }
};
}
}

TEST(Attrs, Basic) {
  using namespace tvm;
  using namespace tvm::test;
  std::shared_ptr<TestAttrs> n = std::make_shared<TestAttrs>();
  try {
    n->InitBySeq("axis", 10);
    LOG(FATAL) << "bad";
  } catch (const tvm::AttrError& e) {
  }
  try {
    n->InitBySeq("axis", 12, "name", "111");
    LOG(FATAL) << "bad";
  } catch (const tvm::AttrError& e) {
  }

  try {
    n->InitBySeq("axisx", 12, "name", "111");
    LOG(FATAL) << "bad";
  } catch (const tvm::AttrError& e) {
    std::string what = e.what();
    CHECK(what.find("expr : Expr, default=1") != std::string::npos);
    CHECK(what.find("axisx") != std::string::npos);
  }
  n->InitBySeq("learning_rate", Expr(1), "expr", 128, "name", "xx");
  CHECK_EQ(n->learning_rate, 1.0);

  n->InitBySeq("name", "xxx", "expr", 128);
  CHECK_EQ(n->name, "xxx");
  CHECK_EQ(n->axis, 10);
  CHECK_EQ(n->expr.as<tvm::ir::IntImm>()->value, 128);
  // Check docstring
  std::ostringstream os;
  n->PrintDocString(os);
  LOG(INFO) << "docstring\n"<< os.str();
  CHECK(os.str().find("expr : Expr, default=1") != std::string::npos);
}


int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  testing::FLAGS_gtest_death_test_style = "threadsafe";
  return RUN_ALL_TESTS();
}
