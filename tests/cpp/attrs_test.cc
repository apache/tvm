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
#include <tvm/ir/attrs.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>

namespace tvm {
namespace test {
// test example usage docs
struct TestAttrs : public AttrsNode<TestAttrs> {
  int axis;
  std::string name;
  PrimExpr expr;
  double learning_rate;

  TVM_DECLARE_ATTRS(TestAttrs, "attrs.cpptest.TestAttrs") {
    TVM_ATTR_FIELD(axis).set_default(10).set_lower_bound(1).set_upper_bound(10).describe(
        "axis field");
    TVM_ATTR_FIELD(name).describe("name of the field");
    TVM_ATTR_FIELD(expr)
        .describe("expression field")
        .set_default(tir::make_const(DataType::Int(32), 1));
    TVM_ATTR_FIELD(learning_rate).describe("learning_rate").set_default(0.1);
  }
};
}  // namespace test
}  // namespace tvm

TEST(Attrs, Basic) {
  using namespace tvm;
  using namespace tvm::test;
  ObjectPtr<TestAttrs> n = make_object<TestAttrs>();
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
    ICHECK(what.find("expr : PrimExpr, default=1") != std::string::npos);
    ICHECK(what.find("axisx") != std::string::npos);
  }
  n->InitBySeq("learning_rate", PrimExpr(1), "expr", 128, "name", "xx");
  ICHECK_EQ(n->learning_rate, 1.0);

  n->InitBySeq("name", "xxx", "expr", 128);
  ICHECK_EQ(n->name, "xxx");
  ICHECK_EQ(n->axis, 10);
  ICHECK_EQ(n->expr.as<tvm::tir::IntImmNode>()->value, 128);
  // Check docstring
  std::ostringstream os;
  n->PrintDocString(os);
  LOG(INFO) << "docstring\n" << os.str();
  ICHECK(os.str().find("expr : PrimExpr, default=1") != std::string::npos);
}
