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
#include <tvm/ir/global_var_supply.h>
#include <tvm/ir/module.h>
#include <tvm/ir/name_supply.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/function.h>

using namespace tvm;

NameSupply preambleNameSupply() {
  NameSupply name_supply("prefix");
  name_supply->FreshName("test");
  return name_supply;
}

TEST(NameSupply, FreshName) {
  NameSupply name_supply = preambleNameSupply();
  String fresh = name_supply->FreshName("test");

  EXPECT_EQ(fresh.compare("prefix_test_1"), 0);
}

TEST(NameSupply, FreshNameNoConflict) {
  NameSupply name_supply = preambleNameSupply();
  String fresh = name_supply->FreshName("name_2");
  EXPECT_EQ(fresh.compare("prefix_name_2"), 0);

  fresh = name_supply->FreshName("name");
  EXPECT_EQ(fresh.compare("prefix_name"), 0);

  fresh = name_supply->FreshName("name");
  EXPECT_EQ(fresh.compare("prefix_name_1"), 0);

  fresh = name_supply->FreshName("name");
  EXPECT_EQ(fresh.compare("prefix_name_3"), 0);
}

TEST(NameSupply, ContainsName) {
  NameSupply name_supply = preambleNameSupply();

  EXPECT_TRUE(name_supply->ContainsName("test"));
  EXPECT_FALSE(name_supply->ContainsName("test_1"));
}

TEST(NameSupply, ReserveName) {
  NameSupply name_supply = preambleNameSupply();
  name_supply->ReserveName("otherTest", false);

  EXPECT_TRUE(name_supply->ContainsName("otherTest", false));
  EXPECT_FALSE(name_supply->ContainsName("otherTest"));

  name_supply->ReserveName("otherTest");
  EXPECT_TRUE(name_supply->ContainsName("prefix_otherTest", false));
  EXPECT_TRUE(name_supply->ContainsName("otherTest"));
}

GlobalVarSupply preambleVarSupply() {
  GlobalVarSupply global_var_supply;
  global_var_supply->FreshGlobal("test");
  return global_var_supply;
}

TEST(GlobalVarSupply, FreshGlobal) {
  GlobalVarSupply global_var_supply = preambleVarSupply();
  GlobalVar first_var = global_var_supply->FreshGlobal("test");
  GlobalVar second_var = global_var_supply->FreshGlobal("test");

  EXPECT_FALSE(tvm::StructuralEqual()(first_var, second_var));
  EXPECT_EQ(first_var->name_hint.compare("test_1"), 0);
  EXPECT_EQ(second_var->name_hint.compare("test_2"), 0);
}

TEST(GlobalVarSupply, UniqueGlobalFor) {
  GlobalVarSupply global_var_supply = preambleVarSupply();
  GlobalVar first_var = global_var_supply->UniqueGlobalFor("someName");
  GlobalVar second_var = global_var_supply->UniqueGlobalFor("someName");

  EXPECT_TRUE(tvm::StructuralEqual()(first_var, second_var));
  EXPECT_EQ(first_var->name_hint.compare("someName"), 0);
  EXPECT_EQ(second_var->name_hint.compare("someName"), 0);
}

TEST(GlobalVarSupply, ReserveGlobal) {
  GlobalVarSupply global_var_supply = preambleVarSupply();
  GlobalVar var = GlobalVar("someName");
  global_var_supply->ReserveGlobalVar(var);
  GlobalVar second_var = global_var_supply->UniqueGlobalFor("someName");
  GlobalVar third_var = global_var_supply->FreshGlobal("someName");

  EXPECT_TRUE(tvm::StructuralEqual()(var, second_var));
  EXPECT_FALSE(tvm::StructuralEqual()(var, third_var));
  EXPECT_EQ(second_var->name_hint.compare("someName"), 0);
  EXPECT_EQ(third_var->name_hint.compare("someName_1"), 0);
}

TEST(GlobalVarSupply, BuildIRModule) {
  auto x = relay::Var("x", relay::Type());
  auto f = relay::Function(tvm::Array<relay::Var>{x}, x, relay::Type(), {});
  GlobalVar var = GlobalVar("test");
  IRModule module = IRModule({{var, f}});

  GlobalVarSupply global_var_supply = GlobalVarSupply(module);
  GlobalVar second_var = global_var_supply->UniqueGlobalFor("test", false);
  GlobalVar third_var = global_var_supply->FreshGlobal("test", false);

  EXPECT_TRUE(tvm::StructuralEqual()(var, second_var));
  EXPECT_FALSE(tvm::StructuralEqual()(var, third_var));
  EXPECT_EQ(second_var->name_hint.compare("test"), 0);
  EXPECT_EQ(third_var->name_hint.compare("test_1"), 0);
}
