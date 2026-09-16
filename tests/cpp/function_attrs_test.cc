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
#include <tvm/ir/function.h>
#include <tvm/ir/module.h>
#include <tvm/relax/expr.h>
#include <tvm/tirx/function.h>
#include <tvm/tirx/stmt.h>

namespace tvm {
namespace {

class CustomFuncNode : public BaseFuncNode {
 public:
  ffi::String payload = "preserved";
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.AttrsCustomFunc", CustomFuncNode, BaseFuncNode);
};

class MissingCopyFuncNode : public BaseFuncNode {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.AttrsMissingCopyFunc", MissingCopyFuncNode, BaseFuncNode);
};

class InvalidCopyFuncNode : public BaseFuncNode {
 public:
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("test.AttrsInvalidCopyFunc", InvalidCopyFuncNode, BaseFuncNode);
};

int invalid_copy_mode = 0;
TVM_FFI_STATIC_INIT_BLOCK() {
  ffi::reflection::ObjectDef<CustomFuncNode>().def_ro("payload", &CustomFuncNode::payload);
  ffi::reflection::TypeAttrDef<InvalidCopyFuncNode>().def(
      ffi::reflection::type_attr::kShallowCopy, [](BaseFunc func) -> Any {
        if (invalid_copy_mode == 0) return func;
        if (invalid_copy_mode == 1) return BaseFunc(ffi::make_object<CustomFuncNode>());
        return nullptr;
      });
}

TEST(FunctionAttrs, GenericSubtypeAndSharedAttributes) {
  BaseFunc original(ffi::make_object<CustomFuncNode>());
  original = WithAttr(std::move(original), "keep", 1);
  DictAttrs attrs = original->attrs;
  auto check_copy = [&](const BaseFunc& result) {
    EXPECT_FALSE(result.same_as(original));
    EXPECT_EQ(result->type_index(), original->type_index());
    EXPECT_EQ(result.as<CustomFuncNode>()->payload, "preserved");
    EXPECT_TRUE(result->ty.same_as(original->ty));
    EXPECT_TRUE(result->span.same_as(original->span));
    EXPECT_EQ(original->attrs->dict.size(), 1);
    EXPECT_TRUE(original->attrs.same_as(attrs));
    EXPECT_EQ(attrs->dict.at("keep").cast<int>(), 1);
  };
  auto added = WithAttr(original, "added", 2);
  check_copy(added);
  EXPECT_EQ(added->attrs->dict.at("added").cast<int>(), 2);
  auto updated = WithAttrs(original, {{"keep", 3}, {"added", 4}});
  check_copy(updated);
  EXPECT_EQ(updated->attrs->dict.at("keep").cast<int>(), 3);
  auto removed = WithoutAttr(original, "keep");
  check_copy(removed);
  EXPECT_TRUE(removed->attrs->dict.empty());
  EXPECT_TRUE(WithAttrs(original, {}).same_as(original));
}

TEST(FunctionAttrs, UniqueReuseAndSharedDictionary) {
  BaseFunc func(ffi::make_object<CustomFuncNode>());
  const auto* ptr = func.get();
  DictAttrs shared_attrs = func->attrs;
  func = WithAttr(std::move(func), "key", 1);
  EXPECT_EQ(func.get(), ptr);
  EXPECT_TRUE(shared_attrs->dict.empty());
  func = WithAttrs(std::move(func), {{"key", 2}, {"other", 3}});
  EXPECT_EQ(func.get(), ptr);
  func = WithoutAttr(std::move(func), "key");
  EXPECT_EQ(func.get(), ptr);
  EXPECT_FALSE(func->attrs->dict.count("key"));
  EXPECT_EQ(func->attrs->dict.at("other").cast<int>(), 3);
}

TEST(FunctionAttrs, TypedFunctionsAndModule) {
  tirx::PrimFunc prim({}, tirx::Evaluate(0));
  auto prim_copy = WithAttr(prim, "key", 1);
  EXPECT_FALSE(prim_copy.same_as(prim));
  EXPECT_TRUE(prim_copy->body.same_as(prim->body));
  EXPECT_TRUE(prim_copy->params.same_as(prim->params));
  EXPECT_TRUE(prim_copy->ret_type.same_as(prim->ret_type));
  BaseFunc base = prim;
  auto generic_copy = WithAttrs(base, {{"key", 1}});
  EXPECT_EQ(generic_copy->type_index(), prim->type_index());
  EXPECT_TRUE(generic_copy.as<tirx::PrimFuncNode>()->body.same_as(prim->body));

  relax::ExternFunc ext("external_symbol");
  auto ext_copy = WithoutAttr(WithAttr(ext, "key", 1), "key");
  EXPECT_EQ(ext_copy->global_symbol, ext->global_symbol);
  EXPECT_TRUE(ext_copy->attrs->dict.empty());
  IRModule mod = IRModule::FromExpr(prim);
  auto mod_copy = WithAttrs(mod, {{"key", 1}});
  EXPECT_TRUE(mod->attrs->dict.empty());
  EXPECT_EQ(mod_copy->attrs->dict.at("key").cast<int>(), 1);
  EXPECT_TRUE(WithoutAttr(mod_copy, "key")->attrs->dict.empty());
}

TEST(FunctionAttrs, MissingAndInvalidHooksPreserveInput) {
  BaseFunc missing(ffi::make_object<MissingCopyFuncNode>());
  EXPECT_THROW(WithAttr(missing, "key", 1), ffi::Error);
  EXPECT_THROW(WithAttrs(missing, {{"key", 1}}), ffi::Error);
  EXPECT_THROW(WithoutAttr(missing, "key"), ffi::Error);
  EXPECT_TRUE(WithAttrs(missing, {}).same_as(missing));
  // Unique input needs no copy hook.
  const auto* ptr = missing.get();
  missing = WithAttr(std::move(missing), "key", 1);
  EXPECT_EQ(missing.get(), ptr);

  BaseFunc invalid(ffi::make_object<InvalidCopyFuncNode>());
  invalid = WithAttr(std::move(invalid), "key", 1);
  for (invalid_copy_mode = 0; invalid_copy_mode < 3; ++invalid_copy_mode) {
    EXPECT_THROW(WithAttr(invalid, "key", 2), ffi::Error);
    EXPECT_THROW(WithAttrs(invalid, {{"key", 2}}), ffi::Error);
    EXPECT_THROW(WithoutAttr(invalid, "key"), ffi::Error);
    EXPECT_EQ(invalid->attrs->dict.at("key").cast<int>(), 1);
  }
}

TEST(FunctionAttrs, MovedFromAttributes) {
  BaseFunc func(ffi::make_object<CustomFuncNode>());
  // Simulate a caller moving through the base handle, bypassing DictAttrs' reset-on-move.
  ffi::ObjectRef moved =
      std::move(static_cast<ffi::ObjectRef&>(const_cast<BaseFuncNode*>(func.operator->())->attrs));
  EXPECT_EQ(WithAttr(func, "key", 1)->attrs->dict.size(), 1);
  EXPECT_EQ(WithAttrs(func, {{"key", 1}})->attrs->dict.size(), 1);
  EXPECT_TRUE(WithoutAttr(func, "key")->attrs->dict.empty());
  EXPECT_FALSE(func->attrs.defined());
}

}  // namespace
}  // namespace tvm
