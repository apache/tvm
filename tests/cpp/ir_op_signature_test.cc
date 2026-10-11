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
#include <tvm/ir/op.h>

#include <optional>
#include <string>

namespace tvm {

static std::string ValidationError(const Op& op, const Call& call) {
  try {
    op.Validate(call.get());
  } catch (const ffi::Error& error) {
    return error.what();
  }
  return "";
}

TEST(OpSignature, MetadataAndValidator) {
  OpDef def("test.op_signature.typed", "A synthetic signature.");
  def.signature(sig::arg("value", "The input."), sig::arg<PrimExpr>("index"),
                sig::var_args<PrimExpr>("rest", "Remaining indices."), sig::ty_arg<PrimType>("T"),
                sig::var_ty_args<PrimType>("Ts", "More types."), sig::call_attrs<DictAttrsNode>());
  Op op = def.op();
  EXPECT_EQ(op->args_info.size(), 2);
  EXPECT_EQ(op->args_info[0]->name, "value");
  EXPECT_EQ(op->args_info[0]->doc, "The input.");
  EXPECT_EQ(op->args_info[1]->name, "index");
  ASSERT_TRUE(op->var_args_info.has_value());
  EXPECT_EQ(op->var_args_info.value()->name, "rest");
  EXPECT_EQ(op->var_args_info.value()->doc, "Remaining indices.");
  EXPECT_EQ(op->ty_args_info.size(), 1);
  EXPECT_EQ(op->ty_args_info[0]->name, "T");
  ASSERT_TRUE(op->var_ty_args_info.has_value());
  EXPECT_EQ(op->var_ty_args_info.value()->name, "Ts");
  EXPECT_EQ(op->var_ty_args_info.value()->doc, "More types.");
  EXPECT_EQ(op->attrs_type_key, DictAttrsNode::_type_key);
  EXPECT_EQ(op->attrs_type_index, DictAttrsNode::RuntimeTypeIndex());

  Expr ordinary = Var("value", AnyType());
  PrimExpr index = PrimVar("index");
  Type scalar = PrimType::Int(32);
  Call minimum(AnyType(), op, {ordinary, index}, DictAttrs(), {scalar});
  Call extended(AnyType(), op, {ordinary, index, PrimVar("extra")}, DictAttrs(),
                {scalar, PrimType::Float(32)});
  EXPECT_TRUE(minimum.defined());
  EXPECT_TRUE(extended.defined());

  EXPECT_THROW(Call(AnyType(), op, {ordinary}, DictAttrs(), {scalar}).Validate(), ffi::Error);
  Call too_few = Call(AnyType(), op, {ordinary}, DictAttrs(), {scalar});
  EXPECT_TRUE(too_few.defined());

  std::string fixed_error =
      ValidationError(op, Call(AnyType(), op, {ordinary, ordinary}, DictAttrs(), {scalar}));
  EXPECT_EQ(fixed_error,
            "Op `test.op_signature.typed`: `Call.args[1]` (`index`) expected `ir.PrimExpr`, got "
            "`ir.Var[ty=ir.AnyType]`.");
  std::string tail_error =
      ValidationError(op, Call(AnyType(), op, {ordinary, index, ordinary}, DictAttrs(), {scalar}));
  EXPECT_EQ(tail_error,
            "Op `test.op_signature.typed`: `Call.args[2]` (`rest`) expected `ir.PrimExpr`, got "
            "`ir.Var[ty=ir.AnyType]`.");
  EXPECT_EQ(ValidationError(
                op, Call(AnyType(), op, {ordinary, index}, DictAttrs(), {scalar, StringType()})),
            "Op `test.op_signature.typed`: `Call.ty_args[1]` (`Ts`) expected `ir.PrimType`, got "
            "`ir.StringType`.");
  EXPECT_EQ(ValidationError(op, too_few),
            "Op `test.op_signature.typed`: Call.args expected at least 2 arguments, got 1");
  EXPECT_EQ(ValidationError(op, Call(AnyType(), op, {ordinary, index}, DictAttrs(), {})),
            "Op `test.op_signature.typed`: Call.ty_args expected at least 1 type argument, got "
            "0");
  EXPECT_EQ(ValidationError(op, Call(AnyType(), op, {ordinary, index}, std::nullopt, {scalar})),
            "Op `test.op_signature.typed`: Call.attrs expected `ir.DictAttrs`, got None");
  EXPECT_EQ(ValidationError(op, Call(AnyType(), op, {ordinary, index},
                                     Attrs(ffi::make_object<AttrsNode>()), {scalar})),
            "Op `test.op_signature.typed`: Call.attrs expected `ir.DictAttrs`, got `ir.Attrs`");

  ffi::Any non_expr(ffi::String("text"));
  TVMFFIAny raw = ffi::AnyView(non_expr).CopyToTVMFFIAny();
  EXPECT_EQ(ffi::TypeTraits<PrimExpr>::GetMismatchTypeInfo(&raw),
            ffi::TypeTraitsBase::GetMismatchTypeInfo(&raw));

  OpDef(op->name).set_attr<bool>("FSignatureTestMarker", true);
  EXPECT_EQ(op->args_info.size(), 2);
  EXPECT_TRUE(op->var_args_info.has_value());
  EXPECT_NO_THROW(OpDef(op->name).signature(sig::arg("replacement")));
  ASSERT_EQ(op->args_info.size(), 1);
  EXPECT_EQ(op->args_info[0]->name, "replacement");
  EXPECT_FALSE(op->var_args_info.has_value());
  EXPECT_TRUE(op->ty_args_info.empty());
  EXPECT_FALSE(op->var_ty_args_info.has_value());
  EXPECT_TRUE(op->attrs_type_key.empty());
  EXPECT_NO_THROW(op.Validate(minimum.get()));
  EXPECT_NO_THROW(OpDef(op->name).add_arg("legacy", ""));
  EXPECT_NO_THROW(OpDef(op->name).add_ty_arg("legacy_type", ""));
  EXPECT_NO_THROW(OpDef(op->name).attrs_type<AttrsNode>());
}

TEST(OpSignature, DefaultsAndRepeatedRegistration) {
  OpDef plain("test.op_signature.defaults");
  plain.signature(sig::arg("value"), sig::ty_arg("T"));
  Op op = plain.op();
  EXPECT_FALSE(op->var_args_info.has_value());
  EXPECT_FALSE(op->var_ty_args_info.has_value());
  EXPECT_NO_THROW(Call(AnyType(), op, {Var("x", AnyType())}, std::nullopt, {AnyType()}));
  EXPECT_NO_THROW(Call(AnyType(), op, {Var("x", AnyType())}, DictAttrs(), {AnyType()}));
  EXPECT_EQ(ValidationError(op, Call(AnyType(), op, {Var("x", AnyType()), Var("y", AnyType())},
                                     std::nullopt, {AnyType()})),
            "Op `test.op_signature.defaults`: Call.args expected 1 argument, got 2");

  OpDef unnamed("test.op_signature.unnamed");
  unnamed.signature(sig::arg<PrimExpr>(""));
  EXPECT_EQ(ValidationError(unnamed.op(), Call(AnyType(), unnamed.op(), {Var("x", AnyType())})),
            "Op `test.op_signature.unnamed`: `Call.args[0]` expected `ir.PrimExpr`, got "
            "`ir.Var[ty=ir.AnyType]`.");

  OpDef tails("test.op_signature.default_tails");
  tails.signature(sig::var_args("values"), sig::var_ty_args("types"));
  EXPECT_TRUE(tails.op()->args_info.empty());
  EXPECT_TRUE(tails.op()->ty_args_info.empty());
  EXPECT_EQ(tails.op()->var_args_info.value()->name, "values");
  EXPECT_EQ(tails.op()->var_ty_args_info.value()->name, "types");
  EXPECT_NO_THROW(Call(AnyType(), tails.op(), {}, std::nullopt, {}));
  EXPECT_NO_THROW(Call(AnyType(), tails.op(), {Var("x", AnyType())}, std::nullopt, {AnyType()}));

  OpDef repeated("test.op_signature.repeated");
  EXPECT_NO_THROW(repeated.signature(sig::arg("", "value doc"), sig::var_args("", "tail doc"),
                                     sig::ty_arg("", "type doc"),
                                     sig::var_ty_args("", "type tail doc")));
  EXPECT_EQ(repeated.op()->args_info[0]->name, "");
  EXPECT_EQ(repeated.op()->var_args_info.value()->name, "");
  EXPECT_EQ(repeated.op()->ty_args_info[0]->name, "");
  EXPECT_EQ(repeated.op()->var_ty_args_info.value()->name, "");
  EXPECT_TRUE(repeated.op()->validator != nullptr);
  EXPECT_NO_THROW(repeated.op().Validate(
      Call(AnyType(), repeated.op(), {Var("x", AnyType())}, std::nullopt, {AnyType()}).get()));
  EXPECT_NO_THROW(repeated.signature(sig::arg("changed"), sig::arg("")));
  EXPECT_EQ(repeated.op()->args_info.size(), 2);
  EXPECT_EQ(repeated.op()->args_info[0]->name, "changed");
  EXPECT_EQ(repeated.op()->args_info[1]->name, "");

  OpDef legacy("test.op_signature.legacy");
  legacy.add_arg("old", "Existing metadata.");
  EXPECT_NO_THROW(legacy.signature(sig::arg("new")));
  EXPECT_EQ(legacy.op()->args_info.size(), 1);
  EXPECT_EQ(legacy.op()->args_info[0]->name, "new");
  EXPECT_FALSE(legacy.op()->var_args_info.has_value());
  EXPECT_TRUE(legacy.op()->validator != nullptr);

  OpDef manual("test.op_signature.manual_hook");
  int manual_calls = 0;
  ffi::TypedFunction<ffi::Expected<void>(const CallNode*)> manual_hook(
      [&](const CallNode*) -> ffi::Expected<void> {
        ++manual_calls;
        return {};
      });
  manual.set_validator(manual_hook);
  EXPECT_THROW(manual.set_validator(manual_hook), ffi::Error);
  EXPECT_NO_THROW(manual.set_validator(manual_hook, true));
  EXPECT_NO_THROW(manual.signature(sig::arg("value")));
  EXPECT_EQ(manual.op()->args_info[0]->name, "value");
  Call manual_call(AnyType(), manual.op(), {Var("x", AnyType())});
  manual.op().Validate(manual_call.get());
  EXPECT_EQ(manual_calls, 1);
  ffi::TypedFunction<ffi::Expected<void>(const CallNode*)> failing_hook(
      [](const CallNode*) -> ffi::Expected<void> {
        return TVM_FFI_UNEXPECTED(TypeError) << "packed validator rejected Call";
      });
  manual.set_validator(failing_hook, true);
  EXPECT_THROW(manual.op().Validate(manual_call.get()), ffi::Error);
}

}  // namespace tvm
