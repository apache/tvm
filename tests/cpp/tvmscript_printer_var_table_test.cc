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
#include <tvm/node/object_path.h>
#include <tvm/runtime/logging.h>
#include <tvm/script/printer/frame.h>
#include <tvm/script/printer/var_table.h>
#include <tvm/tir/var.h>

using namespace tvm;
using namespace tvm::script::printer;

TEST(PrinterVarTableTest, Define) {
  VarTable vars;
  MetadataFrame frame;
  tir::Var x("x");
  ObjectPath object_path = ObjectPath::Root();

  IdDoc doc = vars->Define(x, "x", object_path, frame);

  ICHECK_EQ(doc->name, "x");

  IdDoc second_doc = Downcast<IdDoc>(vars->GetVarDoc(x, object_path).value());

  ICHECK_EQ(second_doc->name, "x");
}

TEST(PrinterVarTableTest, DefineByDoc) {
  VarTable vars;
  MetadataFrame frame;
  tir::Var x("x");
  ObjectPath object_path = ObjectPath::Root();

  auto doc_factory = []() { return LiteralDoc::Str("x"); };

  vars->DefineByDoc(x, doc_factory, frame);

  ExprDoc doc = vars->GetVarDoc(x, object_path).value();

  ICHECK_EQ(Downcast<String>(Downcast<LiteralDoc>(doc)->value), "x");
}

TEST(PrinterVarTableTest, GetVarDocWithUnknownVariable) {
  VarTable vars;
  MetadataFrame frame;
  tir::Var x("x");
  tir::Var y("y");
  ObjectPath object_path = ObjectPath::Root();

  Doc doc = vars->Define(x, "x", object_path, frame);
  ICHECK(!vars->GetVarDoc(y, object_path).defined());
}

TEST(PrinterVarTableTest, GetVarDocWithObjectPath) {
  VarTable vars;
  MetadataFrame frame;
  tir::Var x("x");
  ObjectPath object_path = ObjectPath::Root();
  ObjectPath second_object_path = ObjectPath::Root()->Attr("x");

  IdDoc doc = vars->Define(x, "x", object_path, frame);
  ICHECK_EQ(doc->source_paths[0], object_path);
  ICHECK_EQ(doc->source_paths.size(), 1);

  Doc second_doc = vars->GetVarDoc(x, second_object_path).value();
  ICHECK_EQ(second_doc->source_paths[0], second_object_path);
  ICHECK_EQ(second_doc->source_paths.size(), 1);
}

TEST(PrinterVarTableTest, IsVarDefined) {
  VarTable vars;
  MetadataFrame frame;
  tir::Var x("x");
  tir::Var y("y");
  ObjectPath object_path = ObjectPath::Root();

  vars->Define(x, "x", object_path, frame);
  ICHECK(vars->IsVarDefined(x));
  ICHECK(!vars->IsVarDefined(y));
}

TEST(PrinterVarTableTest, VarRemovedAfterFrameOutOfScope) {
  VarTable vars;
  MetadataFrame frame;
  tir::Var x("x");
  ObjectPath object_path = ObjectPath::Root();

  vars->Define(x, "x", object_path, frame);
  ICHECK(vars->IsVarDefined(x));

  frame->ExitWithScope();
  ICHECK(!vars->IsVarDefined(x));
}

TEST(PrinterVarTableTest, DefineDuplicateName) {
  VarTable vars;
  MetadataFrame frame;
  tir::Var x("x");
  tir::Var y("y");
  ObjectPath object_path = ObjectPath::Root();

  IdDoc x_doc = vars->Define(x, "x", object_path, frame);
  IdDoc y_doc = vars->Define(y, "x", object_path, frame);

  ICHECK_NE(x_doc->name, y_doc->name);
}

TEST(PrinterVarTableTest, DefineDuplicateVariable) {
  VarTable vars;
  MetadataFrame frame;
  tir::Var x("x");
  ObjectPath object_path = ObjectPath::Root();

  vars->Define(x, "x", object_path, frame);

  bool failed = false;
  try {
    vars->Define(x, "x", object_path, frame);
  } catch (...) {
    failed = true;
  }
  ASSERT_EQ(failed, true);
}

TEST(PrinterVarTableTest, DefineByDocWithIdDoc) {
  VarTable vars;
  MetadataFrame frame;
  tir::Var x("x");
  ObjectPath object_path = ObjectPath::Root();

  bool failed = false;
  try {
    // User has to use `Define` if variable needs to be mapped to IdDoc
    vars->DefineByDoc(
        x, []() { return IdDoc("x"); }, frame);
  } catch (...) {
    failed = true;
  }
  ASSERT_EQ(failed, true);
}
