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
#include <tvm/runtime/logging.h>
#include <tvm/runtime/memory.h>
#include <tvm/script/printer/doc.h>
#include <tvm/script/printer/ir_docsifier.h>
#include <tvm/script/printer/traced_object.h>

using namespace tvm;
using namespace tvm::script::printer;

class TestObjectNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "test.script.printer.irdocsifier.TestObject";
  TVM_DECLARE_FINAL_OBJECT_INFO(TestObjectNode, Object);
};

class TestObject : public ObjectRef {
 public:
  TestObject() : ObjectRef(runtime::make_object<TestObjectNode>()) {}
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(TestObject, ObjectRef, TestObjectNode);
};

TVM_REGISTER_NODE_TYPE(TestObjectNode);

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch([](TracedObject<TestObject> obj, IRDocsifier p) { return IdDoc("x"); });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch("tir", [](TracedObject<TestObject> obj, IRDocsifier p) { return IdDoc("tir"); });

TVM_STATIC_IR_FUNCTOR(IRDocsifier, vtable)
    .set_dispatch("relax",
                  [](TracedObject<TestObject> obj, IRDocsifier p) { return IdDoc("relax"); });

TEST(PrinterIRDocsifierTest, AsDoc) {
  IRDocsifier p(Map<String, String>{});
  ObjectPath path = ObjectPath::Root();
  TestObject obj;

  IdDoc doc = p->AsDoc<IdDoc>(MakeTraced(obj, path));

  ICHECK_EQ(doc->name, "x");
}

TEST(PrinterIRDocsifierTest, AsExprDoc) {
  IRDocsifier p(Map<String, String>{});
  ObjectPath path = ObjectPath::Root();
  TestObject obj;

  ExprDoc doc = p->AsExprDoc(MakeTraced(obj, path));

  ICHECK_EQ(Downcast<IdDoc>(doc)->name, "x");
}

TEST(PrinterIRDocsifierTest, WithDispatchToken) {
  IRDocsifier p(Map<String, String>{});
  TracedObject<TestObject> obj = MakeTraced(TestObject(), ObjectPath::Root());

  ICHECK_EQ(p->AsDoc<IdDoc>(obj)->name, "x");

  {
    auto ctx = p->WithDispatchToken("tir");
    ICHECK_EQ(p->AsDoc<IdDoc>(obj)->name, "tir");

    {
      auto ctx = p->WithDispatchToken("relax");
      ICHECK_EQ(p->AsDoc<IdDoc>(obj)->name, "relax");
    }

    ICHECK_EQ(p->AsDoc<IdDoc>(obj)->name, "tir");
  }

  ICHECK_EQ(p->AsDoc<IdDoc>(obj)->name, "x");
}

TEST(PrinterIRDocsifierTest, WithFrame) {
  IRDocsifier p(Map<String, String>{});
  TestObject obj;

  {
    VarDefFrame frame;
    auto ctx = p->WithFrame(frame);
    ICHECK_EQ(p->frames.size(), 1);

    p->vars->Define(obj, "x", ObjectPath::Root(), frame);
    ICHECK(p->vars->IsVarDefined(obj));
  }
  ICHECK_EQ(p->frames.size(), 0);
  ICHECK(!p->vars->IsVarDefined(obj));
}
