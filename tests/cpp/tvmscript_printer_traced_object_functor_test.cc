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
#include <tvm/runtime/packed_func.h>
#include <tvm/script/printer/traced_object.h>
#include <tvm/script/printer/traced_object_functor.h>

using namespace tvm;
using namespace tvm::script::printer;

namespace {

class FooObjectNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "test.FooObject";
  TVM_DECLARE_FINAL_OBJECT_INFO(FooObjectNode, Object);
};

class FooObject : public ObjectRef {
 public:
  FooObject() { this->data_ = make_object<FooObjectNode>(); }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(FooObject, ObjectRef, FooObjectNode);
};

TVM_REGISTER_NODE_TYPE(FooObjectNode);

class BarObjectNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "test.BarObject";
  TVM_DECLARE_FINAL_OBJECT_INFO(BarObjectNode, Object);
};

class BarObject : public ObjectRef {
 public:
  BarObject() { this->data_ = make_object<BarObjectNode>(); }
  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(BarObject, ObjectRef, BarObjectNode);
};

TVM_REGISTER_NODE_TYPE(BarObjectNode);

String ComputeFoo(TracedObject<FooObject> foo) { return "Foo"; }

}  // anonymous namespace

TEST(TracedObjectFunctorTest, NormalRegistration) {
  TracedObjectFunctor<String> functor;
  ObjectPath path = ObjectPath::Root();

  functor.set_dispatch([](TracedObject<FooObject> o) -> String { return "Foo"; });
  functor.set_dispatch([](TracedObject<BarObject> o) -> String { return "Bar"; });

  ICHECK_EQ(functor("", MakeTraced(FooObject(), path)), "Foo");
  ICHECK_EQ(functor("", MakeTraced(BarObject(), path)), "Bar");
}

TEST(TracedObjectFunctorTest, RegistrationWithFunction) {
  TracedObjectFunctor<String> functor;
  ObjectPath path = ObjectPath::Root();

  functor.set_dispatch([](TracedObject<FooObject> o) -> String { return "FooLambda"; });
  functor.set_dispatch("tir", ComputeFoo);

  ICHECK_EQ(functor("", MakeTraced(FooObject(), path)), "FooLambda");
  ICHECK_EQ(functor("tir", MakeTraced(FooObject(), path)), "Foo");
}

TEST(TracedObjectFunctorTest, RegistrationWithDispatchToken) {
  TracedObjectFunctor<String> functor;
  ObjectPath path = ObjectPath::Root();

  functor.set_dispatch([](TracedObject<FooObject> o) -> String { return "Foo"; });
  functor.set_dispatch("tir", [](TracedObject<FooObject> o) -> String { return "Foo tir"; });
  functor.set_dispatch("relax", [](TracedObject<FooObject> o) -> String { return "Foo relax"; });

  ICHECK_EQ(functor("", MakeTraced(FooObject(), path)), "Foo");
  ICHECK_EQ(functor("tir", MakeTraced(FooObject(), path)), "Foo tir");
  ICHECK_EQ(functor("relax", MakeTraced(FooObject(), path)), "Foo relax");
  ICHECK_EQ(functor("xyz", MakeTraced(FooObject(), path)), "Foo");
}

TEST(TracedObjectFunctorTest, RegistrationWithPackedFunc) {
  TracedObjectFunctor<String> functor;
  ObjectPath path = ObjectPath::Root();

  auto f_default = [](runtime::TVMArgs, runtime::TVMRetValue* ret) { *ret = String("default"); };
  auto f_tir = [](runtime::TVMArgs, runtime::TVMRetValue* ret) { *ret = String("tir"); };

  functor.set_dispatch("", FooObjectNode::RuntimeTypeIndex(), runtime::PackedFunc(f_default));
  functor.set_dispatch("tir", FooObjectNode::RuntimeTypeIndex(), runtime::PackedFunc(f_tir));

  ICHECK_EQ(functor("", MakeTraced(FooObject(), path)), "default");
  ICHECK_EQ(functor("tir", MakeTraced(FooObject(), path)), "tir");
}

TEST(TracedObjectFunctorTest, ExtraArg) {
  TracedObjectFunctor<int, int> functor;
  ObjectPath path = ObjectPath::Root();

  functor.set_dispatch([](TracedObject<FooObject> o, int x) { return x; });
  functor.set_dispatch([](TracedObject<BarObject> o, int x) { return x + 1; });

  ICHECK_EQ(functor("", MakeTraced(FooObject(), path), 2), 2);
  ICHECK_EQ(functor("", MakeTraced(BarObject(), path), 2), 3);
  ICHECK_EQ(functor("tir", MakeTraced(BarObject(), path), 2), 3);
}

TEST(TracedObjectFunctorTest, RemoveDispatchFunction) {
  TracedObjectFunctor<String> functor;
  ObjectPath path = ObjectPath::Root();

  functor.set_dispatch([](TracedObject<FooObject> o) -> String { return "Foo"; });
  functor.set_dispatch("tir", [](TracedObject<FooObject> o) -> String { return "Foo tir"; });

  ICHECK_EQ(functor("", MakeTraced(FooObject(), path)), "Foo");
  ICHECK_EQ(functor("tir", MakeTraced(FooObject(), path)), "Foo tir");

  functor.remove_dispatch("tir", FooObjectNode::RuntimeTypeIndex());
  ICHECK_EQ(functor("tir", MakeTraced(FooObject(), path)), "Foo");
}

TEST(TracedObjectFunctorTest, CallWithUnregisteredType) {
  TracedObjectFunctor<int, int> functor;
  ObjectPath path = ObjectPath::Root();

  bool failed = false;
  try {
    ICHECK_EQ(functor("", MakeTraced(FooObject(), path), 2), 2);
  } catch (...) {
    failed = true;
  }
  ASSERT_EQ(failed, true);
}

TEST(TracedObjectFunctorTest, DuplicateRegistration_WithoutToken) {
  TracedObjectFunctor<int, int> functor;
  ObjectPath path = ObjectPath::Root();

  functor.set_dispatch([](TracedObject<FooObject> o, int x) { return x; });

  bool failed = false;
  try {
    functor.set_dispatch([](TracedObject<FooObject> o, int x) { return x; });
  } catch (...) {
    failed = true;
  }
  ASSERT_EQ(failed, true);
}

TEST(TracedObjectFunctorTest, DuplicateRegistration_WithToken) {
  TracedObjectFunctor<int, int> functor;
  ObjectPath path = ObjectPath::Root();

  functor.set_dispatch("tir", [](TracedObject<FooObject> o, int x) { return x; });

  bool failed = false;
  try {
    functor.set_dispatch("tir", [](TracedObject<FooObject> o, int x) { return x; });
  } catch (...) {
    failed = true;
  }
  ASSERT_EQ(failed, true);
}
