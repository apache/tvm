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
#include <tvm/node/repr_printer.h>
#include <tvm/runtime/container/map.h>
#include <tvm/script/printer/traced_object.h>

using namespace tvm;

namespace {

class DummyObjectNode : public Object {
 public:
  void VisitAttrs(AttrVisitor* v) {}

  static constexpr const char* _type_key = "TracedObjectTestDummyObject";
  TVM_DECLARE_FINAL_OBJECT_INFO(DummyObjectNode, Object);
};

class DummyObject : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(DummyObject, ObjectRef, DummyObjectNode);
};

TVM_REGISTER_NODE_TYPE(DummyObjectNode);

class ObjectWithAttrsNode : public Object {
 public:
  int64_t int64_attr = 5;
  Map<String, String> map_attr;
  Array<String> array_attr;
  DummyObject obj_attr;

  ObjectWithAttrsNode() : obj_attr(make_object<DummyObjectNode>()) {}

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("int64_attr", &int64_attr);
    v->Visit("map_attr", &map_attr);
    v->Visit("array_attr", &array_attr);
    v->Visit("obj_attr", &obj_attr);
  }

  static constexpr const char* _type_key = "TracedObjectTestObjectWithAttrs";
  TVM_DECLARE_FINAL_OBJECT_INFO(ObjectWithAttrsNode, Object);
};

class ObjectWithAttrs : public ObjectRef {
 public:
  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(ObjectWithAttrs, ObjectRef, ObjectWithAttrsNode);
};

TVM_REGISTER_NODE_TYPE(ObjectWithAttrsNode);

}  // anonymous namespace

TEST(TracedObjectTest, MakeTraced_RootObject) {
  ObjectWithAttrs root(make_object<ObjectWithAttrsNode>());
  auto root_traced = MakeTraced(root);

  static_assert(std::is_same<decltype(root_traced), TracedObject<ObjectWithAttrs>>::value);
  ICHECK(root_traced.GetPath()->PathsEqual(ObjectPath::Root()));
  ICHECK_EQ(root_traced.Get().get(), root.get());
}

TEST(TracedObjectTest, MakeTraced_WithPath) {
  ObjectWithAttrs obj(make_object<ObjectWithAttrsNode>());
  auto traced = MakeTraced(obj, ObjectPath::Root()->Attr("foo"));

  static_assert(std::is_same<decltype(traced), TracedObject<ObjectWithAttrs>>::value);
  ICHECK(traced.GetPath()->PathsEqual(ObjectPath::Root()->Attr("foo")));
  ICHECK_EQ(traced.Get().get(), obj.get());
}

TEST(TracedObjectTest, TracedObject_ImplicitConversionFromDerived) {
  DummyObject obj(make_object<DummyObjectNode>());
  auto traced = MakeTraced(obj);
  static_assert(std::is_same<decltype(traced), TracedObject<DummyObject>>::value);

  // Check that TracedObject<DummyObject> is implicitly converted to TracedObject<ObjectRef>
  auto base_traced = [](const TracedObject<ObjectRef>& base) { return base; }(traced);

  static_assert(std::is_same<decltype(base_traced), TracedObject<ObjectRef>>::value);
}

TEST(TracedObjectTest, TracedObject_GetAttr_ObjectRef) {
  ObjectWithAttrs root(make_object<ObjectWithAttrsNode>());
  auto root_traced = MakeTraced(root);
  auto obj_attr = root_traced.GetAttr(&ObjectWithAttrsNode::obj_attr);
  static_assert(std::is_same<decltype(obj_attr), TracedObject<DummyObject>>::value);
  ICHECK(obj_attr.GetPath()->PathsEqual(ObjectPath::Root()->Attr("obj_attr")));
  ICHECK_EQ(obj_attr.Get().get(), root->obj_attr.get());
}

TEST(TracedObjectTest, TracedObject_GetAttr_Map) {
  ObjectWithAttrs root(make_object<ObjectWithAttrsNode>());
  root->map_attr.Set("foo", "bar");

  auto root_traced = MakeTraced(root);
  auto map_attr = root_traced.GetAttr(&ObjectWithAttrsNode::map_attr);
  static_assert(std::is_same<decltype(map_attr), TracedMap<String, String>>::value);
  ICHECK(map_attr.GetPath()->PathsEqual(ObjectPath::Root()->Attr("map_attr")));
  ICHECK_EQ(map_attr.Get().get(), root->map_attr.get());

  auto map_val = map_attr.at("foo");
  ICHECK_EQ(map_val.Get(), "bar");
  ICHECK(
      map_val.GetPath()->PathsEqual(ObjectPath::Root()->Attr("map_attr")->MapValue(String("foo"))));
}

TEST(TracedObjectTest, TracedObject_GetAttr_Array) {
  ObjectWithAttrs root(make_object<ObjectWithAttrsNode>());
  root->array_attr.push_back("foo");
  root->array_attr.push_back("bar");

  auto root_traced = MakeTraced(root);
  auto array_attr = root_traced.GetAttr(&ObjectWithAttrsNode::array_attr);
  static_assert(std::is_same<decltype(array_attr), TracedArray<String>>::value);
  ICHECK(array_attr.GetPath()->PathsEqual(ObjectPath::Root()->Attr("array_attr")));
  ICHECK_EQ(array_attr.Get().get(), root->array_attr.get());

  auto array_val = array_attr[1];
  ICHECK_EQ(array_val.Get(), "bar");
  ICHECK(array_val.GetPath()->PathsEqual(ObjectPath::Root()->Attr("array_attr")->ArrayIndex(1)));
}

TEST(TracedObjectTest, TracedObject_GetAttr_Int64) {
  ObjectWithAttrs root(make_object<ObjectWithAttrsNode>());
  auto root_traced = MakeTraced(root);

  auto int64_attr = root_traced.GetAttr(&ObjectWithAttrsNode::int64_attr);
  static_assert(std::is_same<decltype(int64_attr), TracedBasicValue<int64_t>>::value);
  ICHECK_EQ(int64_attr.Get(), 5);
  ICHECK(int64_attr.GetPath()->PathsEqual(ObjectPath::Root()->Attr("int64_attr")));
}

TEST(TracedObjectTest, TracedObject_IsInstance) {
  ObjectRef dummy(make_object<DummyObjectNode>());
  auto traced = MakeTraced(dummy);
  ICHECK(traced.IsInstance<DummyObject>());
  ICHECK(!traced.IsInstance<ObjectWithAttrs>());
}

TEST(TracedObjectTest, TracedObject_Downcast) {
  ObjectRef root(make_object<DummyObjectNode>());
  auto traced = MakeTraced(root);

  auto as_dummy = traced.Downcast<DummyObject>();
  static_assert(std::is_same<decltype(as_dummy), TracedObject<DummyObject>>::value);
  ICHECK_EQ(as_dummy.Get(), root);

  // Try downcasting to a wrong type
  bool caught = false;
  try {
    traced.Downcast<ObjectWithAttrs>();
  } catch (std::exception& e) {
    caught = strstr(e.what(),
                    "Downcast from TracedObjectTestDummyObject to TracedObjectTestObjectWithAttrs "
                    "failed") != nullptr;
  }
  ICHECK(caught);
}

TEST(TracedObjectTest, TracedObject_TryDowncast) {
  ObjectRef root(make_object<DummyObjectNode>());
  auto traced = MakeTraced(root);

  auto as_dummy = traced.TryDowncast<DummyObject>();
  static_assert(std::is_same<decltype(as_dummy), TracedOptional<DummyObject>>::value);
  ICHECK(as_dummy.defined());
  ICHECK_EQ(as_dummy.value().Get(), root);

  // Try downcasting to a wrong type
  ICHECK(!traced.TryDowncast<ObjectWithAttrs>().defined());
}

TEST(TracedObjectTest, TracedMap_At) {
  Map<String, String> m({{"k1", "foo"}, {"k2", "bar"}});
  auto traced = MakeTraced(m);

  auto traced_foo = traced.at("k1");
  static_assert(std::is_same<decltype(traced_foo), TracedObject<String>>::value);
  ICHECK_EQ(traced_foo.Get(), "foo");
  ICHECK(traced_foo.GetPath()->PathsEqual(ObjectPath::Root()->MapValue(String("k1"))));
}

TEST(TracedObjectTest, TracedMap_Iterator) {
  Map<String, String> m({{"k1", "foo"}, {"k2", "bar"}});
  auto traced = MakeTraced(m);

  size_t k1_count = 0;
  size_t k2_count = 0;

  for (const auto& kv : traced) {
    if (kv.first == "k1") {
      ++k1_count;
      ICHECK_EQ(kv.second.Get(), "foo");
      ICHECK(kv.second.GetPath()->PathsEqual(ObjectPath::Root()->MapValue(String("k1"))));
    } else if (kv.first == "k2") {
      ++k2_count;
      ICHECK_EQ(kv.second.Get(), "bar");
      ICHECK(kv.second.GetPath()->PathsEqual(ObjectPath::Root()->MapValue(String("k2"))));
    } else {
      ICHECK(false);
    }
  }

  ICHECK_EQ(k1_count, 1);
  ICHECK_EQ(k2_count, 1);
}

TEST(TracedObjectTest, TracedArray_Index) {
  Array<String> a = {"foo", "bar"};
  auto traced = MakeTraced(a);

  auto traced_bar = traced[1];
  static_assert(std::is_same<decltype(traced_bar), TracedObject<String>>::value);
  ICHECK_EQ(traced_bar.Get(), "bar");
  ICHECK(traced_bar.GetPath()->PathsEqual(ObjectPath::Root()->ArrayIndex(1)));
}

TEST(TracedObjectTest, TracedArray_Iterator) {
  Array<String> a = {"foo", "bar"};
  auto traced = MakeTraced(a);

  size_t index = 0;
  for (const auto& x : traced) {
    if (index == 0) {
      ICHECK_EQ(x.Get(), "foo");
      ICHECK(x.GetPath()->PathsEqual(ObjectPath::Root()->ArrayIndex(0)));
    } else if (index == 1) {
      ICHECK_EQ(x.Get(), "bar");
      ICHECK(x.GetPath()->PathsEqual(ObjectPath::Root()->ArrayIndex(1)));
    } else {
      ICHECK(false);
    }
    ++index;
  }

  ICHECK_EQ(index, 2);
}

TEST(TracedObjectTest, TracedBasicValue_ApplyFunc) {
  auto traced = MakeTraced(123, ObjectPath::Root()->Attr("foo"));
  static_assert(std::is_same<decltype(traced), TracedBasicValue<int>>::value);

  auto transformed = traced.ApplyFunc([](int x) { return x + 4.0; });
  static_assert(std::is_same<decltype(transformed), TracedBasicValue<double>>::value);

  ICHECK(transformed.GetPath()->PathsEqual(ObjectPath::Root()->Attr("foo")));
}
