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
#include <cassert>
#include <sstream>

#include "picojson.h"

using picojson::object_with_ordered_keys;

void test_constructor() {
  object_with_ordered_keys obj;
  obj["foo"] = picojson::value(true);
  assert((obj.ordered_keys() == std::vector<std::string>{"foo"}));

  object_with_ordered_keys obj1{{"foo", picojson::value(true)}, {"bar", picojson::value(false)}};
  assert((obj1.ordered_keys() == std::vector<std::string>{"foo", "bar"}));

  object_with_ordered_keys obj2(obj1);
  assert((obj2.ordered_keys() == std::vector<std::string>{"foo", "bar"}));

  object_with_ordered_keys obj3(std::move(obj2));
  assert((obj3.ordered_keys() == std::vector<std::string>{"foo", "bar"}));

  obj = obj3;
  assert((obj.ordered_keys() == std::vector<std::string>{"foo", "bar"}));
}

void test_modifier() {
  object_with_ordered_keys obj{{"foo", picojson::value(true)}, {"bar", picojson::value(false)}};
  obj.insert({"abc", picojson::value(false)});
  assert((obj.ordered_keys() == std::vector<std::string>{"foo", "bar", "abc"}));
  obj.emplace("def", picojson::value(true));
  assert((obj.ordered_keys() == std::vector<std::string>{"foo", "bar", "abc", "def"}));
  obj.insert({"abc", picojson::value(true)});
  assert((obj.ordered_keys() == std::vector<std::string>{"foo", "bar", "abc", "def"}));
  auto it = obj.find("abc");
  it = obj.erase(it);
  assert((obj.ordered_keys() == std::vector<std::string>{"foo", "bar", "def"}));
  obj.erase("foo");
  assert((obj.ordered_keys() == std::vector<std::string>{"bar", "def"}));
  obj.clear();
  assert((obj.ordered_keys() == std::vector<std::string>{}));
}

void test_serializer() {
  picojson::object obj;

  obj["bar"] = picojson::value(static_cast<int64_t>(10));
  obj["baz"] = picojson::value(10.5);
  obj["foo"] = picojson::value(true);

  picojson::value v(obj);

  assert((v.serialize(false) == "{\"bar\":10,\"baz\":10.5,\"foo\":true}"));
}

int main() {
  test_constructor();
  test_modifier();
  test_serializer();
  return 0;
}
