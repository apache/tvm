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

#ifndef TVM_IR_NAME_SUPPLY_H_
#define TVM_IR_NAME_SUPPLY_H_

#include <string>
#include <unordered_map>

#include "tvm/ir/expr.h"

namespace tvm {

class NameSupplyNode : public Object {
 public:
  NameSupplyNode() : NameSupplyNode("") {}

  explicit NameSupplyNode(const String& prefix);

  String FreshName(const String& name, bool add_prefix = true);

  String ReserveName(const String& name, bool add_prefix = true);

  bool ContainsName(const String& name, bool add_prefix = true);

  void Clear();

  void VisitAttrs(AttrVisitor* v) { v->Visit("prefix", &prefix_); }

  // Prefix for all GlobalVar names. It can be empty.
  std::string prefix_;

  static constexpr const char* _type_key = "NameSupply";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(NameSupplyNode, Object);

 private:
  String prefix_module_name(const String& name);

  std::string GetUniqueName(std::string name);

  // Key is function_name. Value is a counter.
  std::unordered_map<std::string, int> name_map;

  friend class NameSupply;
};

class NameSupply : public ObjectRef {
 public:
  TVM_DLL explicit NameSupply();

  TVM_DLL explicit NameSupply(const String& prefix,
                              std::unordered_map<std::string, int> name_map = {});

  explicit NameSupply(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*! \return mutable pointers to the node. */
  NameSupplyNode* operator->() const {
    auto* ptr = get_mutable();
    ICHECK(ptr != nullptr);
    return static_cast<NameSupplyNode*>(ptr);
  }

  TVM_DEFINE_OBJECT_REF_COW_METHOD(NameSupplyNode);
};

}  // namespace tvm

#endif  // TVM_IR_NAME_SUPPLY_H_
