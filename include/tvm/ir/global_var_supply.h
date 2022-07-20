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

#ifndef TVM_IR_GLOBAL_VAR_SUPPLY_H_
#define TVM_IR_GLOBAL_VAR_SUPPLY_H_

#include <string>
#include <unordered_map>

#include "tvm/ir/expr.h"
#include "tvm/ir/module.h"
#include "tvm/ir/name_supply.h"

namespace tvm {

class GlobalVarSupplyNode : public Object {
 public:
  GlobalVarSupplyNode() : GlobalVarSupplyNode(NameSupply("")) {}

  explicit GlobalVarSupplyNode(NameSupply name_supply);

  GlobalVar FreshGlobal(String name, bool add_prefix = true);

  GlobalVar UniqueGlobalFor(const String& name, bool add_prefix = true);

  void ReserveGlobalVar(const GlobalVar& var, bool allow_conflict = false);

  void VisitAttrs(AttrVisitor* v) { v->Visit("name_supply", &name_supply_); }

  NameSupply name_supply_;

  static constexpr const char* _type_key = "GlobalVarSupply";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(GlobalVarSupplyNode, Object);

 private:
  std::unordered_map<std::string, GlobalVar> name_to_var_map_;

  friend class GlobalVarSupply;
};

class GlobalVarSupply : public ObjectRef {
 public:
  TVM_DLL explicit GlobalVarSupply(const NameSupply& name_supply = NameSupply(),
                                   std::unordered_map<std::string, GlobalVar> name_to_var_map = {});

  TVM_DLL explicit GlobalVarSupply(const Array<IRModule>& modules);

  TVM_DLL explicit GlobalVarSupply(const IRModule module);

  explicit GlobalVarSupply(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*! \return mutable pointers to the node. */
  GlobalVarSupplyNode* operator->() const {
    auto* ptr = get_mutable();
    ICHECK(ptr != nullptr);
    return static_cast<GlobalVarSupplyNode*>(ptr);
  }

  TVM_DEFINE_OBJECT_REF_COW_METHOD(GlobalVarSupplyNode);
};

}  // namespace tvm

#endif  // TVM_IR_GLOBAL_VAR_SUPPLY_H_
