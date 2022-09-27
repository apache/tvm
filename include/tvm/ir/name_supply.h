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

/*!
 * \file tvm/ir/name_supply.h
 * \brief NameSupply that can be used to generate unique variable names.
 */
#ifndef TVM_IR_NAME_SUPPLY_H_
#define TVM_IR_NAME_SUPPLY_H_

#include <string>
#include <unordered_map>
#include <utility>

#include "tvm/ir/expr.h"

namespace tvm {

/*!
 * \brief NameSupply can be used to generate unique names.
 */
class NameSupplyNode : public Object {
 public:
  /*!
   * \brief Empty constructor. Needed by the TVM_REGISTER_NODE_TYPE macro.
   */
  NameSupplyNode() = default;

  /*!
   * \brief Constructor.
   * \param prefix The prefix to be used with this NameSupply.
   * \param name_map The map used to guarantee uniqueness.
   */
  NameSupplyNode(const String& prefix, std::unordered_map<std::string, int> name_map)
      : prefix_(prefix), name_map(std::move(name_map)) {}

  /*!
   * \brief Generates a unique name from this NameSupply.
   * \param name The name from which the generated name is derived.
   * \param add_prefix If set to true, then the prefix of this NameSupply will be prepended to the
   * name. \return A unique name.
   */
  String FreshName(const String& name, bool add_prefix = true);

  /*!
   * \brief Reserves an existing name with this NameSupply.
   * \param name The name to be reserved.
   * \param add_prefix If set to true, then the prefix of this NameSupply will be prepended to the
   * name before reserving it. \return The name that was reserved with the NameSupply. It can be
   * different if a prefix is added.
   */
  String ReserveName(const String& name, bool add_prefix = true);

  /*!
   * \brief Checks if this NameSupply already generated a name.
   * \param name The name to check.
   * \param add_prefix If set to true, then the prefix of this NameSupply will be prepended to the
   * name before checking for it. \return True if the name has already been generated. False
   * otherwise.
   */
  bool ContainsName(const String& name, bool add_prefix = true);

  void VisitAttrs(AttrVisitor* v) {}

  // Prefix for all GlobalVar names. It can be empty.
  std::string prefix_;

  static constexpr const char* _type_key = "NameSupply";
  static constexpr const bool _type_has_method_sequal_reduce = false;
  static constexpr const bool _type_has_method_shash_reduce = false;
  TVM_DECLARE_FINAL_OBJECT_INFO(NameSupplyNode, Object);

 private:
  /*! \brief Helper function to add the NameSupply prefix to the name. */
  String add_prefix_to_name(const String& name);

  /*!
   * \brief Function that will generate a unique name.
   * \param name The name to be used as a base.
   * \return A unique name.
   */
  std::string GetUniqueName(std::string name);

  /*! \brief A map that is used to generate unique names. */
  std::unordered_map<std::string, int> name_map;
};

/*!
 * \brief Managed reference class to NameSupplyNode.
 * \sa NameSupplyNode
 */
class NameSupply : public ObjectRef {
 public:
  /*!
   * \brief Constructor.
   * \param prefix The prefix to be used with this NameSupply.
   * \param name_map An optional map.
   */
  TVM_DLL explicit NameSupply(const String& prefix,
                              std::unordered_map<std::string, int> name_map = {});

  TVM_DEFINE_MUTABLE_OBJECT_REF_METHODS(NameSupply, ObjectRef, NameSupplyNode);
};

}  // namespace tvm

#endif  // TVM_IR_NAME_SUPPLY_H_
