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
 * \file tvm/ir/unique_name_supply.h
 * \brief UniqueNameSupply that can be used to generate unique variable names.
 */
#ifndef TVM_IR_UNIQUE_NAME_SUPPLY_H_
#define TVM_IR_UNIQUE_NAME_SUPPLY_H_

#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/expr.h>

#include <cstdint>
#include <string>
#include <utility>

namespace tvm {

/*!
 * \brief UniqueNameSupply can be used to generate unique names.
 */
class UniqueNameSupplyNode : public ffi::Object {
 public:
  /*!
   * \brief Empty constructor. Needed by the TVM_REGISTER_NODE_TYPE macro.
   */
  UniqueNameSupplyNode() = default;

  /*!
   * \brief Constructor.
   * \param prefix The prefix to be used with this UniqueNameSupply.
   * \param name_map The map used to guarantee uniqueness.
   */
  UniqueNameSupplyNode(const ffi::String& prefix, ffi::Map<ffi::String, int64_t> name_map)
      : prefix_(prefix), name_map(std::move(name_map)) {}

  /*!
   * \brief Generates a unique name from this UniqueNameSupply.
   * \param name The name from which the generated name is derived.
   * \param add_prefix If set to true, then the prefix of this UniqueNameSupply will be prepended to
   * the name.
   * \param add_underscore If set to true, add '_' between prefix and a digit.
   * \return A unique name.
   */
  ffi::String FreshName(const ffi::String& name, bool add_prefix = true,
                        bool add_underscore = true);

  /*!
   * \brief Reserves an existing name with this UniqueNameSupply.
   * \param name The name to be reserved.
   * \param add_prefix If set to true, then the prefix of this UniqueNameSupply will be prepended to
   * the name before reserving it. \return The name that was reserved with the UniqueNameSupply. It
   * can be different if a prefix is added.
   */
  ffi::String ReserveName(const ffi::String& name, bool add_prefix = true);

  /*!
   * \brief Checks if this UniqueNameSupply already generated a name.
   * \param name The name to check.
   * \param add_prefix If set to true, then the prefix of this UniqueNameSupply will be prepended to
   * the name before checking for it. \return True if the name has already been generated. False
   * otherwise.
   */
  bool ContainsName(const ffi::String& name, bool add_prefix = true);

  // Prefix for all GlobalVar names. It can be empty.
  std::string prefix_;

  static constexpr const bool _type_mutable = true;

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<UniqueNameSupplyNode>();
  }

  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("ir.UniqueNameSupply", UniqueNameSupplyNode, ffi::Object);

 private:
  /*! \brief Helper function to add the UniqueNameSupply prefix to the name. */
  ffi::String AddPrefixToName(const ffi::String& name);

  /*!
   * \brief Function that will generate a unique name.
   * \param name The name to be used as a base.
   * \param add_underscore If set to true, add '_' between prefix and a digit.
   * \return A unique name.
   */
  std::string GetUniqueName(std::string name, bool add_underscore = true);

  /*! \brief A map that is used to generate unique names. */
  ffi::Map<ffi::String, int64_t> name_map;
};

/*!
 * \brief Managed reference class to UniqueNameSupplyNode.
 * \sa UniqueNameSupplyNode
 */
class UniqueNameSupply : public ffi::ObjectRef {
 public:
  /*!
   * \brief Constructor.
   * \param prefix The prefix to be used with this UniqueNameSupply.
   * \param name_map An optional map.
   */
  TVM_DLL explicit UniqueNameSupply(const ffi::String& prefix = "",
                                    ffi::Map<ffi::String, int64_t> name_map = {});

  /*!
   * \brief Construct UniqueNameSupply by reserving names from the given iterator range.
   *
   * The functor should return the name of the dereferenced object.
   */
  template <typename Iter, typename Lambda>
  TVM_DLL UniqueNameSupply(Iter begin, Iter end, Lambda f) : UniqueNameSupply("") {
    for (auto it = begin; it != end; ++it) {
      this->operator->()->ReserveName(f(*it), false);
    }
  }

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(UniqueNameSupply, ffi::ObjectRef,
                                                UniqueNameSupplyNode);
};

}  // namespace tvm

#endif  // TVM_IR_UNIQUE_NAME_SUPPLY_H_
