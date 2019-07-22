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
 * \file tvm/relay/module.h
 * \brief The global environment: contains information needed to
 * compile & optimize Relay programs.
 */
#ifndef TVM_RELAY_MODULE_H_
#define TVM_RELAY_MODULE_H_

#include <tvm/relay/error.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/adt.h>
#include <tvm/relay/op.h>
#include <tvm/relay/type.h>
#include <string>
#include <vector>
#include <unordered_map>

namespace tvm {
namespace relay {

struct Module;

/*! \brief The global environment of Relay programs.
 *
 *  The global environment contains the global
 *  information needed to compile a Relay program.
 *
 *  It contains all global functions, and configuration
 *  options.
 *
 *  Many operations require access to the global
 *  Module. We pass the Module by value
 *  in a functional style as an explicit argument,
 *  but we mutate the Module while optimizing
 *  Relay programs.
 *
 *  The functional style allows users to construct custom
 *  environments easily, for example each thread can store
 *  a Module while auto-tuning.
 */

class ModuleNode : public RelayNode {
 public:
  /*! \brief A map from ids to all global functions. */
  tvm::Map<GlobalVar, Function> functions;
  /*! \brief A map from global type vars to ADT type data. */
  tvm::Map<GlobalTypeVar, TypeData> type_definitions;

  ModuleNode() {}

  void VisitAttrs(tvm::AttrVisitor* v) final {
    v->Visit("functions", &functions);
    v->Visit("type_definitions", &type_definitions);
    v->Visit("global_var_map_", &global_var_map_);
    v->Visit("global_type_var_map_", &global_type_var_map_);
  }

  TVM_DLL static Module make(tvm::Map<GlobalVar, Function> global_funcs,
                             tvm::Map<GlobalTypeVar, TypeData> global_type_defs);

  /*!
   * \brief Add a function to the global environment.
   * \param var The var of the global function.
   * \param func The function.
   * \param update Controls whether you can replace a definition in the
   * environment.
   */
  TVM_DLL void Add(const GlobalVar& var, const Function& func, bool update = false);

  /*!
   * \brief Add a type-level definition to the global environment.
   * \param var The var of the global type definition.
   * \param type The type definition.
   */
  TVM_DLL void AddDef(const GlobalTypeVar& var, const TypeData& type);

  /*!
   * \brief Add a function to the global environment.
   * \param var The name of the global function.
   * \param func The function.
   *
   * It does not do type inference as Add does.
   */
  TVM_DLL void AddUnchecked(const GlobalVar& var, const Function& func);

  /*!
   * \brief Update a function in the global environment.
   * \param var The name of the global function to update.
   * \param func The new function.
   */
  TVM_DLL void Update(const GlobalVar& var, const Function& func);

  /*!
   * \brief Remove a function from the global environment.
   * \param var The name of the global function to update.
   */
  TVM_DLL void Remove(const GlobalVar& var);

  /*!
   * \brief Check if the global_var_map_ contains a global variable.
   * \param name The variable name.
   * \returns true if contains, otherise false.
   */
  TVM_DLL bool ContainGlobalVar(const std::string& name) const;

  /*!
   * \brief Lookup a global function by its variable.
   * \param str The unique string specifying the global variable.
   * \returns The global variable.
   */
  TVM_DLL GlobalVar GetGlobalVar(const std::string& str) const;

  /*!
   * \brief Look up a global function by its name.
   * \param str The unique string specifying the global variable.
   * \returns The global variable.
   */
  TVM_DLL GlobalTypeVar GetGlobalTypeVar(const std::string& str) const;

  /*!
   * \brief Look up a global function by its variable.
   * \param var The global var to lookup.
   * \returns The function named by the variable argument.
   */
  TVM_DLL Function Lookup(const GlobalVar& var) const;

  /*!
   * \brief Look up a global function by its string name
   * \param name The name of the function.
   * \returns The function named by the argument.
   */
  TVM_DLL Function Lookup(const std::string& name) const;

  /*!
   * \brief Look up a global type definition by its variable.
   * \param var The var of the global type definition.
   * \return The type definition.
   */
  TVM_DLL TypeData LookupDef(const GlobalTypeVar& var) const;

  /*!
   * \brief Look up a global type definition by its name.
   * \param var The name of the global type definition.
   * \return The type definition.
   */
  TVM_DLL TypeData LookupDef(const std::string& var) const;

  /*!
   * \brief Look up a constructor by its tag.
   * \param tag The tag for the constructor.
   * \return The constructor object.
   */
  TVM_DLL Constructor LookupTag(const int32_t tag);

  /*!
   * \brief Update the functions inside this environment by
   *        functions in another environment.
   * \param other The other environment.
   */
  TVM_DLL void Update(const Module& other);

  /*! \brief Construct a module from a standalone expression.
   *
   * Allows one to optionally pass a global function map and
   * map of type definitions as well.
   *
   * \param expr The expression to set as the main function to the module.
   * \param global_funcs The global function map.
   * \param type_definitions Map of global type definitions
   *
   * \returns A module with expr set as the main function.
   */
  TVM_DLL static Module FromExpr(
    const Expr& expr,
    const tvm::Map<GlobalVar, Function>& global_funcs = {},
    const tvm::Map<GlobalTypeVar, TypeData>& type_definitions = {});

  static constexpr const char* _type_key = "relay.Module";
  TVM_DECLARE_NODE_TYPE_INFO(ModuleNode, Node);

 private:
  /*! \brief Helper function for registering a typedef's constructors */
  void RegisterConstructors(const GlobalTypeVar& var, const TypeData& type);

  /*! \brief A map from string names to global variables that
   * ensures global uniqueness.
   */
  tvm::Map<std::string, GlobalVar> global_var_map_;

  /*! \brief A map from string names to global type variables (ADT names)
   * that ensures global uniqueness.
   */
  tvm::Map<std::string, GlobalTypeVar> global_type_var_map_;

  /*! \brief A map from constructor tags to constructor objects
   * for convenient access
   */
  std::unordered_map<int32_t, Constructor> constructor_tag_map_;
};

struct Module : public NodeRef {
  Module() {}
  explicit Module(NodePtr<tvm::Node> p) : NodeRef(p) {}

  inline ModuleNode* operator->() const {
    return static_cast<ModuleNode*>(node_.get());
  }

  using ContainerType = ModuleNode;
};


}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_MODULE_H_
