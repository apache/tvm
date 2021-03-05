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
 * \file tvm/ir/module.h
 * \brief IRModule that holds the functions and type definitions.
 */
#ifndef TVM_IR_MODULE_H_
#define TVM_IR_MODULE_H_

#include <tvm/ir/adt.h>
#include <tvm/ir/expr.h>
#include <tvm/ir/function.h>
#include <tvm/ir/type.h>
#include <tvm/parser/source_map.h>
#include <tvm/runtime/container.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tvm {
class IRModule;
/*!
 * \brief IRModule that holds functions and type definitions.
 *
 *  IRModule is the basic unit for all IR transformations across the stack.
 *
 *  Many operations require access to the global IRModule.
 *  We pass the IRModule by value in a functional style as an explicit argument,
 *  but we mutate the Module while optimizing programs.
 * \sa IRModule
 */
class IRModuleNode : public Object {
 public:
  /*! \brief A map from ids to all global functions. */
  Map<GlobalVar, BaseFunc> functions;
  /*! \brief A map from global type vars to ADT type data. */
  Map<GlobalTypeVar, TypeData> type_definitions;
  /*! \brief The source map for the module. */
  parser::SourceMap source_map;

  IRModuleNode() : source_map() {}

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("functions", &functions);
    v->Visit("type_definitions", &type_definitions);
    v->Visit("global_var_map_", &global_var_map_);
    v->Visit("global_type_var_map_", &global_type_var_map_);
    v->Visit("source_map", &source_map);
  }

  TVM_DLL bool SEqualReduce(const IRModuleNode* other, SEqualReducer equal) const;

  TVM_DLL void SHashReduce(SHashReducer hash_reduce) const;

  /*!
   * \brief Add a function to the global environment.
   * \param var The var of the global function.
   * \param func The function.
   * \param update Controls whether you can replace a definition in the
   * environment.
   */
  TVM_DLL void Add(const GlobalVar& var, const BaseFunc& func, bool update = false);

  /*!
   * \brief Add a function to the global environment.
   * \param var The name of the global function.
   * \param func The function.
   *
   * It does not do type inference as Add does.
   */
  TVM_DLL void AddUnchecked(const GlobalVar& var, const BaseFunc& func);

  /*!
   * \brief Add a type-level definition to the global environment.
   * \param var The var of the global type definition.
   * \param type The ADT.
   * \param update Controls whether you can replace a definition in the
   * environment.
   */
  TVM_DLL void AddTypeDef(const GlobalTypeVar& var, const TypeData& type, bool update = false);

  /*!
   * \brief Add a type-level definition to the global environment.
   * \param var The var of the global type definition.
   * \param type The ADT.
   * \param update Controls whether you can replace a definition in the
   * environment.
   *
   * It does not do type checking as AddTypeDef does.
   */
  TVM_DLL void AddTypeDefUnchecked(const GlobalTypeVar& var, const TypeData& type,
                                   bool update = false);

  /*!
   * \brief Update a function in the global environment.
   * \param var The name of the global function to update.
   * \param func The new function.
   */
  TVM_DLL void Update(const GlobalVar& var, const BaseFunc& func);

  /*!
   * \brief Update a type definition in the global environment.
   * \param var The name of the global type definition to update.
   * \param type The new ADT.
   */
  TVM_DLL void UpdateTypeDef(const GlobalTypeVar& var, const TypeData& type);

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
  TVM_DLL bool ContainGlobalVar(const String& name) const;

  /*!
   * \brief Check if the global_type_var_map_ contains a global type variable.
   * \param name The variable name.
   * \returns true if contains, otherise false.
   */
  TVM_DLL bool ContainGlobalTypeVar(const String& name) const;

  /*!
   * \brief Lookup a global function by its variable.
   * \param str The unique string specifying the global variable.
   * \returns The global variable.
   */
  TVM_DLL GlobalVar GetGlobalVar(const String& str) const;

  /*!
   * \brief Collect all global vars defined in this module.
   * \returns An array of global vars
   */
  TVM_DLL Array<GlobalVar> GetGlobalVars() const;

  /*!
   * \brief Look up a global function by its name.
   * \param str The unique string specifying the global variable.
   * \returns The global variable.
   */
  TVM_DLL GlobalTypeVar GetGlobalTypeVar(const String& str) const;

  /*!
   * \brief Collect all global type vars defined in this module.
   * \returns An array of global type vars
   */
  TVM_DLL Array<GlobalTypeVar> GetGlobalTypeVars() const;

  /*!
   * \brief Find constructor of ADT using name
   * \param adt name of the ADT the constructor belongs to
   * \param cons name of the constructor
   * \returns Constructor of ADT, error if not found
   */
  TVM_DLL Constructor GetConstructor(const String& adt, const String& cons) const;

  /*!
   * \brief Look up a global function by its variable.
   * \param var The global var to lookup.
   * \returns The function named by the variable argument.
   */
  TVM_DLL BaseFunc Lookup(const GlobalVar& var) const;

  /*!
   * \brief Look up a global function by its string name
   * \param name The name of the function.
   * \returns The function named by the argument.
   */
  TVM_DLL BaseFunc Lookup(const String& name) const;

  /*!
   * \brief Look up a global type definition by its variable.
   * \param var The var of the global type definition.
   * \return The type definition.
   */
  TVM_DLL TypeData LookupTypeDef(const GlobalTypeVar& var) const;

  /*!
   * \brief Look up a global type definition by its name.
   * \param var The name of the global type definition.
   * \return The type definition.
   */
  TVM_DLL TypeData LookupTypeDef(const String& var) const;

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
  TVM_DLL void Update(const IRModule& other);

  /*!
   * \brief Import Relay code from the file at path.
   * \param path The path of the Relay code to import.
   *
   * \note The path resolution behavior is standard,
   * if abosolute will be the absolute file, if
   * relative it will be resovled against the current
   * working directory.
   */
  TVM_DLL void Import(const String& path);

  /*!
   * \brief Import Relay code from the file at path, relative to the standard library.
   * \param path The path of the Relay code to import.
   */
  TVM_DLL void ImportFromStd(const String& path);

  /*!
   * \brief The set of imported files.
   */
  TVM_DLL std::unordered_set<String> Imports() const;

  static constexpr const char* _type_key = "IRModule";
  static constexpr const bool _type_has_method_sequal_reduce = true;
  static constexpr const bool _type_has_method_shash_reduce = true;
  TVM_DECLARE_FINAL_OBJECT_INFO(IRModuleNode, Object);

 private:
  /*! \brief Helper function for registering a typedef's constructors */
  void RegisterConstructors(const GlobalTypeVar& var, const TypeData& type);

  /*! \brief A map from string names to global variables that
   * ensures global uniqueness.
   */
  Map<String, GlobalVar> global_var_map_;

  /*! \brief A map from string names to global type variables (ADT names)
   * that ensures global uniqueness.
   */
  Map<String, GlobalTypeVar> global_type_var_map_;

  /*! \brief A map from constructor tags to constructor objects
   * for convenient access
   */
  std::unordered_map<int32_t, Constructor> constructor_tag_map_;

  /*! \brief The files previously imported, required to ensure
      importing is idempotent for each module.
   */
  std::unordered_set<String> import_set_;
  friend class IRModule;
};

/*!
 * \brief Managed reference class to IRModuleNode.
 * \sa IRModuleNode
 */
class IRModule : public ObjectRef {
 public:
  /*!
   * \brief constructor
   * \param functions Functions in the module.
   * \param type_definitions Type definitions in the module.
   * \param import_set Set of imported files in the module
   * \param map The module source map.
   */
  TVM_DLL explicit IRModule(Map<GlobalVar, BaseFunc> functions,
                            Map<GlobalTypeVar, TypeData> type_definitions = {},
                            std::unordered_set<String> import_set = {}, parser::SourceMap map = {});

  /*! \brief default constructor */
  IRModule() : IRModule(Map<GlobalVar, BaseFunc>({})) {}
  /*!
   * \brief constructor
   * \param n The object pointer.
   */
  explicit IRModule(ObjectPtr<Object> n) : ObjectRef(n) {}
  /*! \return mutable pointers to the node. */
  IRModuleNode* operator->() const {
    auto* ptr = get_mutable();
    ICHECK(ptr != nullptr);
    return static_cast<IRModuleNode*>(ptr);
  }

  /*!
   * \brief Construct a module from a standalone expression.
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
  TVM_DLL static IRModule FromExpr(const RelayExpr& expr,
                                   const Map<GlobalVar, BaseFunc>& global_funcs = {},
                                   const Map<GlobalTypeVar, TypeData>& type_definitions = {});

  /*!
   * \brief Parse text format source file into an IRModule.
   * \param text A string of Relay source code.
   * \param source_path The path to the source file.
   * \return A Relay module.
   */
  TVM_DLL static IRModule FromText(const String& text, const String& source_path);

  /*! \brief Declare the container type. */
  using ContainerType = IRModuleNode;

  /*! \brief Declare whether Ref is nullable. */
  static constexpr bool _type_is_nullable = false;

  // allow copy on write.
  TVM_DEFINE_OBJECT_REF_COW_METHOD(IRModuleNode);
};

/*!
 * \brief Pretty print a node for debug purposes.
 *
 * \param node The node to be printed.
 * \return The text reperesentation.
 * \note This function does not show version or meta-data.
 *       Use AsText if you want to store the text.
 * \sa AsText.
 */
TVM_DLL String PrettyPrint(const ObjectRef& node);

/*!
 * \brief Render the node as a string in the text format.
 *
 * \param node The node to be rendered.
 * \param show_meta_data Whether to print meta data section.
 * \param annotate An optional callback function for attaching
 *        additional comment block to an expr.
 *
 * \note We support a limited set of IR nodes that are part of
 *       relay IR and
 *
 * \sa PrettyPrint.
 * \return The text representation.
 */
TVM_DLL String AsText(const ObjectRef& node, bool show_meta_data = true,
                      runtime::TypedPackedFunc<String(ObjectRef)> annotate = nullptr);
}  // namespace tvm
#endif  // TVM_IR_MODULE_H_
