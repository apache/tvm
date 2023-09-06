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
 * \file library_module.h
 * \brief Module that builds from a libary of symbols.
 */
#ifndef TVM_RUNTIME_LIBRARY_MODULE_H_
#define TVM_RUNTIME_LIBRARY_MODULE_H_

#include <tvm/runtime/c_backend_api.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/module.h>

#include <functional>
#include <string>
#include <vector>

namespace tvm {
namespace runtime {

/*! \brief Load a module with the given type key directly from the stream.
 *  This function wraps the registry mechanism used to store type based deserializers
 *  for each runtime::Module sub-class.
 *
 * \param type_key The type key of the serialized module.
 * \param stream A pointer to the stream containing the serialized module.
 * \return module The deserialized module.
 */
Module LoadModuleFromBinary(const std::string& type_key, dmlc::Stream* stream);

/*!
 * \brief Library is the common interface
 *  for storing data in the form of shared libaries.
 *
 * \sa dso_library.cc
 * \sa system_library.cc
 */
class Library : public Object {
 public:
  // destructor.
  virtual ~Library() {}
  /*!
   * \brief Get the symbol address for a given name.
   * \param name The name of the symbol.
   * \return The symbol.
   */
  virtual void* GetSymbol(const char* name) = 0;
  // NOTE: we do not explicitly create an type index and type_key here for libary.
  // This is because we do not need dynamic type downcasting.
};

/*!
 * \brief Wrap a TVMBackendPackedCFunc to packed function.
 * \param faddr The function address
 * \param mptr The module pointer node.
 */
PackedFunc WrapPackedFunc(TVMBackendPackedCFunc faddr, const ObjectPtr<Object>& mptr);

/*!
 * \brief Utility to initialize conext function symbols during startup
 * \param fgetsymbol A symbol lookup function.
 */
void InitContextFunctions(std::function<void*(const char*)> fgetsymbol);

/*!
 * \brief Helper classes to get into internal of a module.
 */
class ModuleInternal {
 public:
  // Get mutable reference of imports.
  static std::vector<Module>* GetImportsAddr(ModuleNode* node) { return &(node->imports_); }
};

/*!
 * \brief Type alias for function to wrap a TVMBackendPackedCFunc.
 * \param The function address imported from a module.
 * \param mptr The module pointer node.
 * \return Packed function that wraps the invocation of the function at faddr.
 */
using PackedFuncWrapper =
    std::function<PackedFunc(TVMBackendPackedCFunc faddr, const ObjectPtr<Object>& mptr)>;

/*! \brief Return a library object interface over dynamic shared
 *  libraries in Windows and Linux providing support for
 *  loading/unloading and symbol lookup.
 *  \param Full path to shared library.
 *  \return Returns pointer to the Library providing symbol lookup.
 */
ObjectPtr<Library> CreateDSOLibraryObject(std::string library_path);

/*!
 * \brief Create a module from a library.
 *
 * \param lib The library.
 * \param wrapper Optional function used to wrap a TVMBackendPackedCFunc,
 * by default WrapPackedFunc is used.
 * \param symbol_prefix Optional symbol prefix that can be used to search alternative symbols.
 *
 * \return The corresponding loaded module.
 *
 * \note This function can create multiple linked modules
 *       by parsing the binary blob section of the library.
 */
Module CreateModuleFromLibrary(ObjectPtr<Library> lib, PackedFuncWrapper wrapper = WrapPackedFunc);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_LIBRARY_MODULE_H_
