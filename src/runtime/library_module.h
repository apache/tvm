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

#include <tvm/runtime/module.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>
#include <functional>

namespace tvm {
namespace runtime {
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
  virtual void *GetSymbol(const char* name) = 0;
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
 * \brief Create a module from a library.
 *
 * \param lib The library.
 * \return The corresponding loaded module.
 *
 * \note This function can create multiple linked modules
 *       by parsing the binary blob section of the library.
 */
Module CreateModuleFromLibrary(ObjectPtr<Library> lib);
}  // namespace runtime
}  // namespace tvm
#endif   // TVM_RUNTIME_LIBRARY_MODULE_H_
