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
 * \file dso_module.h
 * \brief Abstraction over dynamic shared librariess in
 * Windows and Linux providing support for loading/unloading
 * and symbol lookup.
 */
#ifndef TVM_RUNTIME_DSO_LIBRARY_H_
#define TVM_RUNTIME_DSO_LIBRARY_H_

#include <tvm/runtime/module.h>

#include <string>

#if defined(_WIN32)
#include <windows.h>
#endif

#include "library_module.h"

namespace tvm {
namespace runtime {

/*!
 * \brief Dynamic shared library object used to load
 * and retrieve symbols by name. This is the default
 * module TVM uses for host-side AOT compilation.
 */
class DSOLibrary final : public Library {
 public:
  ~DSOLibrary();
  /*!
   * \brief Initialize by loading and storing
   * a handle to the underlying shared library.
   * \param name The string name/path to the
   * shared library over which to initialize.
   */
  void Init(const std::string& name);
  /*!
   * \brief Returns the symbol address within
   * the shared library for a given symbol name.
   * \param name The name of the symbol.
   * \return The symbol.
   */
  void* GetSymbol(const char* name) final;

 private:
  /*! \brief Private implementation of symbol lookup.
   *  Implementation is operating system dependent.
   *  \param The name of the symbol.
   * \return The symbol.
   */
  void* GetSymbol_(const char* name);
  /*! \brief Implementation of shared library load.
   *  Implementation is operating system dependent.
   *  \param The name/path of the shared library.
   */
  void Load(const std::string& name);
  /*! \brief Implementation of shared library unload.
   *  Implementation is operating system dependent.
   */
  void Unload();

#if defined(_WIN32)
  //! \brief Windows library handle
  HMODULE lib_handle_{nullptr};
#else
  // \brief Linux library handle
  void* lib_handle_{nullptr};
#endif
};
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_DSO_LIBRARY_H_
