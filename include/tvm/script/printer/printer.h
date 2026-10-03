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
 * \file tvm/script/printer/printer.h
 * \brief TVMScript text entry points.
 */
#ifndef TVM_SCRIPT_PRINTER_PRINTER_H_
#define TVM_SCRIPT_PRINTER_PRINTER_H_

#include <tvm/ffi/container/map.h>
#include <tvm/ffi/optional.h>
#include <tvm/script/printer/config.h>

#include <optional>
#include <string>

namespace tvm {

/*!
 * \brief Print an IR object as TVMScript, using repr when no translation hook exists.
 * \param node The input IR object.
 * \param config Optional translation and rendering configuration.
 * \return The rendered script or fallback representation.
 */
TVM_DLL std::string Script(const ffi::ObjectRef& node,
                           const ffi::Optional<PrinterConfig>& config = std::nullopt);

namespace script {
namespace printer {

/*!
 * \brief Register a namespace alias during dialect static initialization.
 * \param key The existing prefix configuration key, such as "tirx.prefix".
 * \param default_alias The alias reserved before translation assigns variable names.
 */
TVM_DLL void RegisterNamespaceAlias(const ffi::String& key, const ffi::String& default_alias);

/*! \brief Read the registered namespace aliases. */
TVM_DLL const ffi::Map<ffi::String, ffi::String>& GetNamespaceAliases();

/*!
 * \brief Translate IR, recover diagnostic paths, and render Python text.
 * \param obj The input IR object.
 * \param config The translation and rendering options.
 * \return The rendered script.
 */
TVM_DLL ffi::String Script(const ffi::ObjectRef& obj, const PrinterConfig& config);

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_PRINTER_H_
