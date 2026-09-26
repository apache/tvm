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
 * \brief Entry-point header for TVMScript printing.
 *
 * Declares the free function `tvm::Script(node, optional_config)` and the
 * dispatch vtable `TVMScriptPrinter::vtable()` used by per-dialect printers.
 * `PrinterConfig` and its dataclass helpers live in config.h; this header is
 * what callers include to invoke printing.
 */
#ifndef TVM_SCRIPT_PRINTER_PRINTER_H_
#define TVM_SCRIPT_PRINTER_PRINTER_H_

#include <tvm/ir/object_functor.h>
#include <tvm/script/printer/config.h>

namespace tvm {

/*! \brief Print \p node as TVMScript with the given \p config.
 *
 *  Falls back to ffi::ReprPrint for types not registered with TVMScriptPrinter.
 */
TVM_DLL std::string Script(const ffi::ObjectRef& node,
                           const ffi::Optional<PrinterConfig>& config = std::nullopt);

/*! \brief Dispatch table for TVMScript printing and repr registration. */
class TVMScriptPrinter {
 public:
  using FType = ObjectFunctor<std::string(const ffi::ObjectRef&, const PrinterConfig&)>;
  TVM_DLL static FType& vtable();

  /*! \brief Register a printer method for script dispatch and FFI repr.
   * \tparam ObjectType Concrete object node type.
   * \tparam Method Callable printer method type.
   * \param method Printer dispatch method for that type.
   *
   * Example:
   * \code
   * TVM_FFI_STATIC_INIT_BLOCK() {
   *   TVMScriptPrinter::Register<tirx::ForNode>(ReprPrintTIR);
   *   TVMScriptPrinter::Register<tirx::WhileNode>(ReprPrintTIR);
   * }
   * \endcode
   */
  template <typename ObjectType, typename Method>
  static void Register(Method method) {
    namespace refl = tvm::ffi::reflection;
    refl::TypeAttrDef<ObjectType>().def(refl::type_attr::kRepr,
                                        [](ffi::ObjectRef obj, ffi::Function) -> ffi::String {
                                          return RedirectedReprPrinterMethod(obj);
                                        });
    vtable().SetDispatch<ObjectType>(method);
  }
};

}  // namespace tvm
#endif  // TVM_SCRIPT_PRINTER_PRINTER_H_
