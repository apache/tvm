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

#include <tvm/ir/node_functor.h>
#include <tvm/script/printer/config.h>

namespace tvm {

/*! \brief Print \p node as TVMScript with the given \p config.
 *
 *  Falls back to ffi::ReprPrint for types not registered with TVMScriptPrinter.
 */
TVM_DLL std::string Script(const ffi::ObjectRef& node,
                           const ffi::Optional<PrinterConfig>& config = std::nullopt);

/*! \brief Dispatch vtable used by per-dialect printers to register their
 *         object-type printing functions.  Internal, but exposed here because
 *         TVM_REGISTER_SCRIPT_AS_REPR refers to it.
 */
class TVMScriptPrinter {
 public:
  using FType = NodeFunctor<std::string(const ffi::ObjectRef&, const PrinterConfig&)>;
  TVM_DLL static FType& vtable();
};

/*!
 * \brief Register Script as the kRepr callback for ObjectType and install
 *        the per-type dispatch entry in TVMScriptPrinter::vtable().
 *
 * \param ObjectType  The concrete object node type (e.g. tirx::VarNode).
 * \param Method      The TVMScriptPrinter vtable dispatch function.
 */
#define TVM_REGISTER_SCRIPT_AS_REPR(ObjectType, Method)                                        \
  TVM_FFI_STATIC_INIT_BLOCK() {                                                                \
    namespace refl = tvm::ffi::reflection;                                                     \
    refl::TypeAttrDef<ObjectType>().def(refl::type_attr::kRepr,                                \
                                        [](ffi::ObjectRef obj, ffi::Function) -> ffi::String { \
                                          return RedirectedReprPrinterMethod(obj);             \
                                        });                                                    \
  }                                                                                            \
  TVM_STATIC_IR_FUNCTOR(TVMScriptPrinter, vtable).set_dispatch<ObjectType>(Method)

}  // namespace tvm
#endif  // TVM_SCRIPT_PRINTER_PRINTER_H_
