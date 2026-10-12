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
 * \file tvm/script/printer/config.h
 * \brief Configuration object for the TVMScript printer.
 *
 * Contains PrinterConfig / PrinterConfigObj, GetBuiltinKeywords, GetExtraConfig,
 * and RedirectedReprPrinterMethod. Translation APIs live in doc_translator.h,
 * and text entry points live in printer.h.
 */
#ifndef TVM_SCRIPT_PRINTER_CONFIG_H_
#define TVM_SCRIPT_PRINTER_CONFIG_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/runtime/base.h>

#include <string>

namespace tvm {

class PrinterConfigObj : public ffi::Object {
 public:
  /*! \brief A stack that tracks the names of the binding hierarchy */
  ffi::Array<ffi::String> binding_names = {};
  /*! \brief Whether or not to show metadata. */
  bool show_meta = false;
  /*! \brief Whether or not to verbose print expressions. */
  bool verbose_expr = false;
  /*! \brief Number of spaces used for indentation*/
  int indent_spaces = 4;
  /*! \brief Whether to print line numbers */
  bool print_line_numbers = false;
  /*! \brief Number of context lines to print around the underlined text */
  int num_context_lines = -1;

  /*! \brief Whether to render access-path context for invisible underlined paths. Defaults true. */
  bool render_invisible_path_info = true;

  /* \brief ffi::Object path to be underlined */
  ffi::Array<ffi::reflection::AccessPath> path_to_underline;
  /*! \brief ffi::Object path to be annotated. */
  ffi::Map<ffi::reflection::AccessPath, ffi::String> path_to_annotate;
  /*! \brief ffi::Object to be underlined. */
  ffi::Array<ffi::ObjectRef> obj_to_underline = ffi::Array<ffi::ObjectRef>();
  /*! \brief ffi::Object to be annotated. */
  ffi::Map<ffi::ObjectRef, ffi::String> obj_to_annotate = ffi::Map<ffi::ObjectRef, ffi::String>();

  /*!
   * \brief Generic extension map for dialect-specific config knobs.
   *
   * Keys are conventionally namespaced as "<dialect>.<knob>", e.g.:
   *   "ir.prefix"                — the IR prefix (default "I")
   *   "tirx.prefix"              — the TIR prefix (default "T")
   *   "relax.prefix"             — the Relax prefix (default "R")
   *   "relax.show_all_ty" — whether to show all Relax type annotations (default true)
   *
   * Use GetExtraConfig<T>(key, fallback) to read values with a typed fallback.
   */
  ffi::Map<ffi::String, ffi::Any> extra_config;

  /*!
   * \brief Look up a value in extra_config with type cast and fallback.
   *
   * Keys are conventionally namespaced as "<dialect>.<knob>"
   * (e.g. "tirx.prefix", "relax.show_all_ty").
   */
  template <typename T>
  T GetExtraConfig(const ffi::String& key, T fallback) const {
    auto it = extra_config.find(key);
    if (it == extra_config.end()) return fallback;
    return (*it).second.as_or_throw<T>();
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PrinterConfigObj>()
        .def_ro("binding_names", &PrinterConfigObj::binding_names)
        .def_ro("show_meta", &PrinterConfigObj::show_meta)
        .def_ro("verbose_expr", &PrinterConfigObj::verbose_expr)
        .def_ro("indent_spaces", &PrinterConfigObj::indent_spaces)
        .def_ro("print_line_numbers", &PrinterConfigObj::print_line_numbers)
        .def_ro("num_context_lines", &PrinterConfigObj::num_context_lines)
        .def_ro("render_invisible_path_info", &PrinterConfigObj::render_invisible_path_info)
        .def_ro("path_to_underline", &PrinterConfigObj::path_to_underline)
        .def_ro("path_to_annotate", &PrinterConfigObj::path_to_annotate)
        .def_ro("obj_to_underline", &PrinterConfigObj::obj_to_underline)
        .def_ro("obj_to_annotate", &PrinterConfigObj::obj_to_annotate)
        .def_ro("extra_config", &PrinterConfigObj::extra_config);
  }

  ffi::Array<ffi::String> GetBuiltinKeywords();

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.PrinterConfig", PrinterConfigObj, ffi::Object);
};

class TVM_DLL PrinterConfig : public ffi::ObjectRef {
 public:
  explicit PrinterConfig(
      ffi::Map<ffi::String, ffi::Any> config_dict = ffi::Map<ffi::String, ffi::Any>());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PrinterConfig, ffi::ObjectRef, PrinterConfigObj);
};

/*!
 * \brief Render an IR object for its registered repr callback.
 *
 * Tries to translate \p obj directly; on error falls back to a plain
 * address string.  Defined in src/script/printer/script_printer.cc so that
 * <tvm/runtime/logging.h> is not pulled into this public header.
 */
TVM_DLL std::string RedirectedReprPrinterMethod(const ffi::ObjectRef& obj);

}  // namespace tvm
#endif  // TVM_SCRIPT_PRINTER_CONFIG_H_
