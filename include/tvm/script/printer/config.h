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
 * Contains PrinterConfig / PrinterConfigNode, GetBuiltinKeywords, GetExtraConfig,
 * and RedirectedReprPrinterMethod.  The entry-point free function tvm::Script()
 * and the dispatch vtable TVMScriptPrinter live in printer.h.
 */
#ifndef TVM_SCRIPT_PRINTER_CONFIG_H_
#define TVM_SCRIPT_PRINTER_CONFIG_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/data_type.h>

#include <string>

namespace tvm {

class PrinterConfigNode : public ffi::Object {
 public:
  /*! \brief A stack that tracks the names of the binding hierarchy */
  ffi::Array<ffi::String> binding_names = {};
  /*! \brief Whether or not to show metadata. */
  bool show_meta = false;
  /*! \brief The prefix of IR nodes */
  ffi::String ir_prefix = "I";
  /*!
   * \brief The alias of the current module at cross-function call
   * \note Directly use module name if it's empty.
   */
  ffi::String module_alias = "cls";
  /*! \brief Default buffer dtype */
  DataType buffer_dtype = DataType::Float(32);
  /*! \brief Default data type of integer literals */
  DataType int_dtype = DataType::Int(32);
  /*!
   * \brief Default data type of float literals. Right now we always print out the explicit type
   * of floating point values, so setting it to Void means we do not print without the
   * T.float32/T.float64 wrapper.
   */
  DataType float_dtype = DataType::Void();
  /*! \brief Whether or not to verbose print expressions. */
  bool verbose_expr = false;
  /*! \brief Number of spaces used for indentation*/
  int indent_spaces = 4;
  /*! \brief Whether to print line numbers */
  bool print_line_numbers = false;
  /*! \brief Number of context lines to print around the underlined text */
  int num_context_lines = -1;
  /*! \brief Whether to output with syntax sugar, set false for complete printing. */
  bool syntax_sugar = true;
  /*! \brief Whether variable names should include the object's address */
  bool show_object_address = false;

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
   *   "tirx.prefix"              — the TIR prefix (default "T")
   *   "relax.prefix"             — the Relax prefix (default "R")
   *   "relax.show_all_ty" — whether to show all struct info (default true)
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
    return Downcast<T>((*it).second);
  }

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PrinterConfigNode>()
        .def_ro("binding_names", &PrinterConfigNode::binding_names)
        .def_ro("show_meta", &PrinterConfigNode::show_meta)
        .def_ro("ir_prefix", &PrinterConfigNode::ir_prefix)
        .def_ro("module_alias", &PrinterConfigNode::module_alias)
        .def_ro("buffer_dtype", &PrinterConfigNode::buffer_dtype)
        .def_ro("int_dtype", &PrinterConfigNode::int_dtype)
        .def_ro("float_dtype", &PrinterConfigNode::float_dtype)
        .def_ro("verbose_expr", &PrinterConfigNode::verbose_expr)
        .def_ro("indent_spaces", &PrinterConfigNode::indent_spaces)
        .def_ro("print_line_numbers", &PrinterConfigNode::print_line_numbers)
        .def_ro("num_context_lines", &PrinterConfigNode::num_context_lines)
        .def_ro("syntax_sugar", &PrinterConfigNode::syntax_sugar)
        .def_ro("show_object_address", &PrinterConfigNode::show_object_address)
        .def_ro("path_to_underline", &PrinterConfigNode::path_to_underline)
        .def_ro("path_to_annotate", &PrinterConfigNode::path_to_annotate)
        .def_ro("obj_to_underline", &PrinterConfigNode::obj_to_underline)
        .def_ro("obj_to_annotate", &PrinterConfigNode::obj_to_annotate)
        .def_ro("extra_config", &PrinterConfigNode::extra_config);
  }

  ffi::Array<ffi::String> GetBuiltinKeywords();

  static constexpr const bool _type_mutable = true;
  TVM_FFI_DECLARE_OBJECT_INFO_FINAL("script.PrinterConfig", PrinterConfigNode, ffi::Object);
};

class TVM_DLL PrinterConfig : public ffi::ObjectRef {
 public:
  explicit PrinterConfig(
      ffi::Map<ffi::String, ffi::Any> config_dict = ffi::Map<ffi::String, ffi::Any>());

  TVM_FFI_DEFINE_OBJECT_REF_METHODS_NOTNULLABLE(PrinterConfig, ffi::ObjectRef, PrinterConfigNode);
};

/*!
 * \brief The fallback body used by TVM_REGISTER_SCRIPT_AS_REPR (defined in printer.h).
 *
 * Tries to format \p obj via tvm::Script; on error falls back to a plain
 * address string.  Defined in src/script/printer/config.cc so that
 * <tvm/runtime/logging.h> is not pulled into this public header.
 */
TVM_DLL std::string RedirectedReprPrinterMethod(const ffi::ObjectRef& obj);

}  // namespace tvm
#endif  // TVM_SCRIPT_PRINTER_CONFIG_H_
