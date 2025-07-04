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
 * \file tvm/node/repr_printer.h
 * \brief Printer class to print repr string of each AST/IR nodes.
 */
#ifndef TVM_NODE_SCRIPT_PRINTER_H_
#define TVM_NODE_SCRIPT_PRINTER_H_

#include <tvm/ffi/any.h>
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/map.h>
#include <tvm/ffi/reflection/reflection.h>
#include <tvm/ffi/string.h>
#include <tvm/node/functor.h>
#include <tvm/node/object_path.h>
#include <tvm/node/reflection.h>
#include <tvm/runtime/data_type.h>

#include <iostream>
#include <string>

namespace tvm {

class PrinterConfigNode : public Object {
 public:
  /*! \brief A stack that tracks the names of the binding hierarchy */
  Array<String> binding_names = {};
  /*! \brief Whether or not to show metadata. */
  bool show_meta = false;
  /*! \brief The prefix of IR nodes */
  std::string ir_prefix = "I";
  /*! \brief The prefix of TIR nodes */
  std::string tir_prefix = "T";
  /*! \brief The prefix of Relax nodes */
  std::string relax_prefix = "R";
  /*!
   * \brief The alias of the current module at cross-function call
   * \note Directly use module name if it's empty.
   */
  std::string module_alias = "cls";
  /*! \brief Default data type of TIR buffer */
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

  /*! \brief In Relax, whether to show all StructInfo annotations
   *
   * If true (default), all variable bindings will be annotated with
   * the struct info of the variable being bound.
   *
   * If false, the annotations will only be shown when they are
   * required for correct parsing of the Relax function.  For example,
   * function parameters must always have struct info annotations, but
   * the struct info for expressions within a function body may be inferred from their
   * arguments, and are therefore
   *
   * Example:
   *
   * \code{.py}
   *     # func.show(show_all_struct_info=True)
   *     @R.function
   *     def func(
   *         A: R.Tensor((10, 20), dtype="float32"),
   *         B: R.Tensor((10,20), dtype="float32"),
   *     ) -> R.Tensor((10, 20), dtype="float32"):
   *         C: R.Tensor((10,20), dtype="float32") = R.add(A, B2)
   *         return C
   *
   *     # func.show(show_all_struct_info=False)
   *     @R.function
   *     def func(
   *         A: R.Tensor((10, 20), dtype="float32"),
   *         B: R.Tensor((10,20), dtype="float32"),
   *     ) -> R.Tensor((10, 20), dtype="float32"):
   *         C = R.add(A, B2)
   *         return C
   * \endcode
   */
  bool show_all_struct_info = true;

  /* \brief Object path to be underlined */
  Array<ObjectPath> path_to_underline = Array<ObjectPath>();
  /*! \brief Object path to be annotated. */
  Map<ObjectPath, String> path_to_annotate = Map<ObjectPath, String>();
  /*! \brief Object to be underlined. */
  Array<ObjectRef> obj_to_underline = Array<ObjectRef>();
  /*! \brief Object to be annotated. */
  Map<ObjectRef, String> obj_to_annotate = Map<ObjectRef, String>();

  static void RegisterReflection() {
    namespace refl = tvm::ffi::reflection;
    refl::ObjectDef<PrinterConfigNode>()
        .def_ro("binding_names", &PrinterConfigNode::binding_names)
        .def_ro("show_meta", &PrinterConfigNode::show_meta)
        .def_ro("ir_prefix", &PrinterConfigNode::ir_prefix)
        .def_ro("tir_prefix", &PrinterConfigNode::tir_prefix)
        .def_ro("relax_prefix", &PrinterConfigNode::relax_prefix)
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
        .def_ro("show_all_struct_info", &PrinterConfigNode::show_all_struct_info)
        .def_ro("path_to_underline", &PrinterConfigNode::path_to_underline)
        .def_ro("path_to_annotate", &PrinterConfigNode::path_to_annotate)
        .def_ro("obj_to_underline", &PrinterConfigNode::obj_to_underline)
        .def_ro("obj_to_annotate", &PrinterConfigNode::obj_to_annotate);
  }

  Array<String> GetBuiltinKeywords();

  static constexpr const char* _type_key = "script.PrinterConfig";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrinterConfigNode, Object);
};

class PrinterConfig : public ObjectRef {
 public:
  explicit PrinterConfig(Map<String, ffi::Any> config_dict = Map<String, ffi::Any>());

  TVM_DEFINE_MUTABLE_NOTNULLABLE_OBJECT_REF_METHODS(PrinterConfig, runtime::ObjectRef,
                                                    PrinterConfigNode);
};

/*! \brief Legacy behavior of ReprPrinter. */
class TVMScriptPrinter {
 public:
  /* Convert the object to TVMScript format */
  static std::string Script(const ObjectRef& node, const Optional<PrinterConfig>& cfg);
  // Allow registration to be printer.
  using FType = NodeFunctor<std::string(const ObjectRef&, const PrinterConfig&)>;
  TVM_DLL static FType& vtable();
};

#define TVM_OBJECT_ENABLE_SCRIPT_PRINTER()                                                      \
  std::string Script(const Optional<PrinterConfig>& config = std::nullopt) const {              \
    return TVMScriptPrinter::Script(GetRef<ObjectRef>(this), config.value_or(PrinterConfig())); \
  }

}  // namespace tvm
#endif  // TVM_NODE_SCRIPT_PRINTER_H_
