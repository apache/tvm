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

  void VisitAttrs(AttrVisitor* v) {
    v->Visit("binding_names", &binding_names);
    v->Visit("show_meta", &show_meta);
    v->Visit("ir_prefix", &ir_prefix);
    v->Visit("tir_prefix", &tir_prefix);
    v->Visit("relax_prefix", &relax_prefix);
    v->Visit("module_alias", &module_alias);
    v->Visit("buffer_dtype", &buffer_dtype);
    v->Visit("int_dtype", &int_dtype);
    v->Visit("float_dtype", &float_dtype);
    v->Visit("verbose_expr", &verbose_expr);
    v->Visit("indent_spaces", &indent_spaces);
    v->Visit("print_line_numbers", &print_line_numbers);
    v->Visit("num_context_lines", &num_context_lines);
    v->Visit("syntax_sugar", &syntax_sugar);
    v->Visit("show_object_address", &show_object_address);
    v->Visit("show_all_struct_info", &show_all_struct_info);
    v->Visit("path_to_underline", &path_to_underline);
    v->Visit("path_to_annotate", &path_to_annotate);
    v->Visit("obj_to_underline", &obj_to_underline);
    v->Visit("obj_to_annotate", &obj_to_annotate);
  }

  Array<String> GetBuiltinKeywords();

  static constexpr const char* _type_key = "node.PrinterConfig";
  TVM_DECLARE_FINAL_OBJECT_INFO(PrinterConfigNode, Object);
};

class PrinterConfig : public ObjectRef {
 public:
  explicit PrinterConfig(Map<String, ObjectRef> config_dict = Map<String, ObjectRef>());

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
  std::string Script(const Optional<PrinterConfig>& config = NullOpt) const {                   \
    return TVMScriptPrinter::Script(GetRef<ObjectRef>(this), config.value_or(PrinterConfig())); \
  }

}  // namespace tvm
#endif  // TVM_NODE_SCRIPT_PRINTER_H_
