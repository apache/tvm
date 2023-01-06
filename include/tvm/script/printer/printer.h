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
#ifndef TVM_SCRIPT_PRINTER_PRINTER_H_
#define TVM_SCRIPT_PRINTER_PRINTER_H_

#include <tvm/node/node.h>
#include <tvm/script/printer/ir_docsifier.h>

#include <unordered_map>
#include <vector>

namespace tvm {
namespace script {
namespace printer {

/*! \brief Default values in the TVMScript printer */
struct Default {
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
  /*! \brief Returns a singleton of the configuration */
  static Default* Instance();
  static DataType& BufferDType() { return Instance()->buffer_dtype; }
  static DataType& IntDType() { return Instance()->int_dtype; }
  static DataType& FloatDType() { return Instance()->float_dtype; }
};

/*!
 * \brief The entry method for TVMScript printing
 * \param obj The object to be printed
 * \param ir_prefix The prefix of IR nodes
 * \param indent_spaces Number of spaces used for indentation
 * \param print_line_numbers Whether to print line numbers
 * \param num_context_lines Number of context lines to print around the underlined text
 * \param path_to_underline Object path to be underlined
 * \return The TVMScript text format
 */
String Script(ObjectRef obj,                                                //
              Map<String, String> ir_prefix = {{"ir", "I"}, {"tir", "T"}},  //
              int indent_spaces = 4,                                        //
              bool print_line_numbers = false,                              //
              int num_context_lines = -1,                                   //
              Optional<ObjectPath> path_to_underline = NullOpt);

/*!
 * \brief Convert Doc into Python script.
 * \param doc Doc to be converted
 * \param indent_spaces Number of spaces used for indentation
 * \param print_line_numbers Whether to print line numbers
 * \param num_context_lines Number of context lines to print around the underlined text
 * \param path_to_underline Object path to be underlined
 */
String DocToPythonScript(Doc doc,                          //
                         int indent_spaces = 4,            //
                         bool print_line_numbers = false,  //
                         int num_context_lines = -1,       //
                         Optional<ObjectPath> path_to_underline = NullOpt);

}  // namespace printer
}  // namespace script
}  // namespace tvm

#endif  // TVM_SCRIPT_PRINTER_PRINTER_H_
