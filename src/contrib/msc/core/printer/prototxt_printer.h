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
 * \file src/contrib/msc/core/printer/prototxt_printer.h
 * \brief Prototxt Printer.
 */

#ifndef TVM_CONTRIB_MSC_CORE_PRINTER_PROTOTXT_PRINTER_H_
#define TVM_CONTRIB_MSC_CORE_PRINTER_PROTOTXT_PRINTER_H_

#include <string>
#include <utility>
#include <vector>

#include "msc_base_printer.h"

namespace tvm {
namespace contrib {
namespace msc {

using namespace tvm::script::printer;

/*!
 * \brief PrototxtPrinter change list of dict to prototxt format
 * \sa Doc
 */
class PrototxtPrinter : public MSCBasePrinter {
 public:
  /*!
   * \brief The constructor of PrototxtPrinter
   * \param options the options for printer.
   */
  explicit PrototxtPrinter(const std::string& options = "") : MSCBasePrinter(options) {}

  /*! \brief Change object to LiteralDoc*/
  static LiteralDoc ToLiteralDoc(const ObjectRef& obj);

  /*! \brief Change map to DictDoc*/
  static DictDoc ToDictDoc(const Map<String, ObjectRef>& dict);

  /*! \brief Change ordered pairs to DictDoc*/
  static DictDoc ToDictDoc(const std::vector<std::pair<String, ObjectRef>>& dict);

  /*! \brief Append a map into the final content*/
  void Append(const Map<String, ObjectRef>& dict);

  /*! \brief Append ordered pairs into the final content*/
  void Append(const std::vector<std::pair<String, ObjectRef>>& dict);

  /*! \brief Append a map pair into the final content*/
  void AppendPair(const String& key, const ObjectRef& value);

 protected:
  /*! * \brief Print a DictDoc to prototxt format*/
  void PrintTypedDoc(const DictDoc& doc) final;
};

}  // namespace msc
}  // namespace contrib
}  // namespace tvm

#endif  // TVM_CONTRIB_MSC_CORE_PRINTER_PROTOTXT_PRINTER_H_
