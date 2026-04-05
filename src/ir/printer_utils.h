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
#ifndef TVM_IR_PRINTER_UTILS_H_
#define TVM_IR_PRINTER_UTILS_H_

#include <tvm/ffi/extra/ir_traits.h>
#include <tvm/ffi/extra/pyast.h>
#include <tvm/ffi/reflection/access_path.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/dtype.h>

namespace tvm {
namespace printer {

namespace text = ::tvm::ffi::pyast;
namespace tr = ::tvm::ffi::ir_traits;
namespace refl = ::tvm::ffi::reflection;

/*! \brief Convert a DataType to its string representation. */
inline std::string DType2Str(const runtime::DataType& dtype) {
  return dtype.is_void() ? "void" : runtime::DLDataTypeToString(dtype);
}

/*! \brief Build `prefix.attr` as an IdAST with dot notation. */
inline text::ExprAST PrefixedId(const std::string& prefix, const std::string& attr) {
  return text::ExprAttr(text::IdAST(prefix), attr);
}

/*! \brief Build `T.attr` */
inline text::ExprAST TIR(const std::string& attr) { return PrefixedId("T", attr); }

/*! \brief Build `R.attr` */
inline text::ExprAST Relax(const std::string& attr) { return PrefixedId("R", attr); }

/*! \brief Build `I.attr` */
inline text::ExprAST IR(const std::string& attr) { return PrefixedId("I", attr); }

/*! \brief Print an object through the printer dispatch. */
inline text::ExprAST Print(const text::IRPrinter& printer, ffi::Any obj, text::AccessPath path) {
  return printer->operator()(std::move(obj), std::move(path)).cast<text::ExprAST>();
}


}  // namespace printer
}  // namespace tvm

#endif  // TVM_IR_PRINTER_UTILS_H_
