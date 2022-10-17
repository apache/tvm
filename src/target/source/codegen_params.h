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
 * \file codegen_params.h
 */

#ifndef TVM_TARGET_SOURCE_CODEGEN_PARAMS_H_
#define TVM_TARGET_SOURCE_CODEGEN_PARAMS_H_

#include <tvm/runtime/ndarray.h>

#include <iostream>
#include <string>

namespace tvm {
namespace codegen {

/*!
 * \brief Write a C representation of arr to os.
 *
 * This function generates a comma-separated, indented list of C integer listeals suitable for use
 * in an initializer. The NDArray is flattened and then the list is produced element by element.
 * For the int16_t NDArray [-3, -2, -1, 0, 1, 2, 3, ...], and indent_chars = 4, the following output
 * is produced:
 *     -0x0003, -0x0002, -0x0001, +0x0000, +0x0001, +0x0002, +0x0003
 *
 * \param arr The array to generate
 * \param indent_chars Number of chars to indent
 * \param os Output stream where the array data should be written.
 */
void NDArrayDataToC(::tvm::runtime::NDArray arr, int indent_chars, std::ostream& os,
                    const std::string& eol = "\n");

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_PARAMS_H_
