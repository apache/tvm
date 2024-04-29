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
 * \file regex.h
 * \brief Exposes calls to python's `re` library.
 */
#ifndef TVM_RUNTIME_REGEX_H_
#define TVM_RUNTIME_REGEX_H_

#include <string>

namespace tvm {
namespace runtime {

/* \brief Check if a pattern matches a regular expression
 *
 * This function should be used instead of `std::regex` within C++
 * call sites, to avoid ABI incompatibilities with pytorch.
 *
 * Currently, the pytorch wheels available through pip install use
 * the pre-C++11 ABI by setting `-DUSE_CXX11_ABI=0` [0]. If TVM were to
 * user the pre-C++11 ABI, this would cause breakages with
 * dynamically-linked LLVM environments.
 *
 * Use of the `<regex>` header in TVM should be avoided, as its
 * implementation is not supported by gcc's dual ABI. This ABI
 * incompatibility results in runtime errors either when `std::regex`
 * is called from TVM, or when `std::regex` is called from pytorch,
 * depending on which library was loaded first.  This restriction can
 * be removed when a version of pytorch compiled using
 * `-DUSE_CXX11_ABI=1` is available from PyPI.
 *
 * [0] https://github.com/pytorch/pytorch/issues/51039
 *
 * \param match_against The string against which to match the regular expression
 *
 * \param regex_pattern The regular expression
 *
 * \returns match_result True if `match_against` matches the pattern
 *     defined by `regex_pattern`, and False otherwise.
 */

bool regex_match(const std::string& match_against, const std::string& regex_pattern);

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_REGEX_H_
