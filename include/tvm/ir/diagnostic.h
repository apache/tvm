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
 * \file diagnostic.h
 * \brief A new diagnostic interface for TVM error reporting.
 *
 * A prototype of the new diagnostic reporting interface for TVM.
 *
 * Eventually we hope to promote this file to the top-level and
 * replace the existing errors.h.
 */

#ifndef TVM_IR_DIAGNOSTIC_H_
#define TVM_IR_DIAGNOSTIC_H_

#include <dmlc/logging.h>

namespace tvm {

extern const char* kTVM_INTERNAL_ERROR_MESSAGE;

#define ICHECK_INDENT "  "

#define ICHECK_BINARY_OP(name, op, x, y)                           \
  if (dmlc::LogCheckError _check_err = dmlc::LogCheck##name(x, y)) \
  dmlc::LogMessageFatal(__FILE__, __LINE__).stream()               \
      << kTVM_INTERNAL_ERROR_MESSAGE << std::endl                  \
      << ICHECK_INDENT << "Check failed: " << #x " " #op " " #y << *(_check_err.str) << ": "

#define ICHECK(x)                                    \
  if (!(x))                                          \
  dmlc::LogMessageFatal(__FILE__, __LINE__).stream() \
      << kTVM_INTERNAL_ERROR_MESSAGE << ICHECK_INDENT << "Check failed: " #x << " == false: "

#define ICHECK_LT(x, y) ICHECK_BINARY_OP(_LT, <, x, y)
#define ICHECK_GT(x, y) ICHECK_BINARY_OP(_GT, >, x, y)
#define ICHECK_LE(x, y) ICHECK_BINARY_OP(_LE, <=, x, y)
#define ICHECK_GE(x, y) ICHECK_BINARY_OP(_GE, >=, x, y)
#define ICHECK_EQ(x, y) ICHECK_BINARY_OP(_EQ, ==, x, y)
#define ICHECK_NE(x, y) ICHECK_BINARY_OP(_NE, !=, x, y)
#define ICHECK_NOTNULL(x)                                                                   \
  ((x) == nullptr ? dmlc::LogMessageFatal(__FILE__, __LINE__).stream()                      \
                        << kTVM_INTERNAL_ERROR_MESSAGE << __INDENT << "Check not null: " #x \
                        << ' ',                                                             \
   (x) : (x))  // NOLINT(*)

/*! \brief The diagnostic level, controls the printing of the message. */
enum class DiagnosticLevel : int {
  kBug = 10,
  kError = 20,
  kWarning = 30,
  kNote = 40,
  kHelp = 50,
};

}  // namespace tvm
#endif  // TVM_IR_DIAGNOSTIC_H_
