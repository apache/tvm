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
 * \file src/relay/qnn/op/unary_elementwise_op.cc
 * \brief QNN unary elementwise operators.
 */

#include "op_common.h"

namespace tvm {
namespace relay {
namespace qnn {

QNN_CREATE_UNARY_ELEMENTWISE_OP("asinh");
QNN_CREATE_UNARY_ELEMENTWISE_OP("acosh");
QNN_CREATE_UNARY_ELEMENTWISE_OP("atanh");

QNN_CREATE_UNARY_ELEMENTWISE_OP("sinh");
QNN_CREATE_UNARY_ELEMENTWISE_OP("cosh");
QNN_CREATE_UNARY_ELEMENTWISE_OP("tanh");

QNN_CREATE_UNARY_ELEMENTWISE_OP("sin");
QNN_CREATE_UNARY_ELEMENTWISE_OP("cos");
QNN_CREATE_UNARY_ELEMENTWISE_OP("tan");

QNN_CREATE_UNARY_ELEMENTWISE_OP("exp");
QNN_CREATE_UNARY_ELEMENTWISE_OP("log10");
QNN_CREATE_UNARY_ELEMENTWISE_OP("log2");

QNN_CREATE_UNARY_ELEMENTWISE_OP("sqrt");
QNN_CREATE_UNARY_ELEMENTWISE_OP("rsqrt");
QNN_CREATE_UNARY_ELEMENTWISE_OP("erf");
QNN_CREATE_UNARY_ELEMENTWISE_OP("sigmoid");

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
