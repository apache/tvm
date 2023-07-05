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

QNN_CREATE_UNARY_ELEMENTWISE_OP("tanh").set_attr<FTVMLegalize>(
    "FTVMQnnCanonicalize", QNN_UNARY_OP_DEFAULT_CANONICALIZATION(Tanh));

QNN_CREATE_UNARY_ELEMENTWISE_OP("exp").set_attr<FTVMLegalize>(
    "FTVMQnnCanonicalize", QNN_UNARY_OP_DEFAULT_CANONICALIZATION(Exp));

QNN_CREATE_UNARY_ELEMENTWISE_OP("sqrt").set_attr<FTVMLegalize>(
    "FTVMQnnCanonicalize", QNN_UNARY_OP_DEFAULT_CANONICALIZATION(Sqrt));

QNN_CREATE_UNARY_ELEMENTWISE_OP("rsqrt").set_attr<FTVMLegalize>(
    "FTVMQnnCanonicalize", QNN_UNARY_OP_DEFAULT_CANONICALIZATION(Rsqrt));

QNN_CREATE_UNARY_ELEMENTWISE_OP("erf").set_attr<FTVMLegalize>(
    "FTVMQnnCanonicalize", QNN_UNARY_OP_DEFAULT_CANONICALIZATION(Erf));

QNN_CREATE_UNARY_ELEMENTWISE_OP("sigmoid").set_attr<FTVMLegalize>(
    "FTVMQnnCanonicalize", QNN_UNARY_OP_DEFAULT_CANONICALIZATION(Sigmoid));

QNN_CREATE_UNARY_ELEMENTWISE_OP("hardswish")
    .set_attr<FTVMLegalize>("FTVMQnnCanonicalize",
                            QNN_UNARY_OP_DEFAULT_CANONICALIZATION(Hardswish));

QNN_CREATE_UNARY_ELEMENTWISE_OP("log").set_attr<FTVMLegalize>(
    "FTVMQnnCanonicalize", QNN_UNARY_OP_DEFAULT_CANONICALIZATION(Log));

QNN_CREATE_UNARY_ELEMENTWISE_OP("abs").set_attr<FTVMLegalize>(
    "FTVMQnnCanonicalize", QNN_UNARY_OP_DEFAULT_CANONICALIZATION(Abs));

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
