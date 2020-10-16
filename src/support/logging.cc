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
 * \file src/support/logging.cc
 * \brief Definition of logging variables.
 */

namespace tvm {

const char* kTVM_INTERNAL_ERROR_MESSAGE =
    "\n---------------------------------------------------------------\n"
    "An internal invariant was violated during the execution of TVM.\n"
    "Please read TVM's error reporting guidelines.\n"
    "More details can be found here: https://discuss.tvm.ai/t/error-reporting/7793.\n"
    "---------------------------------------------------------------\n";

}  // namespace tvm
