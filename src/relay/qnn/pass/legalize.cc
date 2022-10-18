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
 * \file relay/qnn/pass/legalize.cc
 * \brief The Legalize wrapper for QNN.
 */

#include <tvm/relay/qnn/transform.h>

namespace tvm {
namespace relay {
namespace qnn {

namespace transform {

Pass Legalize() {
  Array<Pass> pass_seqs;
  pass_seqs.push_back(relay::transform::Legalize("FTVMQnnLegalize"));
  pass_seqs.push_back(relay::transform::Legalize("FTVMQnnCanonicalize"));
  relay::transform::Pass seq = relay::transform::Sequential(pass_seqs, "qnn.Legalize");
  return seq;
}

TVM_REGISTER_GLOBAL("relay.qnn._transform.Legalize").set_body_typed(Legalize);

}  // namespace transform

}  // namespace qnn
}  // namespace relay
}  // namespace tvm
