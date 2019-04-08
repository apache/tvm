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
 *  Copyright (c) 2017 by Contributors
 * \file channel.cc
 */
#include <tvm/channel.h>

namespace tvm {

Channel ChannelNode::make(Var handle_var, Type dtype) {
  auto n = make_node<ChannelNode>();
  n->handle_var = handle_var;
  n->dtype = dtype;
  return Channel(n);
}

TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<ChannelNode>([](const ChannelNode *op, IRPrinter *p) {
    p->stream << "channel(" << op->handle_var << ", " << op->dtype << ")";
});

TVM_REGISTER_NODE_TYPE(ChannelNode);
}  // namespace tvm
