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
 *
 * \file src/relay/op/nn/upsampling.h
 * \brief implementation of the InferCorrectLayout pass for upsampling
 */

#ifndef TVM_RELAY_OP_NN_UPSAMPLING_H_
#define TVM_RELAY_OP_NN_UPSAMPLING_H_

#include <tvm/relay/attrs/nn.h>
#include <tvm/tir/data_layout.h>

#include "../op_common.h"

namespace tvm {
namespace relay {

template <typename T>
InferCorrectLayoutOutput UpsamplingInferCorrectLayout(const Attrs& attrs,
                                                      const Array<Layout>& new_in_layouts,
                                                      const Array<Layout>& old_in_layouts,
                                                      const Array<tvm::relay::Type>& old_in_types) {
  const auto* attrs_ptr = attrs.as<T>();
  ICHECK(attrs_ptr);
  ObjectPtr<T> params = make_object<T>(*attrs_ptr);

  if (new_in_layouts.defined()) {
    ICHECK_EQ(new_in_layouts.size(), 1);

    Layout raw_layout(params->layout);
    Layout input = new_in_layouts[0];
    if (input.IndexOf(LayoutAxis::Get('W')) == raw_layout.IndexOf(LayoutAxis::Get('W')) &&
        input.IndexOf(LayoutAxis::Get('H')) == raw_layout.IndexOf(LayoutAxis::Get('H')) &&
        !input.Contains(LayoutAxis::Get('w')) && !input.Contains(LayoutAxis::Get('h')) &&
        (input.IndexOf(LayoutAxis::Get('D')) == -1 ||
         (input.IndexOf(LayoutAxis::Get('D')) == raw_layout.IndexOf(LayoutAxis::Get('D')) &&
          !input.Contains(LayoutAxis::Get('d'))))) {
      params->layout = input.name();  // modify self to follow the input layout
    }
  }

  return InferCorrectLayoutOutput({params->layout}, {params->layout}, Attrs(params));
}

}  // namespace relay
}  // namespace tvm

#endif  // TVM_RELAY_OP_NN_UPSAMPLING_H_
