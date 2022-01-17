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
#include "block_config.h"

#include <tvm/runtime/container/array.h>
#include <tvm/runtime/object.h>
#include <tvm/runtime/registry.h>

#include <utility>
#include <vector>

#include "common.h"

namespace tvm {
namespace contrib {
namespace ethosu {
namespace cascader {

void BlockConfigNode::VisitAttrs(AttrVisitor* v) {
  Array<Integer> tmp_arr = make_array(output_shape_);
  v->Visit("_output_shape", &tmp_arr);
}

BlockConfig::BlockConfig(const std::vector<int>& output_shape, int compute_cycles,
                         int output_cycles) {
  auto n = make_object<BlockConfigNode>();
  n->output_shape_ = std::move(output_shape);
  n->compute_cycles_ = compute_cycles;
  n->output_cycles_ = output_cycles;
  data_ = std::move(n);
}

TVM_REGISTER_GLOBAL("contrib.ethosu.cascader.BlockConfig")
    .set_body_typed([](Array<Integer> output_shape, int compute_cycles, int output_cycles) {
      std::vector<int> voutput_shape = make_vector<int, Integer>(output_shape);
      return BlockConfig(voutput_shape, compute_cycles, output_cycles);
    });

TVM_REGISTER_NODE_TYPE(BlockConfigNode);

}  // namespace cascader
}  // namespace ethosu
}  // namespace contrib
}  // namespace tvm
