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

#include "../../src/runtime/crt/include/tvm/runtime/crt/internal/graph_executor/graph_executor.h"

#include <gtest/gtest.h>

#include "../../src/runtime/crt/include/tvm/runtime/crt/internal/graph_executor/load_json.h"

namespace {

constexpr const char* kJson = R"(
{
  "nodes": [
    {
      "op": "null",
      "name": "x",
      "inputs": []
    },
    {
      "op": "null",
      "name": "p0",
      "inputs": []
    },
    {
      "op": "tvm_op",
      "name": "tvmgen_default_fused_add",
      "attrs": {
        "num_outputs": "1",
        "num_inputs": "2",
        "flatten_data": "0",
        "func_name": "tvmgen_default_fused_add",
        "hash": "a2b7e0a88031366c"
      },
      "inputs": [
        [
          0,
          0,
          0
        ],
        [
          1,
          0,
          0
        ]
      ]
    }
  ],
  "arg_nodes": [0, 1],
  "heads": [
    [
      2,
      0,
      0
    ]
  ],
  "attrs": {
    "dltype": [
      "list_str",
      [
        "float32",
        "float32",
        "float32"
      ]
    ],
    "device_index": [
      "list_int",
      [1, 1, 1]
    ],
    "storage_id": [
      "list_int",
      [0, 1, 2]
    ],
    "shape": [
      "list_shape",
      [
        [10, 5],
        [1, 5],
        [10, 5]
      ]
    ]
  },
  "node_row_ptr": [0, 1, 2, 3]
}
)";

// Check a JSON graph can be loaded.
TEST(TVMGraphExecutor_Load, Parse) {
  JSONReader reader;
  tvm_crt_error_t err = JSONReader_Create(kJson, &reader);
  EXPECT_EQ(err, kTvmErrorNoError);
  TVMGraphExecutor executor;
  memset(&executor, 0, sizeof(executor));
  int status = TVMGraphExecutor_Load(&executor, &reader);
  EXPECT_EQ(status, 0);
  EXPECT_EQ(executor.nodes_count, 3);
}

}  // namespace
