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

#include "tensorflow/core/framework/op.h"

REGISTER_OP("TvmDsoOp")
    .Input("input_args: ListT")
    .Attr(
        "ListT: list({float16, float32, float64, int8, int16, int32, int64, uint8, uint16,"
        "uint32, uint64})")
    .Input("dynamic_output_shape: int64")
    .Output("output: output_dtype")
    .Attr("lib_path: string")
    .Attr("func_name: string")
    .Attr(
        "output_dtype: {float16, float32, float64, int8, int16, int32, int64, uint8, uint16,"
        "uint32, uint64} = DT_FLOAT")
    .Attr("static_output_shape: list(int) >= 0 = []")
    .Attr("has_static_output_shape: bool");
