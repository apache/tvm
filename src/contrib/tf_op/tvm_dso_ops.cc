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

using namespace tensorflow;

#define REGISTER_TFTVM_OP(n) REGISTER_OP("TvmDsoOp" #n) \
    .Output("output: output_dtype") \
    .Attr("lib_path: string") \
    .Attr("func_name: string") \
    .Attr("output_dtype: {int32, int64, float} = DT_FLOAT") \
    .Attr("static_output_shape: list(int) >= 0 = []") \
    .Attr("has_static_output_shape: bool") \


REGISTER_TFTVM_OP(1)
    .Input("input: T").Attr("T: type") \
    .Input("dynamic_output_shape: int64");

REGISTER_TFTVM_OP(2)
    .Input("input1: T1").Attr("T1: type")
    .Input("input2: T2").Attr("T2: type")
    .Input("dynamic_output_shape: int64");

REGISTER_TFTVM_OP(3)
    .Input("input1: T1").Attr("T1: type")
    .Input("input2: T2").Attr("T2: type")
    .Input("input3: T3").Attr("T3: type")
    .Input("dynamic_output_shape: int64");

REGISTER_TFTVM_OP(4)
    .Input("input1: T1").Attr("T1: type")
    .Input("input2: T2").Attr("T2: type")
    .Input("input3: T3").Attr("T3: type")
    .Input("input4: T4").Attr("T4: type")
    .Input("dynamic_output_shape: int64");

REGISTER_TFTVM_OP(5)
    .Input("input1: T1").Attr("T1: type")
    .Input("input2: T2").Attr("T2: type")
    .Input("input3: T3").Attr("T3: type")
    .Input("input4: T4").Attr("T4: type")
    .Input("input5: T5").Attr("T5: type")
    .Input("dynamic_output_shape: int64");

REGISTER_TFTVM_OP(6)
    .Input("input1: T1").Attr("T1: type")
    .Input("input2: T2").Attr("T2: type")
    .Input("input3: T3").Attr("T3: type")
    .Input("input4: T4").Attr("T4: type")
    .Input("input5: T5").Attr("T5: type")
    .Input("input6: T6").Attr("T6: type")
    .Input("dynamic_output_shape: int64");

REGISTER_TFTVM_OP(7)
    .Input("input1: T1").Attr("T1: type")
    .Input("input2: T2").Attr("T2: type")
    .Input("input3: T3").Attr("T3: type")
    .Input("input4: T4").Attr("T4: type")
    .Input("input5: T5").Attr("T5: type")
    .Input("input6: T6").Attr("T6: type")
    .Input("input7: T7").Attr("T7: type")
    .Input("dynamic_output_shape: int64");

REGISTER_TFTVM_OP(8)
    .Input("input1: T1").Attr("T1: type")
    .Input("input2: T2").Attr("T2: type")
    .Input("input3: T3").Attr("T3: type")
    .Input("input4: T4").Attr("T4: type")
    .Input("input5: T5").Attr("T5: type")
    .Input("input6: T6").Attr("T6: type")
    .Input("input7: T7").Attr("T7: type")
    .Input("input8: T8").Attr("T8: type")
    .Input("dynamic_output_shape: int64");
