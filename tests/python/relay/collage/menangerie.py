# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""A collection of Relay models for exercising Collage."""

import tvm
import onnx
import numpy as np
import logging
import tvm.contrib.target.onnx

MODEL_PREFIX = "/home/mbs/gauntlet/models/"
MNIST = {
    "name": "mnist",
    "filename": "mnist-8.onnx",
    "input_shapes": {"Input3": [1, 1, 28, 28]},
    "input_dtypes": {"Input3": "float32"},
    "main_dtype": "float32",
}
GPT2 = {
    "name": "gpt2",
    "filename": "gpt2.onnx",
    "input_shapes": {"input1": [1, 50, 32]},
    "input_dtypes": {"input1": "int64"},
    "main_dtype": "float32",
}
RESNET50V2 = {
    "name": "resnet50",
    "filename": "resnet50-v2-7.onnx",
    "input_shapes": {"data": [1, 3, 224, 224]},
    "input_dtypes": {"data": "float32"},
    "main_dtype": "float32",
}
MOBILENETV2 = {
    "name": "mobilenet",
    "filename": "mobilenetv2-1.0.onnx",
    "input_shapes": {"data": [1, 3, 224, 224]},
    "input_dtypes": {"data": "float32"},
    "main_dtype": "float32",
}
# Note that resnext50_32_4d below was extracted directly from the pytorch model and not from any onnx file.
RESNEXT50_32_4d = {
    "name": "resnext50_32_4d",
    "filename": "resnext50_32x4d.onnx",
    "input_shapes": {"x": [1, 64, 56, 56]},
    "input_dtypes": {"x": "float32"},
    "main_dtype": "float32",
}


def make_const(dtype, shape):
    return tvm.relay.const(np.random.rand(*shape).astype(dtype))


def make_consts(dtype, shapes):
    return [make_const(dtype, shape) for shape in shapes]


def mnist_consts(dtype):
    return make_consts(
        dtype,
        [
            (8, 1, 5, 5),  # 0
            (8, 1, 1),  # 1
            (16, 8, 5, 5),  # 2
            (16, 1, 1),  # 3
            (10, 256),  # 4
            (1, 10),  # 5
        ],
    )


def mnist():
    metatable = {"relay.Constant": mnist_consts("float32")}
    mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%x: Tensor[(1, 1, 28, 28), float32]) -> Tensor[(1, 10), float32] {
          %0 = nn.pad(%x, 0f, pad_width=[[0, 0], [0, 0], [2, 2], [2, 2]]);
          %1 = nn.conv2d(%0, meta[relay.Constant][0], padding=[0, 0, 0, 0], channels=8, kernel_size=[5, 5]);
          %2 = add(%1, meta[relay.Constant][1]);
          %3 = nn.relu(%2);
          %4 = nn.max_pool2d(%3, pool_size=[2, 2], strides=[2, 2], padding=[0, 0, 0, 0]);
          %5 = nn.pad(%4, 0f, pad_width=[[0, 0], [0, 0], [2, 2], [2, 2]]);
          %6 = nn.conv2d(%5, meta[relay.Constant][2], padding=[0, 0, 0, 0], channels=16, kernel_size=[5, 5]);
          %7 = add(%6, meta[relay.Constant][3]);
          %8 = nn.relu(%7);
          %9 = nn.max_pool2d(%8, pool_size=[3, 3], strides=[3, 3], padding=[0, 0, 0, 0]);
          %10 = reshape(%9, newshape=[1, 256]);
          %11 = nn.dense(%10, meta[relay.Constant][4], units=None, out_dtype="float32");
          add(%11, meta[relay.Constant][5])
        }
        """,
        "from_string",
        None,
        metatable,
    )
    return {
        "name": "mnist",
        "input_shapes": {"x": [1, 1, 28, 28]},
        "input_dtypes": {"x": "float32"},
        "mod": mod,
        "params": None,
        "main_dtype": "float32",
    }


def gpt2_consts(dtype):
    return make_consts(
        dtype,
        [
            (50257, 768),  # 0
            (1, 32, 768),  # 1
            (768,),  # 2
            (768,),  # 3
            (2304, 768),  # 4
            (2304,),  # 5
            (1, 1, 32, 32),  # 6
            (1, 1, 32, 32),  # 7
            (768, 768),  # 8
            (768,),  # 9
            (768,),  # 10
            (768,),  # 11
            (3072, 768),  # 12
            (3072,),  # 13
            (768, 3072),  # 14
            (768,),  # 15
            (768,),  # 16
            (768,),  # 17
            (2304, 768),  # 18
            (2304,),  # 19
            (1, 1, 32, 32),  # 20
            (1, 1, 32, 32),  # 21
            (768, 768),  # 22
            (768,),  # 23
            (768,),  # 24
            (768,),  # 25
            (3072, 768),  # 26
            (3072,),  # 27
            (768, 3072),  # 28
            (768,),  # 29
            (768,),  # 30
            (768,),  # 31
            (2304, 768),  # 32
            (2304,),  # 33
            (1, 1, 32, 32),  # 34
            (1, 1, 32, 32),  # 35
            (768, 768),  # 36
            (768,),  # 37
            (768,),  # 38
            (768,),  # 39
            (3072, 768),  # 40
            (3072,),  # 41
            (768, 3072),  # 42
            (768,),  # 43
            (768,),  # 44
            (768,),  # 45
            (2304, 768),  # 46
            (2304,),  # 47
            (1, 1, 32, 32),  # 48
            (1, 1, 32, 32),  # 49
            (768, 768),  # 50
            (768,),  # 51
            (768,),  # 52
            (768,),  # 53
            (3072, 768),  # 54
            (3072,),  # 55
            (768, 3072),  # 56
            (768,),  # 57
            (768,),  # 58
            (768,),  # 59
            (2304, 768),  # 60
            (2304,),  # 61
            (1, 1, 32, 32),  # 62
            (1, 1, 32, 32),  # 63
            (768, 768),  # 64
            (768,),  # 65
            (768,),  # 66
            (768,),  # 67
            (3072, 768),  # 68
            (3072,),  # 69
            (768, 3072),  # 70
            (768,),  # 71
            (768,),  # 72
            (768,),  # 73
            (2304, 768),  # 74
            (2304,),  # 75
            (1, 1, 32, 32),  # 76
            (1, 1, 32, 32),  # 77
            (768, 768),  # 78
            (768,),  # 79
            (768,),  # 80
            (768,),  # 81
            (3072, 768),  # 82
            (3072,),  # 83
            (768, 3072),  # 84
            (768,),  # 85
            (768,),  # 86
            (768,),  # 87
            (2304, 768),  # 88
            (2304,),  # 89
            (1, 1, 32, 32),  # 90
            (1, 1, 32, 32),  # 91
            (768, 768),  # 92
            (768,),  # 93
            (768,),  # 94
            (768,),  # 95
            (3072, 768),  # 96
            (3072,),  # 97
            (768, 3072),  # 98
            (768,),  # 99
            (768,),  # 100
            (768,),  # 101
            (2304, 768),  # 102
            (2304,),  # 103
            (1, 1, 32, 32),  # 104
            (1, 1, 32, 32),  # 105
            (768, 768),  # 106
            (768,),  # 107
            (768,),  # 108
            (768,),  # 109
            (3072, 768),  # 110
            (3072,),  # 111
            (768, 3072),  # 112
            (768,),  # 113
            (768,),  # 114
            (768,),  # 115
            (2304, 768),  # 116
            (2304,),  # 117
            (1, 1, 32, 32),  # 118
            (1, 1, 32, 32),  # 119
            (768, 768),  # 120
            (768,),  # 121
            (768,),  # 122
            (768,),  # 123
            (3072, 768),  # 124
            (3072,),  # 125
            (768, 3072),  # 126
            (768,),  # 127
            (768,),  # 128
            (768,),  # 129
            (2304, 768),  # 130
            (2304,),  # 131
            (1, 1, 32, 32),  # 132
            (1, 1, 32, 32),  # 133
            (768, 768),  # 134
            (768,),  # 135
            (768,),  # 136
            (768,),  # 137
            (3072, 768),  # 138
            (3072,),  # 139
            (768, 3072),  # 140
            (768,),  # 141
            (768,),  # 142
            (768,),  # 143
            (2304, 768),  # 144
            (2304,),  # 145
            (1, 1, 32, 32),  # 146
            (1, 1, 32, 32),  # 147
            (768, 768),  # 148
            (768,),  # 149
            (768,),  # 150
            (768,),  # 151
            (3072, 768),  # 152
            (3072,),  # 153
            (768, 3072),  # 154
            (768,),  # 155
            (768,),  # 156
            (768,),  # 157
            (2304, 768),  # 158
            (2304,),  # 159
            (1, 1, 32, 32),  # 160
            (1, 1, 32, 32),  # 161
            (768, 768),  # 162
            (768,),  # 163
            (768,),  # 164
            (768,),  # 165
            (3072, 768),  # 166
            (3072,),  # 167
            (768, 3072),  # 168
            (768,),  # 169
            (768,),  # 170
            (768,),  # 171
        ],
    )


def gpt2():
    metatable = {"relay.Constant": gpt2_consts("float32")}
    mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%x: Tensor[(1, 50, 32), int64]) -> (Tensor[(1, 50, 32, 768), float32],
                                                      Tensor[(2, 50, 12, 32, 64), float32],
                                                      Tensor[(2, 50, 12, 32, 64), float32],
                                                      Tensor[(2, 50, 12, 32, 64), float32],
                                                      Tensor[(2, 50, 12, 32, 64), float32],
                                                      Tensor[(2, 50, 12, 32, 64), float32],
                                                      Tensor[(2, 50, 12, 32, 64), float32],
                                                      Tensor[(2, 50, 12, 32, 64), float32],
                                                      Tensor[(2, 50, 12, 32, 64), float32],
                                                      Tensor[(2, 50, 12, 32, 64), float32],
                                                      Tensor[(2, 50, 12, 32, 64), float32],
                                                      Tensor[(2, 50, 12, 32, 64), float32],
                                                      Tensor[(2, 50, 12, 32, 64), float32]) {
          %0 = reshape(%x, newshape=[-1, 32]);
          %1 = less(%0, 0i64);
          %2 = add(%0, 50257i64);
          %3 = where(%1, %2, %0);
          %4 = take(meta[relay.Constant][0], %3, axis=0);
          %5 = add(%4, meta[relay.Constant][1]);
          %6 = mean(%5, axis=[-1], keepdims=True);
          %7 = subtract(%5, %6);
          %8 = power(%7, 2f);
          %9 = mean(%8, axis=[-1], keepdims=True);
          %10 = add(%9, 1e-05f);
          %11 = sqrt(%10);
          %12 = divide(%7, %11);
          %13 = multiply(%12, meta[relay.Constant][2]);
          %14 = add(%13, meta[relay.Constant][3]);
          %15 = reshape(%14, newshape=[-1, 768]);
          %16 = nn.dense(%15, meta[relay.Constant][4], units=2304);
          %17 = add(%16, meta[relay.Constant][5]);
          %18 = reshape(%17, newshape=[50, 32, 2304]);
          %19 = split(%18, indices_or_sections=[768, 1536], axis=2);
          %20 = %19.0;
          %21 = reshape(%20, newshape=[50, 32, 12, 64]);
          %22 = transpose(%21, axes=[0, 2, 1, 3]);
          %23 = %19.1;
          %24 = reshape(%23, newshape=[50, 32, 12, 64]);
          %25 = transpose(%24, axes=[0, 2, 3, 1]);
          %26 = reshape(%25, newshape=[-1, 64, 32]);
          %27 = reshape(%22, newshape=[-1, 32, 64]);
          %28 = transpose(%26, axes=[0, 2, 1]);
          %29 = nn.batch_matmul(%27, %28, out_dtype="float32", transpose_b=True);
          %30 = reshape(%29, newshape=[50, 12, 32, 32]);
          %31 = divide(%30, 8f);
          %32 = multiply(%31, meta[relay.Constant][6]);
          %33 = subtract(%32, meta[relay.Constant][7]);
          %34 = nn.softmax(%33, axis=3);
          %35 = %19.2;
          %36 = reshape(%35, newshape=[50, 32, 12, 64]);
          %37 = transpose(%36, axes=[0, 2, 1, 3]);
          %38 = reshape(%37, newshape=[-1, 32, 64]);
          %39 = reshape(%34, newshape=[-1, 32, 32]);
          %40 = transpose(%38, axes=[0, 2, 1]);
          %41 = nn.batch_matmul(%39, %40, out_dtype="float32", transpose_b=True);
          %42 = reshape(%41, newshape=[50, 12, 32, 64]);
          %43 = transpose(%42, axes=[0, 2, 1, 3]);
          %44 = reshape(%43, newshape=[50, 32, 768]);
          %45 = reshape(%44, newshape=[-1, 768]);
          %46 = nn.dense(%45, meta[relay.Constant][8], units=768);
          %47 = add(%46, meta[relay.Constant][9]);
          %48 = reshape(%47, newshape=[50, 32, 768]);
          %49 = add(%5, %48);
          %50 = mean(%49, axis=[-1], keepdims=True);
          %51 = subtract(%49, %50);
          %52 = power(%51, 2f);
          %53 = mean(%52, axis=[-1], keepdims=True);
          %54 = add(%53, 1e-05f);
          %55 = sqrt(%54);
          %56 = divide(%51, %55);
          %57 = multiply(%56, meta[relay.Constant][10]);
          %58 = add(%57, meta[relay.Constant][11]);
          %59 = reshape(%58, newshape=[-1, 768]);
          %60 = nn.dense(%59, meta[relay.Constant][12], units=3072);
          %61 = add(%60, meta[relay.Constant][13]);
          %62 = reshape(%61, newshape=[50, 32, 3072]);
          %63 = power(%62, 3f);
          %64 = multiply(%63, 0.044715f);
          %65 = add(%62, %64);
          %66 = multiply(%65, 0.797885f);
          %67 = tanh(%66);
          %68 = multiply(%62, 0.5f);
          %69 = add(%67, 1f);
          %70 = multiply(%68, %69);
          %71 = reshape(%70, newshape=[-1, 3072]);
          %72 = nn.dense(%71, meta[relay.Constant][14], units=768);
          %73 = add(%72, meta[relay.Constant][15]);
          %74 = reshape(%73, newshape=[50, 32, 768]);
          %75 = add(%49, %74);
          %76 = mean(%75, axis=[-1], keepdims=True);
          %77 = subtract(%75, %76);
          %78 = power(%77, 2f);
          %79 = mean(%78, axis=[-1], keepdims=True);
          %80 = add(%79, 1e-05f);
          %81 = sqrt(%80);
          %82 = divide(%77, %81);
          %83 = multiply(%82, meta[relay.Constant][16]);
          %84 = add(%83, meta[relay.Constant][17]);
          %85 = reshape(%84, newshape=[-1, 768]);
          %86 = nn.dense(%85, meta[relay.Constant][18], units=2304);
          %87 = add(%86, meta[relay.Constant][19]);
          %88 = reshape(%87, newshape=[50, 32, 2304]);
          %89 = split(%88, indices_or_sections=[768, 1536], axis=2);
          %90 = %89.0;
          %91 = reshape(%90, newshape=[50, 32, 12, 64]);
          %92 = transpose(%91, axes=[0, 2, 1, 3]);
          %93 = %89.1;
          %94 = reshape(%93, newshape=[50, 32, 12, 64]);
          %95 = transpose(%94, axes=[0, 2, 3, 1]);
          %96 = reshape(%95, newshape=[-1, 64, 32]);
          %97 = reshape(%92, newshape=[-1, 32, 64]);
          %98 = transpose(%96, axes=[0, 2, 1]);
          %99 = nn.batch_matmul(%97, %98, out_dtype="float32", transpose_b=True);
          %100 = reshape(%99, newshape=[50, 12, 32, 32]);
          %101 = divide(%100, 8f);
          %102 = multiply(%101, meta[relay.Constant][20]);
          %103 = subtract(%102, meta[relay.Constant][21]);
          %104 = nn.softmax(%103, axis=3);
          %105 = %89.2;
          %106 = reshape(%105, newshape=[50, 32, 12, 64]);
          %107 = transpose(%106, axes=[0, 2, 1, 3]);
          %108 = reshape(%107, newshape=[-1, 32, 64]);
          %109 = reshape(%104, newshape=[-1, 32, 32]);
          %110 = transpose(%108, axes=[0, 2, 1]);
          %111 = nn.batch_matmul(%109, %110, out_dtype="float32", transpose_b=True);
          %112 = reshape(%111, newshape=[50, 12, 32, 64]);
          %113 = transpose(%112, axes=[0, 2, 1, 3]);
          %114 = reshape(%113, newshape=[50, 32, 768]);
          %115 = reshape(%114, newshape=[-1, 768]);
          %116 = nn.dense(%115, meta[relay.Constant][22], units=768);
          %117 = add(%116, meta[relay.Constant][23]);
          %118 = reshape(%117, newshape=[50, 32, 768]);
          %119 = add(%75, %118);
          %120 = mean(%119, axis=[-1], keepdims=True);
          %121 = subtract(%119, %120);
          %122 = power(%121, 2f);
          %123 = mean(%122, axis=[-1], keepdims=True);
          %124 = add(%123, 1e-05f);
          %125 = sqrt(%124);
          %126 = divide(%121, %125);
          %127 = multiply(%126, meta[relay.Constant][24]);
          %128 = add(%127, meta[relay.Constant][25]);
          %129 = reshape(%128, newshape=[-1, 768]);
          %130 = nn.dense(%129, meta[relay.Constant][26], units=3072);
          %131 = add(%130, meta[relay.Constant][27]);
          %132 = reshape(%131, newshape=[50, 32, 3072]);
          %133 = power(%132, 3f);
          %134 = multiply(%133, 0.044715f);
          %135 = add(%132, %134);
          %136 = multiply(%135, 0.797885f);
          %137 = tanh(%136);
          %138 = multiply(%132, 0.5f);
          %139 = add(%137, 1f);
          %140 = multiply(%138, %139);
          %141 = reshape(%140, newshape=[-1, 3072]);
          %142 = nn.dense(%141, meta[relay.Constant][28], units=768);
          %143 = add(%142, meta[relay.Constant][29]);
          %144 = reshape(%143, newshape=[50, 32, 768]);
          %145 = add(%119, %144);
          %146 = mean(%145, axis=[-1], keepdims=True);
          %147 = subtract(%145, %146);
          %148 = power(%147, 2f);
          %149 = mean(%148, axis=[-1], keepdims=True);
          %150 = add(%149, 1e-05f);
          %151 = sqrt(%150);
          %152 = divide(%147, %151);
          %153 = multiply(%152, meta[relay.Constant][30]);
          %154 = add(%153, meta[relay.Constant][31]);
          %155 = reshape(%154, newshape=[-1, 768]);
          %156 = nn.dense(%155, meta[relay.Constant][32], units=2304);
          %157 = add(%156, meta[relay.Constant][33]);
          %158 = reshape(%157, newshape=[50, 32, 2304]);
          %159 = split(%158, indices_or_sections=[768, 1536], axis=2);
          %160 = %159.0;
          %161 = reshape(%160, newshape=[50, 32, 12, 64]);
          %162 = transpose(%161, axes=[0, 2, 1, 3]);
          %163 = %159.1;
          %164 = reshape(%163, newshape=[50, 32, 12, 64]);
          %165 = transpose(%164, axes=[0, 2, 3, 1]);
          %166 = reshape(%165, newshape=[-1, 64, 32]);
          %167 = reshape(%162, newshape=[-1, 32, 64]);
          %168 = transpose(%166, axes=[0, 2, 1]);
          %169 = nn.batch_matmul(%167, %168, out_dtype="float32", transpose_b=True);
          %170 = reshape(%169, newshape=[50, 12, 32, 32]);
          %171 = divide(%170, 8f);
          %172 = multiply(%171, meta[relay.Constant][34]);
          %173 = subtract(%172, meta[relay.Constant][35]);
          %174 = nn.softmax(%173, axis=3);
          %175 = %159.2;
          %176 = reshape(%175, newshape=[50, 32, 12, 64]);
          %177 = transpose(%176, axes=[0, 2, 1, 3]);
          %178 = reshape(%177, newshape=[-1, 32, 64]);
          %179 = reshape(%174, newshape=[-1, 32, 32]);
          %180 = transpose(%178, axes=[0, 2, 1]);
          %181 = nn.batch_matmul(%179, %180, out_dtype="float32", transpose_b=True);
          %182 = reshape(%181, newshape=[50, 12, 32, 64]);
          %183 = transpose(%182, axes=[0, 2, 1, 3]);
          %184 = reshape(%183, newshape=[50, 32, 768]);
          %185 = reshape(%184, newshape=[-1, 768]);
          %186 = nn.dense(%185, meta[relay.Constant][36], units=768);
          %187 = add(%186, meta[relay.Constant][37]);
          %188 = reshape(%187, newshape=[50, 32, 768]);
          %189 = add(%145, %188);
          %190 = mean(%189, axis=[-1], keepdims=True);
          %191 = subtract(%189, %190);
          %192 = power(%191, 2f);
          %193 = mean(%192, axis=[-1], keepdims=True);
          %194 = add(%193, 1e-05f);
          %195 = sqrt(%194);
          %196 = divide(%191, %195);
          %197 = multiply(%196, meta[relay.Constant][38]);
          %198 = add(%197, meta[relay.Constant][39]);
          %199 = reshape(%198, newshape=[-1, 768]);
          %200 = nn.dense(%199, meta[relay.Constant][40], units=3072);
          %201 = add(%200, meta[relay.Constant][41]);
          %202 = reshape(%201, newshape=[50, 32, 3072]);
          %203 = power(%202, 3f);
          %204 = multiply(%203, 0.044715f);
          %205 = add(%202, %204);
          %206 = multiply(%205, 0.797885f);
          %207 = tanh(%206);
          %208 = multiply(%202, 0.5f);
          %209 = add(%207, 1f);
          %210 = multiply(%208, %209);
          %211 = reshape(%210, newshape=[-1, 3072]);
          %212 = nn.dense(%211, meta[relay.Constant][42], units=768);
          %213 = add(%212, meta[relay.Constant][43]);
          %214 = reshape(%213, newshape=[50, 32, 768]);
          %215 = add(%189, %214);
          %216 = mean(%215, axis=[-1], keepdims=True);
          %217 = subtract(%215, %216);
          %218 = power(%217, 2f);
          %219 = mean(%218, axis=[-1], keepdims=True);
          %220 = add(%219, 1e-05f);
          %221 = sqrt(%220);
          %222 = divide(%217, %221);
          %223 = multiply(%222, meta[relay.Constant][44]);
          %224 = add(%223, meta[relay.Constant][45]);
          %225 = reshape(%224, newshape=[-1, 768]);
          %226 = nn.dense(%225, meta[relay.Constant][46], units=2304);
          %227 = add(%226, meta[relay.Constant][47]);
          %228 = reshape(%227, newshape=[50, 32, 2304]);
          %229 = split(%228, indices_or_sections=[768, 1536], axis=2);
          %230 = %229.0;
          %231 = reshape(%230, newshape=[50, 32, 12, 64]);
          %232 = transpose(%231, axes=[0, 2, 1, 3]);
          %233 = %229.1;
          %234 = reshape(%233, newshape=[50, 32, 12, 64]);
          %235 = transpose(%234, axes=[0, 2, 3, 1]);
          %236 = reshape(%235, newshape=[-1, 64, 32]);
          %237 = reshape(%232, newshape=[-1, 32, 64]);
          %238 = transpose(%236, axes=[0, 2, 1]);
          %239 = nn.batch_matmul(%237, %238, out_dtype="float32", transpose_b=True);
          %240 = reshape(%239, newshape=[50, 12, 32, 32]);
          %241 = divide(%240, 8f);
          %242 = multiply(%241, meta[relay.Constant][48]);
          %243 = subtract(%242, meta[relay.Constant][49]);
          %244 = nn.softmax(%243, axis=3);
          %245 = %229.2;
          %246 = reshape(%245, newshape=[50, 32, 12, 64]);
          %247 = transpose(%246, axes=[0, 2, 1, 3]);
          %248 = reshape(%247, newshape=[-1, 32, 64]);
          %249 = reshape(%244, newshape=[-1, 32, 32]);
          %250 = transpose(%248, axes=[0, 2, 1]);
          %251 = nn.batch_matmul(%249, %250, out_dtype="float32", transpose_b=True);
          %252 = reshape(%251, newshape=[50, 12, 32, 64]);
          %253 = transpose(%252, axes=[0, 2, 1, 3]);
          %254 = reshape(%253, newshape=[50, 32, 768]);
          %255 = reshape(%254, newshape=[-1, 768]);
          %256 = nn.dense(%255, meta[relay.Constant][50], units=768);
          %257 = add(%256, meta[relay.Constant][51]);
          %258 = reshape(%257, newshape=[50, 32, 768]);
          %259 = add(%215, %258);
          %260 = mean(%259, axis=[-1], keepdims=True);
          %261 = subtract(%259, %260);
          %262 = power(%261, 2f);
          %263 = mean(%262, axis=[-1], keepdims=True);
          %264 = add(%263, 1e-05f);
          %265 = sqrt(%264);
          %266 = divide(%261, %265);
          %267 = multiply(%266, meta[relay.Constant][52]);
          %268 = add(%267, meta[relay.Constant][53]);
          %269 = reshape(%268, newshape=[-1, 768]);
          %270 = nn.dense(%269, meta[relay.Constant][54], units=3072);
          %271 = add(%270, meta[relay.Constant][55]);
          %272 = reshape(%271, newshape=[50, 32, 3072]);
          %273 = power(%272, 3f);
          %274 = multiply(%273, 0.044715f);
          %275 = add(%272, %274);
          %276 = multiply(%275, 0.797885f);
          %277 = tanh(%276);
          %278 = multiply(%272, 0.5f);
          %279 = add(%277, 1f);
          %280 = multiply(%278, %279);
          %281 = reshape(%280, newshape=[-1, 3072]);
          %282 = nn.dense(%281, meta[relay.Constant][56], units=768);
          %283 = add(%282, meta[relay.Constant][57]);
          %284 = reshape(%283, newshape=[50, 32, 768]);
          %285 = add(%259, %284);
          %286 = mean(%285, axis=[-1], keepdims=True);
          %287 = subtract(%285, %286);
          %288 = power(%287, 2f);
          %289 = mean(%288, axis=[-1], keepdims=True);
          %290 = add(%289, 1e-05f);
          %291 = sqrt(%290);
          %292 = divide(%287, %291);
          %293 = multiply(%292, meta[relay.Constant][58]);
          %294 = add(%293, meta[relay.Constant][59]);
          %295 = reshape(%294, newshape=[-1, 768]);
          %296 = nn.dense(%295, meta[relay.Constant][60], units=2304);
          %297 = add(%296, meta[relay.Constant][61]);
          %298 = reshape(%297, newshape=[50, 32, 2304]);
          %299 = split(%298, indices_or_sections=[768, 1536], axis=2);
          %300 = %299.0;
          %301 = reshape(%300, newshape=[50, 32, 12, 64]);
          %302 = transpose(%301, axes=[0, 2, 1, 3]);
          %303 = %299.1;
          %304 = reshape(%303, newshape=[50, 32, 12, 64]);
          %305 = transpose(%304, axes=[0, 2, 3, 1]);
          %306 = reshape(%305, newshape=[-1, 64, 32]);
          %307 = reshape(%302, newshape=[-1, 32, 64]);
          %308 = transpose(%306, axes=[0, 2, 1]);
          %309 = nn.batch_matmul(%307, %308, out_dtype="float32", transpose_b=True);
          %310 = reshape(%309, newshape=[50, 12, 32, 32]);
          %311 = divide(%310, 8f);
          %312 = multiply(%311, meta[relay.Constant][62]);
          %313 = subtract(%312, meta[relay.Constant][63]);
          %314 = nn.softmax(%313, axis=3);
          %315 = %299.2;
          %316 = reshape(%315, newshape=[50, 32, 12, 64]);
          %317 = transpose(%316, axes=[0, 2, 1, 3]);
          %318 = reshape(%317, newshape=[-1, 32, 64]);
          %319 = reshape(%314, newshape=[-1, 32, 32]);
          %320 = transpose(%318, axes=[0, 2, 1]);
          %321 = nn.batch_matmul(%319, %320, out_dtype="float32", transpose_b=True);
          %322 = reshape(%321, newshape=[50, 12, 32, 64]);
          %323 = transpose(%322, axes=[0, 2, 1, 3]);
          %324 = reshape(%323, newshape=[50, 32, 768]);
          %325 = reshape(%324, newshape=[-1, 768]);
          %326 = nn.dense(%325, meta[relay.Constant][64], units=768);
          %327 = add(%326, meta[relay.Constant][65]);
          %328 = reshape(%327, newshape=[50, 32, 768]);
          %329 = add(%285, %328);
          %330 = mean(%329, axis=[-1], keepdims=True);
          %331 = subtract(%329, %330);
          %332 = power(%331, 2f);
          %333 = mean(%332, axis=[-1], keepdims=True);
          %334 = add(%333, 1e-05f);
          %335 = sqrt(%334);
          %336 = divide(%331, %335);
          %337 = multiply(%336, meta[relay.Constant][66]);
          %338 = add(%337, meta[relay.Constant][67]);
          %339 = reshape(%338, newshape=[-1, 768]);
          %340 = nn.dense(%339, meta[relay.Constant][68], units=3072);
          %341 = add(%340, meta[relay.Constant][69]);
          %342 = reshape(%341, newshape=[50, 32, 3072]);
          %343 = power(%342, 3f);
          %344 = multiply(%343, 0.044715f);
          %345 = add(%342, %344);
          %346 = multiply(%345, 0.797885f);
          %347 = tanh(%346);
          %348 = multiply(%342, 0.5f);
          %349 = add(%347, 1f);
          %350 = multiply(%348, %349);
          %351 = reshape(%350, newshape=[-1, 3072]);
          %352 = nn.dense(%351, meta[relay.Constant][70], units=768);
          %353 = add(%352, meta[relay.Constant][71]);
          %354 = reshape(%353, newshape=[50, 32, 768]);
          %355 = add(%329, %354);
          %356 = mean(%355, axis=[-1], keepdims=True);
          %357 = subtract(%355, %356);
          %358 = power(%357, 2f);
          %359 = mean(%358, axis=[-1], keepdims=True);
          %360 = add(%359, 1e-05f);
          %361 = sqrt(%360);
          %362 = divide(%357, %361);
          %363 = multiply(%362, meta[relay.Constant][72]);
          %364 = add(%363, meta[relay.Constant][73]);
          %365 = reshape(%364, newshape=[-1, 768]);
          %366 = nn.dense(%365, meta[relay.Constant][74], units=2304);
          %367 = add(%366, meta[relay.Constant][75]);
          %368 = reshape(%367, newshape=[50, 32, 2304]);
          %369 = split(%368, indices_or_sections=[768, 1536], axis=2);
          %370 = %369.0;
          %371 = reshape(%370, newshape=[50, 32, 12, 64]);
          %372 = transpose(%371, axes=[0, 2, 1, 3]);
          %373 = %369.1;
          %374 = reshape(%373, newshape=[50, 32, 12, 64]);
          %375 = transpose(%374, axes=[0, 2, 3, 1]);
          %376 = reshape(%375, newshape=[-1, 64, 32]);
          %377 = reshape(%372, newshape=[-1, 32, 64]);
          %378 = transpose(%376, axes=[0, 2, 1]);
          %379 = nn.batch_matmul(%377, %378, out_dtype="float32", transpose_b=True);
          %380 = reshape(%379, newshape=[50, 12, 32, 32]);
          %381 = divide(%380, 8f);
          %382 = multiply(%381, meta[relay.Constant][76]);
          %383 = subtract(%382, meta[relay.Constant][77]);
          %384 = nn.softmax(%383, axis=3);
          %385 = %369.2;
          %386 = reshape(%385, newshape=[50, 32, 12, 64]);
          %387 = transpose(%386, axes=[0, 2, 1, 3]);
          %388 = reshape(%387, newshape=[-1, 32, 64]);
          %389 = reshape(%384, newshape=[-1, 32, 32]);
          %390 = transpose(%388, axes=[0, 2, 1]);
          %391 = nn.batch_matmul(%389, %390, out_dtype="float32", transpose_b=True);
          %392 = reshape(%391, newshape=[50, 12, 32, 64]);
          %393 = transpose(%392, axes=[0, 2, 1, 3]);
          %394 = reshape(%393, newshape=[50, 32, 768]);
          %395 = reshape(%394, newshape=[-1, 768]);
          %396 = nn.dense(%395, meta[relay.Constant][78], units=768);
          %397 = add(%396, meta[relay.Constant][79]);
          %398 = reshape(%397, newshape=[50, 32, 768]);
          %399 = add(%355, %398);
          %400 = mean(%399, axis=[-1], keepdims=True);
          %401 = subtract(%399, %400);
          %402 = power(%401, 2f);
          %403 = mean(%402, axis=[-1], keepdims=True);
          %404 = add(%403, 1e-05f);
          %405 = sqrt(%404);
          %406 = divide(%401, %405);
          %407 = multiply(%406, meta[relay.Constant][80]);
          %408 = add(%407, meta[relay.Constant][81]);
          %409 = reshape(%408, newshape=[-1, 768]);
          %410 = nn.dense(%409, meta[relay.Constant][82], units=3072);
          %411 = add(%410, meta[relay.Constant][83]);
          %412 = reshape(%411, newshape=[50, 32, 3072]);
          %413 = power(%412, 3f);
          %414 = multiply(%413, 0.044715f);
          %415 = add(%412, %414);
          %416 = multiply(%415, 0.797885f);
          %417 = tanh(%416);
          %418 = multiply(%412, 0.5f);
          %419 = add(%417, 1f);
          %420 = multiply(%418, %419);
          %421 = reshape(%420, newshape=[-1, 3072]);
          %422 = nn.dense(%421, meta[relay.Constant][84], units=768);
          %423 = add(%422, meta[relay.Constant][85]);
          %424 = reshape(%423, newshape=[50, 32, 768]);
          %425 = add(%399, %424);
          %426 = mean(%425, axis=[-1], keepdims=True);
          %427 = subtract(%425, %426);
          %428 = power(%427, 2f);
          %429 = mean(%428, axis=[-1], keepdims=True);
          %430 = add(%429, 1e-05f);
          %431 = sqrt(%430);
          %432 = divide(%427, %431);
          %433 = multiply(%432, meta[relay.Constant][86]);
          %434 = add(%433, meta[relay.Constant][87]);
          %435 = reshape(%434, newshape=[-1, 768]);
          %436 = nn.dense(%435, meta[relay.Constant][88], units=2304);
          %437 = add(%436, meta[relay.Constant][89]);
          %438 = reshape(%437, newshape=[50, 32, 2304]);
          %439 = split(%438, indices_or_sections=[768, 1536], axis=2);
          %440 = %439.0;
          %441 = reshape(%440, newshape=[50, 32, 12, 64]);
          %442 = transpose(%441, axes=[0, 2, 1, 3]);
          %443 = %439.1;
          %444 = reshape(%443, newshape=[50, 32, 12, 64]);
          %445 = transpose(%444, axes=[0, 2, 3, 1]);
          %446 = reshape(%445, newshape=[-1, 64, 32]);
          %447 = reshape(%442, newshape=[-1, 32, 64]);
          %448 = transpose(%446, axes=[0, 2, 1]);
          %449 = nn.batch_matmul(%447, %448, out_dtype="float32", transpose_b=True);
          %450 = reshape(%449, newshape=[50, 12, 32, 32]);
          %451 = divide(%450, 8f);
          %452 = multiply(%451, meta[relay.Constant][90]);
          %453 = subtract(%452, meta[relay.Constant][91]);
          %454 = nn.softmax(%453, axis=3);
          %455 = %439.2;
          %456 = reshape(%455, newshape=[50, 32, 12, 64]);
          %457 = transpose(%456, axes=[0, 2, 1, 3]);
          %458 = reshape(%457, newshape=[-1, 32, 64]);
          %459 = reshape(%454, newshape=[-1, 32, 32]);
          %460 = transpose(%458, axes=[0, 2, 1]);
          %461 = nn.batch_matmul(%459, %460, out_dtype="float32", transpose_b=True);
          %462 = reshape(%461, newshape=[50, 12, 32, 64]);
          %463 = transpose(%462, axes=[0, 2, 1, 3]);
          %464 = reshape(%463, newshape=[50, 32, 768]);
          %465 = reshape(%464, newshape=[-1, 768]);
          %466 = nn.dense(%465, meta[relay.Constant][92], units=768);
          %467 = add(%466, meta[relay.Constant][93]);
          %468 = reshape(%467, newshape=[50, 32, 768]);
          %469 = add(%425, %468);
          %470 = mean(%469, axis=[-1], keepdims=True);
          %471 = subtract(%469, %470);
          %472 = power(%471, 2f);
          %473 = mean(%472, axis=[-1], keepdims=True);
          %474 = add(%473, 1e-05f);
          %475 = sqrt(%474);
          %476 = divide(%471, %475);
          %477 = multiply(%476, meta[relay.Constant][94]);
          %478 = add(%477, meta[relay.Constant][95]);
          %479 = reshape(%478, newshape=[-1, 768]);
          %480 = nn.dense(%479, meta[relay.Constant][96], units=3072);
          %481 = add(%480, meta[relay.Constant][97]);
          %482 = reshape(%481, newshape=[50, 32, 3072]);
          %483 = power(%482, 3f);
          %484 = multiply(%483, 0.044715f);
          %485 = add(%482, %484);
          %486 = multiply(%485, 0.797885f);
          %487 = tanh(%486);
          %488 = multiply(%482, 0.5f);
          %489 = add(%487, 1f);
          %490 = multiply(%488, %489);
          %491 = reshape(%490, newshape=[-1, 3072]);
          %492 = nn.dense(%491, meta[relay.Constant][98], units=768);
          %493 = add(%492, meta[relay.Constant][99]);
          %494 = reshape(%493, newshape=[50, 32, 768]);
          %495 = add(%469, %494);
          %496 = mean(%495, axis=[-1], keepdims=True);
          %497 = subtract(%495, %496);
          %498 = power(%497, 2f);
          %499 = mean(%498, axis=[-1], keepdims=True);
          %500 = add(%499, 1e-05f);
          %501 = sqrt(%500);
          %502 = divide(%497, %501);
          %503 = multiply(%502, meta[relay.Constant][100]);
          %504 = add(%503, meta[relay.Constant][101]);
          %505 = reshape(%504, newshape=[-1, 768]);
          %506 = nn.dense(%505, meta[relay.Constant][102], units=2304);
          %507 = add(%506, meta[relay.Constant][103]);
          %508 = reshape(%507, newshape=[50, 32, 2304]);
          %509 = split(%508, indices_or_sections=[768, 1536], axis=2);
          %510 = %509.0;
          %511 = reshape(%510, newshape=[50, 32, 12, 64]);
          %512 = transpose(%511, axes=[0, 2, 1, 3]);
          %513 = %509.1;
          %514 = reshape(%513, newshape=[50, 32, 12, 64]);
          %515 = transpose(%514, axes=[0, 2, 3, 1]);
          %516 = reshape(%515, newshape=[-1, 64, 32]);
          %517 = reshape(%512, newshape=[-1, 32, 64]);
          %518 = transpose(%516, axes=[0, 2, 1]);
          %519 = nn.batch_matmul(%517, %518, out_dtype="float32", transpose_b=True);
          %520 = reshape(%519, newshape=[50, 12, 32, 32]);
          %521 = divide(%520, 8f);
          %522 = multiply(%521, meta[relay.Constant][104]);
          %523 = subtract(%522, meta[relay.Constant][105]);
          %524 = nn.softmax(%523, axis=3);
          %525 = %509.2;
          %526 = reshape(%525, newshape=[50, 32, 12, 64]);
          %527 = transpose(%526, axes=[0, 2, 1, 3]);
          %528 = reshape(%527, newshape=[-1, 32, 64]);
          %529 = reshape(%524, newshape=[-1, 32, 32]);
          %530 = transpose(%528, axes=[0, 2, 1]);
          %531 = nn.batch_matmul(%529, %530, out_dtype="float32", transpose_b=True);
          %532 = reshape(%531, newshape=[50, 12, 32, 64]);
          %533 = transpose(%532, axes=[0, 2, 1, 3]);
          %534 = reshape(%533, newshape=[50, 32, 768]);
          %535 = reshape(%534, newshape=[-1, 768]);
          %536 = nn.dense(%535, meta[relay.Constant][106], units=768);
          %537 = add(%536, meta[relay.Constant][107]);
          %538 = reshape(%537, newshape=[50, 32, 768]);
          %539 = add(%495, %538);
          %540 = mean(%539, axis=[-1], keepdims=True);
          %541 = subtract(%539, %540);
          %542 = power(%541, 2f);
          %543 = mean(%542, axis=[-1], keepdims=True);
          %544 = add(%543, 1e-05f);
          %545 = sqrt(%544);
          %546 = divide(%541, %545);
          %547 = multiply(%546, meta[relay.Constant][108]);
          %548 = add(%547, meta[relay.Constant][109]);
          %549 = reshape(%548, newshape=[-1, 768]);
          %550 = nn.dense(%549, meta[relay.Constant][110], units=3072);
          %551 = add(%550, meta[relay.Constant][111]);
          %552 = reshape(%551, newshape=[50, 32, 3072]);
          %553 = power(%552, 3f);
          %554 = multiply(%553, 0.044715f);
          %555 = add(%552, %554);
          %556 = multiply(%555, 0.797885f);
          %557 = tanh(%556);
          %558 = multiply(%552, 0.5f);
          %559 = add(%557, 1f);
          %560 = multiply(%558, %559);
          %561 = reshape(%560, newshape=[-1, 3072]);
          %562 = nn.dense(%561, meta[relay.Constant][112], units=768);
          %563 = add(%562, meta[relay.Constant][113]);
          %564 = reshape(%563, newshape=[50, 32, 768]);
          %565 = add(%539, %564);
          %566 = mean(%565, axis=[-1], keepdims=True);
          %567 = subtract(%565, %566);
          %568 = power(%567, 2f);
          %569 = mean(%568, axis=[-1], keepdims=True);
          %570 = add(%569, 1e-05f);
          %571 = sqrt(%570);
          %572 = divide(%567, %571);
          %573 = multiply(%572, meta[relay.Constant][114]);
          %574 = add(%573, meta[relay.Constant][115]);
          %575 = reshape(%574, newshape=[-1, 768]);
          %576 = nn.dense(%575, meta[relay.Constant][116], units=2304);
          %577 = add(%576, meta[relay.Constant][117]);
          %578 = reshape(%577, newshape=[50, 32, 2304]);
          %579 = split(%578, indices_or_sections=[768, 1536], axis=2);
          %580 = %579.0;
          %581 = reshape(%580, newshape=[50, 32, 12, 64]);
          %582 = transpose(%581, axes=[0, 2, 1, 3]);
          %583 = %579.1;
          %584 = reshape(%583, newshape=[50, 32, 12, 64]);
          %585 = transpose(%584, axes=[0, 2, 3, 1]);
          %586 = reshape(%585, newshape=[-1, 64, 32]);
          %587 = reshape(%582, newshape=[-1, 32, 64]);
          %588 = transpose(%586, axes=[0, 2, 1]);
          %589 = nn.batch_matmul(%587, %588, out_dtype="float32", transpose_b=True);
          %590 = reshape(%589, newshape=[50, 12, 32, 32]);
          %591 = divide(%590, 8f);
          %592 = multiply(%591, meta[relay.Constant][118]);
          %593 = subtract(%592, meta[relay.Constant][119]);
          %594 = nn.softmax(%593, axis=3);
          %595 = %579.2;
          %596 = reshape(%595, newshape=[50, 32, 12, 64]);
          %597 = transpose(%596, axes=[0, 2, 1, 3]);
          %598 = reshape(%597, newshape=[-1, 32, 64]);
          %599 = reshape(%594, newshape=[-1, 32, 32]);
          %600 = transpose(%598, axes=[0, 2, 1]);
          %601 = nn.batch_matmul(%599, %600, out_dtype="float32", transpose_b=True);
          %602 = reshape(%601, newshape=[50, 12, 32, 64]);
          %603 = transpose(%602, axes=[0, 2, 1, 3]);
          %604 = reshape(%603, newshape=[50, 32, 768]);
          %605 = reshape(%604, newshape=[-1, 768]);
          %606 = nn.dense(%605, meta[relay.Constant][120], units=768);
          %607 = add(%606, meta[relay.Constant][121]);
          %608 = reshape(%607, newshape=[50, 32, 768]);
          %609 = add(%565, %608);
          %610 = mean(%609, axis=[-1], keepdims=True);
          %611 = subtract(%609, %610);
          %612 = power(%611, 2f);
          %613 = mean(%612, axis=[-1], keepdims=True);
          %614 = add(%613, 1e-05f);
          %615 = sqrt(%614);
          %616 = divide(%611, %615);
          %617 = multiply(%616, meta[relay.Constant][122]);
          %618 = add(%617, meta[relay.Constant][123]);
          %619 = reshape(%618, newshape=[-1, 768]);
          %620 = nn.dense(%619, meta[relay.Constant][124], units=3072);
          %621 = add(%620, meta[relay.Constant][125]);
          %622 = reshape(%621, newshape=[50, 32, 3072]);
          %623 = power(%622, 3f);
          %624 = multiply(%623, 0.044715f);
          %625 = add(%622, %624);
          %626 = multiply(%625, 0.797885f);
          %627 = tanh(%626);
          %628 = multiply(%622, 0.5f);
          %629 = add(%627, 1f);
          %630 = multiply(%628, %629);
          %631 = reshape(%630, newshape=[-1, 3072]);
          %632 = nn.dense(%631, meta[relay.Constant][126], units=768);
          %633 = add(%632, meta[relay.Constant][127]);
          %634 = reshape(%633, newshape=[50, 32, 768]);
          %635 = add(%609, %634);
          %636 = mean(%635, axis=[-1], keepdims=True);
          %637 = subtract(%635, %636);
          %638 = power(%637, 2f);
          %639 = mean(%638, axis=[-1], keepdims=True);
          %640 = add(%639, 1e-05f);
          %641 = sqrt(%640);
          %642 = divide(%637, %641);
          %643 = multiply(%642, meta[relay.Constant][128]);
          %644 = add(%643, meta[relay.Constant][129]);
          %645 = reshape(%644, newshape=[-1, 768]);
          %646 = nn.dense(%645, meta[relay.Constant][130], units=2304);
          %647 = add(%646, meta[relay.Constant][131]);
          %648 = reshape(%647, newshape=[50, 32, 2304]);
          %649 = split(%648, indices_or_sections=[768, 1536], axis=2);
          %650 = %649.0;
          %651 = reshape(%650, newshape=[50, 32, 12, 64]);
          %652 = transpose(%651, axes=[0, 2, 1, 3]);
          %653 = %649.1;
          %654 = reshape(%653, newshape=[50, 32, 12, 64]);
          %655 = transpose(%654, axes=[0, 2, 3, 1]);
          %656 = reshape(%655, newshape=[-1, 64, 32]);
          %657 = reshape(%652, newshape=[-1, 32, 64]);
          %658 = transpose(%656, axes=[0, 2, 1]);
          %659 = nn.batch_matmul(%657, %658, out_dtype="float32", transpose_b=True);
          %660 = reshape(%659, newshape=[50, 12, 32, 32]);
          %661 = divide(%660, 8f);
          %662 = multiply(%661, meta[relay.Constant][132]);
          %663 = subtract(%662, meta[relay.Constant][133]);
          %664 = nn.softmax(%663, axis=3);
          %665 = %649.2;
          %666 = reshape(%665, newshape=[50, 32, 12, 64]);
          %667 = transpose(%666, axes=[0, 2, 1, 3]);
          %668 = reshape(%667, newshape=[-1, 32, 64]);
          %669 = reshape(%664, newshape=[-1, 32, 32]);
          %670 = transpose(%668, axes=[0, 2, 1]);
          %671 = nn.batch_matmul(%669, %670, out_dtype="float32", transpose_b=True);
          %672 = reshape(%671, newshape=[50, 12, 32, 64]);
          %673 = transpose(%672, axes=[0, 2, 1, 3]);
          %674 = reshape(%673, newshape=[50, 32, 768]);
          %675 = reshape(%674, newshape=[-1, 768]);
          %676 = nn.dense(%675, meta[relay.Constant][134], units=768);
          %677 = add(%676, meta[relay.Constant][135]);
          %678 = reshape(%677, newshape=[50, 32, 768]);
          %679 = add(%635, %678);
          %680 = mean(%679, axis=[-1], keepdims=True);
          %681 = subtract(%679, %680);
          %682 = power(%681, 2f);
          %683 = mean(%682, axis=[-1], keepdims=True);
          %684 = add(%683, 1e-05f);
          %685 = sqrt(%684);
          %686 = divide(%681, %685);
          %687 = multiply(%686, meta[relay.Constant][136]);
          %688 = add(%687, meta[relay.Constant][137]);
          %689 = reshape(%688, newshape=[-1, 768]);
          %690 = nn.dense(%689, meta[relay.Constant][138], units=3072);
          %691 = add(%690, meta[relay.Constant][139]);
          %692 = reshape(%691, newshape=[50, 32, 3072]);
          %693 = power(%692, 3f);
          %694 = multiply(%693, 0.044715f);
          %695 = add(%692, %694);
          %696 = multiply(%695, 0.797885f);
          %697 = tanh(%696);
          %698 = multiply(%692, 0.5f);
          %699 = add(%697, 1f);
          %700 = multiply(%698, %699);
          %701 = reshape(%700, newshape=[-1, 3072]);
          %702 = nn.dense(%701, meta[relay.Constant][140], units=768);
          %703 = add(%702, meta[relay.Constant][141]);
          %704 = reshape(%703, newshape=[50, 32, 768]);
          %705 = add(%679, %704);
          %706 = mean(%705, axis=[-1], keepdims=True);
          %707 = subtract(%705, %706);
          %708 = power(%707, 2f);
          %709 = mean(%708, axis=[-1], keepdims=True);
          %710 = add(%709, 1e-05f);
          %711 = sqrt(%710);
          %712 = divide(%707, %711);
          %713 = multiply(%712, meta[relay.Constant][142]);
          %714 = add(%713, meta[relay.Constant][143]);
          %715 = reshape(%714, newshape=[-1, 768]);
          %716 = nn.dense(%715, meta[relay.Constant][144], units=2304);
          %717 = add(%716, meta[relay.Constant][145]);
          %718 = reshape(%717, newshape=[50, 32, 2304]);
          %719 = split(%718, indices_or_sections=[768, 1536], axis=2);
          %720 = %719.0;
          %721 = reshape(%720, newshape=[50, 32, 12, 64]);
          %722 = transpose(%721, axes=[0, 2, 1, 3]);
          %723 = %719.1;
          %724 = reshape(%723, newshape=[50, 32, 12, 64]);
          %725 = transpose(%724, axes=[0, 2, 3, 1]);
          %726 = reshape(%725, newshape=[-1, 64, 32]);
          %727 = reshape(%722, newshape=[-1, 32, 64]);
          %728 = transpose(%726, axes=[0, 2, 1]);
          %729 = nn.batch_matmul(%727, %728, out_dtype="float32", transpose_b=True);
          %730 = reshape(%729, newshape=[50, 12, 32, 32]);
          %731 = divide(%730, 8f);
          %732 = multiply(%731, meta[relay.Constant][146]);
          %733 = subtract(%732, meta[relay.Constant][147]);
          %734 = nn.softmax(%733, axis=3);
          %735 = %719.2;
          %736 = reshape(%735, newshape=[50, 32, 12, 64]);
          %737 = transpose(%736, axes=[0, 2, 1, 3]);
          %738 = reshape(%737, newshape=[-1, 32, 64]);
          %739 = reshape(%734, newshape=[-1, 32, 32]);
          %740 = transpose(%738, axes=[0, 2, 1]);
          %741 = nn.batch_matmul(%739, %740, out_dtype="float32", transpose_b=True);
          %742 = reshape(%741, newshape=[50, 12, 32, 64]);
          %743 = transpose(%742, axes=[0, 2, 1, 3]);
          %744 = reshape(%743, newshape=[50, 32, 768]);
          %745 = reshape(%744, newshape=[-1, 768]);
          %746 = nn.dense(%745, meta[relay.Constant][148], units=768);
          %747 = add(%746, meta[relay.Constant][149]);
          %748 = reshape(%747, newshape=[50, 32, 768]);
          %749 = add(%705, %748);
          %750 = mean(%749, axis=[-1], keepdims=True);
          %751 = subtract(%749, %750);
          %752 = power(%751, 2f);
          %753 = mean(%752, axis=[-1], keepdims=True);
          %754 = add(%753, 1e-05f);
          %755 = sqrt(%754);
          %756 = divide(%751, %755);
          %757 = multiply(%756, meta[relay.Constant][150]);
          %758 = add(%757, meta[relay.Constant][151]);
          %759 = reshape(%758, newshape=[-1, 768]);
          %760 = nn.dense(%759, meta[relay.Constant][152], units=3072);
          %761 = add(%760, meta[relay.Constant][153]);
          %762 = reshape(%761, newshape=[50, 32, 3072]);
          %763 = power(%762, 3f);
          %764 = multiply(%763, 0.044715f);
          %765 = add(%762, %764);
          %766 = multiply(%765, 0.797885f);
          %767 = tanh(%766);
          %768 = multiply(%762, 0.5f);
          %769 = add(%767, 1f);
          %770 = multiply(%768, %769);
          %771 = reshape(%770, newshape=[-1, 3072]);
          %772 = nn.dense(%771, meta[relay.Constant][154], units=768);
          %773 = add(%772, meta[relay.Constant][155]);
          %774 = reshape(%773, newshape=[50, 32, 768]);
          %775 = add(%749, %774);
          %776 = mean(%775, axis=[-1], keepdims=True);
          %777 = subtract(%775, %776);
          %778 = power(%777, 2f);
          %779 = mean(%778, axis=[-1], keepdims=True);
          %780 = add(%779, 1e-05f);
          %781 = sqrt(%780);
          %782 = divide(%777, %781);
          %783 = multiply(%782, meta[relay.Constant][156]);
          %784 = add(%783, meta[relay.Constant][157]);
          %785 = reshape(%784, newshape=[-1, 768]);
          %786 = nn.dense(%785, meta[relay.Constant][158], units=2304);
          %787 = add(%786, meta[relay.Constant][159]);
          %788 = reshape(%787, newshape=[50, 32, 2304]);
          %789 = split(%788, indices_or_sections=[768, 1536], axis=2);
          %790 = %789.0;
          %791 = reshape(%790, newshape=[50, 32, 12, 64]);
          %792 = transpose(%791, axes=[0, 2, 1, 3]);
          %793 = %789.1;
          %794 = reshape(%793, newshape=[50, 32, 12, 64]);
          %795 = transpose(%794, axes=[0, 2, 3, 1]);
          %796 = reshape(%795, newshape=[-1, 64, 32]);
          %797 = reshape(%792, newshape=[-1, 32, 64]);
          %798 = transpose(%796, axes=[0, 2, 1]);
          %799 = nn.batch_matmul(%797, %798, out_dtype="float32", transpose_b=True);
          %800 = reshape(%799, newshape=[50, 12, 32, 32]);
          %801 = divide(%800, 8f);
          %802 = multiply(%801, meta[relay.Constant][160]);
          %803 = subtract(%802, meta[relay.Constant][161]);
          %804 = nn.softmax(%803, axis=3);
          %805 = %789.2;
          %806 = reshape(%805, newshape=[50, 32, 12, 64]);
          %807 = transpose(%806, axes=[0, 2, 1, 3]);
          %808 = reshape(%807, newshape=[-1, 32, 64]);
          %809 = reshape(%804, newshape=[-1, 32, 32]);
          %810 = transpose(%808, axes=[0, 2, 1]);
          %811 = nn.batch_matmul(%809, %810, out_dtype="float32", transpose_b=True);
          %812 = reshape(%811, newshape=[50, 12, 32, 64]);
          %813 = transpose(%812, axes=[0, 2, 1, 3]);
          %814 = reshape(%813, newshape=[50, 32, 768]);
          %815 = reshape(%814, newshape=[-1, 768]);
          %816 = nn.dense(%815, meta[relay.Constant][162], units=768);
          %817 = add(%816, meta[relay.Constant][163]);
          %818 = reshape(%817, newshape=[50, 32, 768]);
          %819 = add(%775, %818);
          %820 = mean(%819, axis=[-1], keepdims=True);
          %821 = subtract(%819, %820);
          %822 = power(%821, 2f);
          %823 = mean(%822, axis=[-1], keepdims=True);
          %824 = add(%823, 1e-05f);
          %825 = sqrt(%824);
          %826 = divide(%821, %825);
          %827 = multiply(%826, meta[relay.Constant][164]);
          %828 = add(%827, meta[relay.Constant][165]);
          %829 = reshape(%828, newshape=[-1, 768]);
          %830 = nn.dense(%829, meta[relay.Constant][166], units=3072);
          %831 = add(%830, meta[relay.Constant][167]);
          %832 = reshape(%831, newshape=[50, 32, 3072]);
          %833 = power(%832, 3f);
          %834 = multiply(%833, 0.044715f);
          %835 = add(%832, %834);
          %836 = multiply(%835, 0.797885f);
          %837 = tanh(%836);
          %838 = multiply(%832, 0.5f);
          %839 = add(%837, 1f);
          %840 = multiply(%838, %839);
          %841 = reshape(%840, newshape=[-1, 3072]);
          %842 = nn.dense(%841, meta[relay.Constant][168], units=768);
          %843 = add(%842, meta[relay.Constant][169]);
          %844 = reshape(%843, newshape=[50, 32, 768]);
          %845 = add(%819, %844);
          %846 = mean(%845, axis=[-1], keepdims=True);
          %847 = subtract(%845, %846);
          %848 = power(%847, 2f);
          %849 = mean(%848, axis=[-1], keepdims=True);
          %850 = add(%849, 1e-05f);
          %851 = sqrt(%850);
          %852 = divide(%847, %851);
          %853 = multiply(%852, meta[relay.Constant][170]);
          %854 = add(%853, meta[relay.Constant][171]);
          %855 = transpose(%24, axes=[0, 2, 1, 3]);
          %856 = expand_dims(%855, axis=0);
          %857 = expand_dims(%37, axis=0);
          %858 = (%856, %857);
          %859 = transpose(%94, axes=[0, 2, 1, 3]);
          %860 = expand_dims(%859, axis=0);
          %861 = expand_dims(%107, axis=0);
          %862 = (%860, %861);
          %863 = transpose(%164, axes=[0, 2, 1, 3]);
          %864 = expand_dims(%863, axis=0);
          %865 = expand_dims(%177, axis=0);
          %866 = (%864, %865);
          %867 = transpose(%234, axes=[0, 2, 1, 3]);
          %868 = expand_dims(%867, axis=0);
          %869 = expand_dims(%247, axis=0);
          %870 = (%868, %869);
          %871 = transpose(%304, axes=[0, 2, 1, 3]);
          %872 = expand_dims(%871, axis=0);
          %873 = expand_dims(%317, axis=0);
          %874 = (%872, %873);
          %875 = transpose(%374, axes=[0, 2, 1, 3]);
          %876 = expand_dims(%875, axis=0);
          %877 = expand_dims(%387, axis=0);
          %878 = (%876, %877);
          %879 = transpose(%444, axes=[0, 2, 1, 3]);
          %880 = expand_dims(%879, axis=0);
          %881 = expand_dims(%457, axis=0);
          %882 = (%880, %881);
          %883 = transpose(%514, axes=[0, 2, 1, 3]);
          %884 = expand_dims(%883, axis=0);
          %885 = expand_dims(%527, axis=0);
          %886 = (%884, %885);
          %887 = transpose(%584, axes=[0, 2, 1, 3]);
          %888 = expand_dims(%887, axis=0);
          %889 = expand_dims(%597, axis=0);
          %890 = (%888, %889);
          %891 = transpose(%654, axes=[0, 2, 1, 3]);
          %892 = expand_dims(%891, axis=0);
          %893 = expand_dims(%667, axis=0);
          %894 = (%892, %893);
          %895 = transpose(%724, axes=[0, 2, 1, 3]);
          %896 = expand_dims(%895, axis=0);
          %897 = expand_dims(%737, axis=0);
          %898 = (%896, %897);
          %899 = transpose(%794, axes=[0, 2, 1, 3]);
          %900 = expand_dims(%899, axis=0);
          %901 = expand_dims(%807, axis=0);
          %902 = (%900, %901);
          %903 = reshape(%854, newshape=[1, 50, 32, 768]);
          %904 = concatenate(%858);
          %905 = concatenate(%862);
          %906 = concatenate(%866);
          %907 = concatenate(%870);
          %908 = concatenate(%874);
          %909 = concatenate(%878);
          %910 = concatenate(%882);
          %911 = concatenate(%886);
          %912 = concatenate(%890);
          %913 = concatenate(%894);
          %914 = concatenate(%898);
          %915 = concatenate(%902);
          (%903, %904, %905, %906, %907, %908, %909, %910, %911, %912, %913, %914, %915)
        }
        """,
        "from_string",
        None,
        metatable,
    )
    return {
        "name": "gpt2",
        "input_shapes": {"x": [1, 50, 32]},
        "input_dtypes": {"x": "int64"},
        "mod": mod,
        "params": None,
        "main_dtype": "float32",
    }


def gpt2_16():
    metatable = {"relay.Constant": gpt2_consts("float16")}
    mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%x: Tensor[(1, 50, 32), int64]) -> (Tensor[(1, 50, 32, 768), float16],
                                                      Tensor[(2, 50, 12, 32, 64), float16],
                                                      Tensor[(2, 50, 12, 32, 64), float16],
                                                      Tensor[(2, 50, 12, 32, 64), float16],
                                                      Tensor[(2, 50, 12, 32, 64), float16],
                                                      Tensor[(2, 50, 12, 32, 64), float16],
                                                      Tensor[(2, 50, 12, 32, 64), float16],
                                                      Tensor[(2, 50, 12, 32, 64), float16],
                                                      Tensor[(2, 50, 12, 32, 64), float16],
                                                      Tensor[(2, 50, 12, 32, 64), float16],
                                                      Tensor[(2, 50, 12, 32, 64), float16],
                                                      Tensor[(2, 50, 12, 32, 64), float16],
                                                      Tensor[(2, 50, 12, 32, 64), float16]) {
          %0 = reshape(%x, newshape=[-1, 32]);
          %1 = less(%0, 0i64);
          %2 = add(%0, 50257i64);
          %3 = where(%1, %2, %0);
          %4 = take(meta[relay.Constant][0], %3, axis=0);
          %5 = add(%4, meta[relay.Constant][1]);
          %6 = mean(%5, axis=[-1], keepdims=True);
          %7 = subtract(%5, %6);
          %8 = power(%7, 2f16);
          %9 = mean(%8, axis=[-1], keepdims=True);
          %10 = add(%9, 1e-05f16);
          %11 = sqrt(%10);
          %12 = divide(%7, %11);
          %13 = multiply(%12, meta[relay.Constant][2]);
          %14 = add(%13, meta[relay.Constant][3]);
          %15 = reshape(%14, newshape=[-1, 768]);
          %16 = nn.dense(%15, meta[relay.Constant][4], units=2304);
          %17 = add(%16, meta[relay.Constant][5]);
          %18 = reshape(%17, newshape=[50, 32, 2304]);
          %19 = split(%18, indices_or_sections=[768, 1536], axis=2);
          %20 = %19.0;
          %21 = reshape(%20, newshape=[50, 32, 12, 64]);
          %22 = transpose(%21, axes=[0, 2, 1, 3]);
          %23 = %19.1;
          %24 = reshape(%23, newshape=[50, 32, 12, 64]);
          %25 = transpose(%24, axes=[0, 2, 3, 1]);
          %26 = reshape(%25, newshape=[-1, 64, 32]);
          %27 = reshape(%22, newshape=[-1, 32, 64]);
          %28 = transpose(%26, axes=[0, 2, 1]);
          %29 = nn.batch_matmul(%27, %28, out_dtype="float16", transpose_b=True);
          %30 = reshape(%29, newshape=[50, 12, 32, 32]);
          %31 = divide(%30, 8f16);
          %32 = multiply(%31, meta[relay.Constant][6]);
          %33 = subtract(%32, meta[relay.Constant][7]);
          %34 = nn.softmax(%33, axis=3);
          %35 = %19.2;
          %36 = reshape(%35, newshape=[50, 32, 12, 64]);
          %37 = transpose(%36, axes=[0, 2, 1, 3]);
          %38 = reshape(%37, newshape=[-1, 32, 64]);
          %39 = reshape(%34, newshape=[-1, 32, 32]);
          %40 = transpose(%38, axes=[0, 2, 1]);
          %41 = nn.batch_matmul(%39, %40, out_dtype="float16", transpose_b=True);
          %42 = reshape(%41, newshape=[50, 12, 32, 64]);
          %43 = transpose(%42, axes=[0, 2, 1, 3]);
          %44 = reshape(%43, newshape=[50, 32, 768]);
          %45 = reshape(%44, newshape=[-1, 768]);
          %46 = nn.dense(%45, meta[relay.Constant][8], units=768);
          %47 = add(%46, meta[relay.Constant][9]);
          %48 = reshape(%47, newshape=[50, 32, 768]);
          %49 = add(%5, %48);
          %50 = mean(%49, axis=[-1], keepdims=True);
          %51 = subtract(%49, %50);
          %52 = power(%51, 2f16);
          %53 = mean(%52, axis=[-1], keepdims=True);
          %54 = add(%53, 1e-05f16);
          %55 = sqrt(%54);
          %56 = divide(%51, %55);
          %57 = multiply(%56, meta[relay.Constant][10]);
          %58 = add(%57, meta[relay.Constant][11]);
          %59 = reshape(%58, newshape=[-1, 768]);
          %60 = nn.dense(%59, meta[relay.Constant][12], units=3072);
          %61 = add(%60, meta[relay.Constant][13]);
          %62 = reshape(%61, newshape=[50, 32, 3072]);
          %63 = power(%62, 3f16);
          %64 = multiply(%63, 0.044715f16);
          %65 = add(%62, %64);
          %66 = multiply(%65, 0.797885f16);
          %67 = tanh(%66);
          %68 = multiply(%62, 0.5f16);
          %69 = add(%67, 1f16);
          %70 = multiply(%68, %69);
          %71 = reshape(%70, newshape=[-1, 3072]);
          %72 = nn.dense(%71, meta[relay.Constant][14], units=768);
          %73 = add(%72, meta[relay.Constant][15]);
          %74 = reshape(%73, newshape=[50, 32, 768]);
          %75 = add(%49, %74);
          %76 = mean(%75, axis=[-1], keepdims=True);
          %77 = subtract(%75, %76);
          %78 = power(%77, 2f16);
          %79 = mean(%78, axis=[-1], keepdims=True);
          %80 = add(%79, 1e-05f16);
          %81 = sqrt(%80);
          %82 = divide(%77, %81);
          %83 = multiply(%82, meta[relay.Constant][16]);
          %84 = add(%83, meta[relay.Constant][17]);
          %85 = reshape(%84, newshape=[-1, 768]);
          %86 = nn.dense(%85, meta[relay.Constant][18], units=2304);
          %87 = add(%86, meta[relay.Constant][19]);
          %88 = reshape(%87, newshape=[50, 32, 2304]);
          %89 = split(%88, indices_or_sections=[768, 1536], axis=2);
          %90 = %89.0;
          %91 = reshape(%90, newshape=[50, 32, 12, 64]);
          %92 = transpose(%91, axes=[0, 2, 1, 3]);
          %93 = %89.1;
          %94 = reshape(%93, newshape=[50, 32, 12, 64]);
          %95 = transpose(%94, axes=[0, 2, 3, 1]);
          %96 = reshape(%95, newshape=[-1, 64, 32]);
          %97 = reshape(%92, newshape=[-1, 32, 64]);
          %98 = transpose(%96, axes=[0, 2, 1]);
          %99 = nn.batch_matmul(%97, %98, out_dtype="float16", transpose_b=True);
          %100 = reshape(%99, newshape=[50, 12, 32, 32]);
          %101 = divide(%100, 8f16);
          %102 = multiply(%101, meta[relay.Constant][20]);
          %103 = subtract(%102, meta[relay.Constant][21]);
          %104 = nn.softmax(%103, axis=3);
          %105 = %89.2;
          %106 = reshape(%105, newshape=[50, 32, 12, 64]);
          %107 = transpose(%106, axes=[0, 2, 1, 3]);
          %108 = reshape(%107, newshape=[-1, 32, 64]);
          %109 = reshape(%104, newshape=[-1, 32, 32]);
          %110 = transpose(%108, axes=[0, 2, 1]);
          %111 = nn.batch_matmul(%109, %110, out_dtype="float16", transpose_b=True);
          %112 = reshape(%111, newshape=[50, 12, 32, 64]);
          %113 = transpose(%112, axes=[0, 2, 1, 3]);
          %114 = reshape(%113, newshape=[50, 32, 768]);
          %115 = reshape(%114, newshape=[-1, 768]);
          %116 = nn.dense(%115, meta[relay.Constant][22], units=768);
          %117 = add(%116, meta[relay.Constant][23]);
          %118 = reshape(%117, newshape=[50, 32, 768]);
          %119 = add(%75, %118);
          %120 = mean(%119, axis=[-1], keepdims=True);
          %121 = subtract(%119, %120);
          %122 = power(%121, 2f16);
          %123 = mean(%122, axis=[-1], keepdims=True);
          %124 = add(%123, 1e-05f16);
          %125 = sqrt(%124);
          %126 = divide(%121, %125);
          %127 = multiply(%126, meta[relay.Constant][24]);
          %128 = add(%127, meta[relay.Constant][25]);
          %129 = reshape(%128, newshape=[-1, 768]);
          %130 = nn.dense(%129, meta[relay.Constant][26], units=3072);
          %131 = add(%130, meta[relay.Constant][27]);
          %132 = reshape(%131, newshape=[50, 32, 3072]);
          %133 = power(%132, 3f16);
          %134 = multiply(%133, 0.044715f16);
          %135 = add(%132, %134);
          %136 = multiply(%135, 0.797885f16);
          %137 = tanh(%136);
          %138 = multiply(%132, 0.5f16);
          %139 = add(%137, 1f16);
          %140 = multiply(%138, %139);
          %141 = reshape(%140, newshape=[-1, 3072]);
          %142 = nn.dense(%141, meta[relay.Constant][28], units=768);
          %143 = add(%142, meta[relay.Constant][29]);
          %144 = reshape(%143, newshape=[50, 32, 768]);
          %145 = add(%119, %144);
          %146 = mean(%145, axis=[-1], keepdims=True);
          %147 = subtract(%145, %146);
          %148 = power(%147, 2f16);
          %149 = mean(%148, axis=[-1], keepdims=True);
          %150 = add(%149, 1e-05f16);
          %151 = sqrt(%150);
          %152 = divide(%147, %151);
          %153 = multiply(%152, meta[relay.Constant][30]);
          %154 = add(%153, meta[relay.Constant][31]);
          %155 = reshape(%154, newshape=[-1, 768]);
          %156 = nn.dense(%155, meta[relay.Constant][32], units=2304);
          %157 = add(%156, meta[relay.Constant][33]);
          %158 = reshape(%157, newshape=[50, 32, 2304]);
          %159 = split(%158, indices_or_sections=[768, 1536], axis=2);
          %160 = %159.0;
          %161 = reshape(%160, newshape=[50, 32, 12, 64]);
          %162 = transpose(%161, axes=[0, 2, 1, 3]);
          %163 = %159.1;
          %164 = reshape(%163, newshape=[50, 32, 12, 64]);
          %165 = transpose(%164, axes=[0, 2, 3, 1]);
          %166 = reshape(%165, newshape=[-1, 64, 32]);
          %167 = reshape(%162, newshape=[-1, 32, 64]);
          %168 = transpose(%166, axes=[0, 2, 1]);
          %169 = nn.batch_matmul(%167, %168, out_dtype="float16", transpose_b=True);
          %170 = reshape(%169, newshape=[50, 12, 32, 32]);
          %171 = divide(%170, 8f16);
          %172 = multiply(%171, meta[relay.Constant][34]);
          %173 = subtract(%172, meta[relay.Constant][35]);
          %174 = nn.softmax(%173, axis=3);
          %175 = %159.2;
          %176 = reshape(%175, newshape=[50, 32, 12, 64]);
          %177 = transpose(%176, axes=[0, 2, 1, 3]);
          %178 = reshape(%177, newshape=[-1, 32, 64]);
          %179 = reshape(%174, newshape=[-1, 32, 32]);
          %180 = transpose(%178, axes=[0, 2, 1]);
          %181 = nn.batch_matmul(%179, %180, out_dtype="float16", transpose_b=True);
          %182 = reshape(%181, newshape=[50, 12, 32, 64]);
          %183 = transpose(%182, axes=[0, 2, 1, 3]);
          %184 = reshape(%183, newshape=[50, 32, 768]);
          %185 = reshape(%184, newshape=[-1, 768]);
          %186 = nn.dense(%185, meta[relay.Constant][36], units=768);
          %187 = add(%186, meta[relay.Constant][37]);
          %188 = reshape(%187, newshape=[50, 32, 768]);
          %189 = add(%145, %188);
          %190 = mean(%189, axis=[-1], keepdims=True);
          %191 = subtract(%189, %190);
          %192 = power(%191, 2f16);
          %193 = mean(%192, axis=[-1], keepdims=True);
          %194 = add(%193, 1e-05f16);
          %195 = sqrt(%194);
          %196 = divide(%191, %195);
          %197 = multiply(%196, meta[relay.Constant][38]);
          %198 = add(%197, meta[relay.Constant][39]);
          %199 = reshape(%198, newshape=[-1, 768]);
          %200 = nn.dense(%199, meta[relay.Constant][40], units=3072);
          %201 = add(%200, meta[relay.Constant][41]);
          %202 = reshape(%201, newshape=[50, 32, 3072]);
          %203 = power(%202, 3f16);
          %204 = multiply(%203, 0.044715f16);
          %205 = add(%202, %204);
          %206 = multiply(%205, 0.797885f16);
          %207 = tanh(%206);
          %208 = multiply(%202, 0.5f16);
          %209 = add(%207, 1f16);
          %210 = multiply(%208, %209);
          %211 = reshape(%210, newshape=[-1, 3072]);
          %212 = nn.dense(%211, meta[relay.Constant][42], units=768);
          %213 = add(%212, meta[relay.Constant][43]);
          %214 = reshape(%213, newshape=[50, 32, 768]);
          %215 = add(%189, %214);
          %216 = mean(%215, axis=[-1], keepdims=True);
          %217 = subtract(%215, %216);
          %218 = power(%217, 2f16);
          %219 = mean(%218, axis=[-1], keepdims=True);
          %220 = add(%219, 1e-05f16);
          %221 = sqrt(%220);
          %222 = divide(%217, %221);
          %223 = multiply(%222, meta[relay.Constant][44]);
          %224 = add(%223, meta[relay.Constant][45]);
          %225 = reshape(%224, newshape=[-1, 768]);
          %226 = nn.dense(%225, meta[relay.Constant][46], units=2304);
          %227 = add(%226, meta[relay.Constant][47]);
          %228 = reshape(%227, newshape=[50, 32, 2304]);
          %229 = split(%228, indices_or_sections=[768, 1536], axis=2);
          %230 = %229.0;
          %231 = reshape(%230, newshape=[50, 32, 12, 64]);
          %232 = transpose(%231, axes=[0, 2, 1, 3]);
          %233 = %229.1;
          %234 = reshape(%233, newshape=[50, 32, 12, 64]);
          %235 = transpose(%234, axes=[0, 2, 3, 1]);
          %236 = reshape(%235, newshape=[-1, 64, 32]);
          %237 = reshape(%232, newshape=[-1, 32, 64]);
          %238 = transpose(%236, axes=[0, 2, 1]);
          %239 = nn.batch_matmul(%237, %238, out_dtype="float16", transpose_b=True);
          %240 = reshape(%239, newshape=[50, 12, 32, 32]);
          %241 = divide(%240, 8f16);
          %242 = multiply(%241, meta[relay.Constant][48]);
          %243 = subtract(%242, meta[relay.Constant][49]);
          %244 = nn.softmax(%243, axis=3);
          %245 = %229.2;
          %246 = reshape(%245, newshape=[50, 32, 12, 64]);
          %247 = transpose(%246, axes=[0, 2, 1, 3]);
          %248 = reshape(%247, newshape=[-1, 32, 64]);
          %249 = reshape(%244, newshape=[-1, 32, 32]);
          %250 = transpose(%248, axes=[0, 2, 1]);
          %251 = nn.batch_matmul(%249, %250, out_dtype="float16", transpose_b=True);
          %252 = reshape(%251, newshape=[50, 12, 32, 64]);
          %253 = transpose(%252, axes=[0, 2, 1, 3]);
          %254 = reshape(%253, newshape=[50, 32, 768]);
          %255 = reshape(%254, newshape=[-1, 768]);
          %256 = nn.dense(%255, meta[relay.Constant][50], units=768);
          %257 = add(%256, meta[relay.Constant][51]);
          %258 = reshape(%257, newshape=[50, 32, 768]);
          %259 = add(%215, %258);
          %260 = mean(%259, axis=[-1], keepdims=True);
          %261 = subtract(%259, %260);
          %262 = power(%261, 2f16);
          %263 = mean(%262, axis=[-1], keepdims=True);
          %264 = add(%263, 1e-05f16);
          %265 = sqrt(%264);
          %266 = divide(%261, %265);
          %267 = multiply(%266, meta[relay.Constant][52]);
          %268 = add(%267, meta[relay.Constant][53]);
          %269 = reshape(%268, newshape=[-1, 768]);
          %270 = nn.dense(%269, meta[relay.Constant][54], units=3072);
          %271 = add(%270, meta[relay.Constant][55]);
          %272 = reshape(%271, newshape=[50, 32, 3072]);
          %273 = power(%272, 3f16);
          %274 = multiply(%273, 0.044715f16);
          %275 = add(%272, %274);
          %276 = multiply(%275, 0.797885f16);
          %277 = tanh(%276);
          %278 = multiply(%272, 0.5f16);
          %279 = add(%277, 1f16);
          %280 = multiply(%278, %279);
          %281 = reshape(%280, newshape=[-1, 3072]);
          %282 = nn.dense(%281, meta[relay.Constant][56], units=768);
          %283 = add(%282, meta[relay.Constant][57]);
          %284 = reshape(%283, newshape=[50, 32, 768]);
          %285 = add(%259, %284);
          %286 = mean(%285, axis=[-1], keepdims=True);
          %287 = subtract(%285, %286);
          %288 = power(%287, 2f16);
          %289 = mean(%288, axis=[-1], keepdims=True);
          %290 = add(%289, 1e-05f16);
          %291 = sqrt(%290);
          %292 = divide(%287, %291);
          %293 = multiply(%292, meta[relay.Constant][58]);
          %294 = add(%293, meta[relay.Constant][59]);
          %295 = reshape(%294, newshape=[-1, 768]);
          %296 = nn.dense(%295, meta[relay.Constant][60], units=2304);
          %297 = add(%296, meta[relay.Constant][61]);
          %298 = reshape(%297, newshape=[50, 32, 2304]);
          %299 = split(%298, indices_or_sections=[768, 1536], axis=2);
          %300 = %299.0;
          %301 = reshape(%300, newshape=[50, 32, 12, 64]);
          %302 = transpose(%301, axes=[0, 2, 1, 3]);
          %303 = %299.1;
          %304 = reshape(%303, newshape=[50, 32, 12, 64]);
          %305 = transpose(%304, axes=[0, 2, 3, 1]);
          %306 = reshape(%305, newshape=[-1, 64, 32]);
          %307 = reshape(%302, newshape=[-1, 32, 64]);
          %308 = transpose(%306, axes=[0, 2, 1]);
          %309 = nn.batch_matmul(%307, %308, out_dtype="float16", transpose_b=True);
          %310 = reshape(%309, newshape=[50, 12, 32, 32]);
          %311 = divide(%310, 8f16);
          %312 = multiply(%311, meta[relay.Constant][62]);
          %313 = subtract(%312, meta[relay.Constant][63]);
          %314 = nn.softmax(%313, axis=3);
          %315 = %299.2;
          %316 = reshape(%315, newshape=[50, 32, 12, 64]);
          %317 = transpose(%316, axes=[0, 2, 1, 3]);
          %318 = reshape(%317, newshape=[-1, 32, 64]);
          %319 = reshape(%314, newshape=[-1, 32, 32]);
          %320 = transpose(%318, axes=[0, 2, 1]);
          %321 = nn.batch_matmul(%319, %320, out_dtype="float16", transpose_b=True);
          %322 = reshape(%321, newshape=[50, 12, 32, 64]);
          %323 = transpose(%322, axes=[0, 2, 1, 3]);
          %324 = reshape(%323, newshape=[50, 32, 768]);
          %325 = reshape(%324, newshape=[-1, 768]);
          %326 = nn.dense(%325, meta[relay.Constant][64], units=768);
          %327 = add(%326, meta[relay.Constant][65]);
          %328 = reshape(%327, newshape=[50, 32, 768]);
          %329 = add(%285, %328);
          %330 = mean(%329, axis=[-1], keepdims=True);
          %331 = subtract(%329, %330);
          %332 = power(%331, 2f16);
          %333 = mean(%332, axis=[-1], keepdims=True);
          %334 = add(%333, 1e-05f16);
          %335 = sqrt(%334);
          %336 = divide(%331, %335);
          %337 = multiply(%336, meta[relay.Constant][66]);
          %338 = add(%337, meta[relay.Constant][67]);
          %339 = reshape(%338, newshape=[-1, 768]);
          %340 = nn.dense(%339, meta[relay.Constant][68], units=3072);
          %341 = add(%340, meta[relay.Constant][69]);
          %342 = reshape(%341, newshape=[50, 32, 3072]);
          %343 = power(%342, 3f16);
          %344 = multiply(%343, 0.044715f16);
          %345 = add(%342, %344);
          %346 = multiply(%345, 0.797885f16);
          %347 = tanh(%346);
          %348 = multiply(%342, 0.5f16);
          %349 = add(%347, 1f16);
          %350 = multiply(%348, %349);
          %351 = reshape(%350, newshape=[-1, 3072]);
          %352 = nn.dense(%351, meta[relay.Constant][70], units=768);
          %353 = add(%352, meta[relay.Constant][71]);
          %354 = reshape(%353, newshape=[50, 32, 768]);
          %355 = add(%329, %354);
          %356 = mean(%355, axis=[-1], keepdims=True);
          %357 = subtract(%355, %356);
          %358 = power(%357, 2f16);
          %359 = mean(%358, axis=[-1], keepdims=True);
          %360 = add(%359, 1e-05f16);
          %361 = sqrt(%360);
          %362 = divide(%357, %361);
          %363 = multiply(%362, meta[relay.Constant][72]);
          %364 = add(%363, meta[relay.Constant][73]);
          %365 = reshape(%364, newshape=[-1, 768]);
          %366 = nn.dense(%365, meta[relay.Constant][74], units=2304);
          %367 = add(%366, meta[relay.Constant][75]);
          %368 = reshape(%367, newshape=[50, 32, 2304]);
          %369 = split(%368, indices_or_sections=[768, 1536], axis=2);
          %370 = %369.0;
          %371 = reshape(%370, newshape=[50, 32, 12, 64]);
          %372 = transpose(%371, axes=[0, 2, 1, 3]);
          %373 = %369.1;
          %374 = reshape(%373, newshape=[50, 32, 12, 64]);
          %375 = transpose(%374, axes=[0, 2, 3, 1]);
          %376 = reshape(%375, newshape=[-1, 64, 32]);
          %377 = reshape(%372, newshape=[-1, 32, 64]);
          %378 = transpose(%376, axes=[0, 2, 1]);
          %379 = nn.batch_matmul(%377, %378, out_dtype="float16", transpose_b=True);
          %380 = reshape(%379, newshape=[50, 12, 32, 32]);
          %381 = divide(%380, 8f16);
          %382 = multiply(%381, meta[relay.Constant][76]);
          %383 = subtract(%382, meta[relay.Constant][77]);
          %384 = nn.softmax(%383, axis=3);
          %385 = %369.2;
          %386 = reshape(%385, newshape=[50, 32, 12, 64]);
          %387 = transpose(%386, axes=[0, 2, 1, 3]);
          %388 = reshape(%387, newshape=[-1, 32, 64]);
          %389 = reshape(%384, newshape=[-1, 32, 32]);
          %390 = transpose(%388, axes=[0, 2, 1]);
          %391 = nn.batch_matmul(%389, %390, out_dtype="float16", transpose_b=True);
          %392 = reshape(%391, newshape=[50, 12, 32, 64]);
          %393 = transpose(%392, axes=[0, 2, 1, 3]);
          %394 = reshape(%393, newshape=[50, 32, 768]);
          %395 = reshape(%394, newshape=[-1, 768]);
          %396 = nn.dense(%395, meta[relay.Constant][78], units=768);
          %397 = add(%396, meta[relay.Constant][79]);
          %398 = reshape(%397, newshape=[50, 32, 768]);
          %399 = add(%355, %398);
          %400 = mean(%399, axis=[-1], keepdims=True);
          %401 = subtract(%399, %400);
          %402 = power(%401, 2f16);
          %403 = mean(%402, axis=[-1], keepdims=True);
          %404 = add(%403, 1e-05f16);
          %405 = sqrt(%404);
          %406 = divide(%401, %405);
          %407 = multiply(%406, meta[relay.Constant][80]);
          %408 = add(%407, meta[relay.Constant][81]);
          %409 = reshape(%408, newshape=[-1, 768]);
          %410 = nn.dense(%409, meta[relay.Constant][82], units=3072);
          %411 = add(%410, meta[relay.Constant][83]);
          %412 = reshape(%411, newshape=[50, 32, 3072]);
          %413 = power(%412, 3f16);
          %414 = multiply(%413, 0.044715f16);
          %415 = add(%412, %414);
          %416 = multiply(%415, 0.797885f16);
          %417 = tanh(%416);
          %418 = multiply(%412, 0.5f16);
          %419 = add(%417, 1f16);
          %420 = multiply(%418, %419);
          %421 = reshape(%420, newshape=[-1, 3072]);
          %422 = nn.dense(%421, meta[relay.Constant][84], units=768);
          %423 = add(%422, meta[relay.Constant][85]);
          %424 = reshape(%423, newshape=[50, 32, 768]);
          %425 = add(%399, %424);
          %426 = mean(%425, axis=[-1], keepdims=True);
          %427 = subtract(%425, %426);
          %428 = power(%427, 2f16);
          %429 = mean(%428, axis=[-1], keepdims=True);
          %430 = add(%429, 1e-05f16);
          %431 = sqrt(%430);
          %432 = divide(%427, %431);
          %433 = multiply(%432, meta[relay.Constant][86]);
          %434 = add(%433, meta[relay.Constant][87]);
          %435 = reshape(%434, newshape=[-1, 768]);
          %436 = nn.dense(%435, meta[relay.Constant][88], units=2304);
          %437 = add(%436, meta[relay.Constant][89]);
          %438 = reshape(%437, newshape=[50, 32, 2304]);
          %439 = split(%438, indices_or_sections=[768, 1536], axis=2);
          %440 = %439.0;
          %441 = reshape(%440, newshape=[50, 32, 12, 64]);
          %442 = transpose(%441, axes=[0, 2, 1, 3]);
          %443 = %439.1;
          %444 = reshape(%443, newshape=[50, 32, 12, 64]);
          %445 = transpose(%444, axes=[0, 2, 3, 1]);
          %446 = reshape(%445, newshape=[-1, 64, 32]);
          %447 = reshape(%442, newshape=[-1, 32, 64]);
          %448 = transpose(%446, axes=[0, 2, 1]);
          %449 = nn.batch_matmul(%447, %448, out_dtype="float16", transpose_b=True);
          %450 = reshape(%449, newshape=[50, 12, 32, 32]);
          %451 = divide(%450, 8f16);
          %452 = multiply(%451, meta[relay.Constant][90]);
          %453 = subtract(%452, meta[relay.Constant][91]);
          %454 = nn.softmax(%453, axis=3);
          %455 = %439.2;
          %456 = reshape(%455, newshape=[50, 32, 12, 64]);
          %457 = transpose(%456, axes=[0, 2, 1, 3]);
          %458 = reshape(%457, newshape=[-1, 32, 64]);
          %459 = reshape(%454, newshape=[-1, 32, 32]);
          %460 = transpose(%458, axes=[0, 2, 1]);
          %461 = nn.batch_matmul(%459, %460, out_dtype="float16", transpose_b=True);
          %462 = reshape(%461, newshape=[50, 12, 32, 64]);
          %463 = transpose(%462, axes=[0, 2, 1, 3]);
          %464 = reshape(%463, newshape=[50, 32, 768]);
          %465 = reshape(%464, newshape=[-1, 768]);
          %466 = nn.dense(%465, meta[relay.Constant][92], units=768);
          %467 = add(%466, meta[relay.Constant][93]);
          %468 = reshape(%467, newshape=[50, 32, 768]);
          %469 = add(%425, %468);
          %470 = mean(%469, axis=[-1], keepdims=True);
          %471 = subtract(%469, %470);
          %472 = power(%471, 2f16);
          %473 = mean(%472, axis=[-1], keepdims=True);
          %474 = add(%473, 1e-05f16);
          %475 = sqrt(%474);
          %476 = divide(%471, %475);
          %477 = multiply(%476, meta[relay.Constant][94]);
          %478 = add(%477, meta[relay.Constant][95]);
          %479 = reshape(%478, newshape=[-1, 768]);
          %480 = nn.dense(%479, meta[relay.Constant][96], units=3072);
          %481 = add(%480, meta[relay.Constant][97]);
          %482 = reshape(%481, newshape=[50, 32, 3072]);
          %483 = power(%482, 3f16);
          %484 = multiply(%483, 0.044715f16);
          %485 = add(%482, %484);
          %486 = multiply(%485, 0.797885f16);
          %487 = tanh(%486);
          %488 = multiply(%482, 0.5f16);
          %489 = add(%487, 1f16);
          %490 = multiply(%488, %489);
          %491 = reshape(%490, newshape=[-1, 3072]);
          %492 = nn.dense(%491, meta[relay.Constant][98], units=768);
          %493 = add(%492, meta[relay.Constant][99]);
          %494 = reshape(%493, newshape=[50, 32, 768]);
          %495 = add(%469, %494);
          %496 = mean(%495, axis=[-1], keepdims=True);
          %497 = subtract(%495, %496);
          %498 = power(%497, 2f16);
          %499 = mean(%498, axis=[-1], keepdims=True);
          %500 = add(%499, 1e-05f16);
          %501 = sqrt(%500);
          %502 = divide(%497, %501);
          %503 = multiply(%502, meta[relay.Constant][100]);
          %504 = add(%503, meta[relay.Constant][101]);
          %505 = reshape(%504, newshape=[-1, 768]);
          %506 = nn.dense(%505, meta[relay.Constant][102], units=2304);
          %507 = add(%506, meta[relay.Constant][103]);
          %508 = reshape(%507, newshape=[50, 32, 2304]);
          %509 = split(%508, indices_or_sections=[768, 1536], axis=2);
          %510 = %509.0;
          %511 = reshape(%510, newshape=[50, 32, 12, 64]);
          %512 = transpose(%511, axes=[0, 2, 1, 3]);
          %513 = %509.1;
          %514 = reshape(%513, newshape=[50, 32, 12, 64]);
          %515 = transpose(%514, axes=[0, 2, 3, 1]);
          %516 = reshape(%515, newshape=[-1, 64, 32]);
          %517 = reshape(%512, newshape=[-1, 32, 64]);
          %518 = transpose(%516, axes=[0, 2, 1]);
          %519 = nn.batch_matmul(%517, %518, out_dtype="float16", transpose_b=True);
          %520 = reshape(%519, newshape=[50, 12, 32, 32]);
          %521 = divide(%520, 8f16);
          %522 = multiply(%521, meta[relay.Constant][104]);
          %523 = subtract(%522, meta[relay.Constant][105]);
          %524 = nn.softmax(%523, axis=3);
          %525 = %509.2;
          %526 = reshape(%525, newshape=[50, 32, 12, 64]);
          %527 = transpose(%526, axes=[0, 2, 1, 3]);
          %528 = reshape(%527, newshape=[-1, 32, 64]);
          %529 = reshape(%524, newshape=[-1, 32, 32]);
          %530 = transpose(%528, axes=[0, 2, 1]);
          %531 = nn.batch_matmul(%529, %530, out_dtype="float16", transpose_b=True);
          %532 = reshape(%531, newshape=[50, 12, 32, 64]);
          %533 = transpose(%532, axes=[0, 2, 1, 3]);
          %534 = reshape(%533, newshape=[50, 32, 768]);
          %535 = reshape(%534, newshape=[-1, 768]);
          %536 = nn.dense(%535, meta[relay.Constant][106], units=768);
          %537 = add(%536, meta[relay.Constant][107]);
          %538 = reshape(%537, newshape=[50, 32, 768]);
          %539 = add(%495, %538);
          %540 = mean(%539, axis=[-1], keepdims=True);
          %541 = subtract(%539, %540);
          %542 = power(%541, 2f16);
          %543 = mean(%542, axis=[-1], keepdims=True);
          %544 = add(%543, 1e-05f16);
          %545 = sqrt(%544);
          %546 = divide(%541, %545);
          %547 = multiply(%546, meta[relay.Constant][108]);
          %548 = add(%547, meta[relay.Constant][109]);
          %549 = reshape(%548, newshape=[-1, 768]);
          %550 = nn.dense(%549, meta[relay.Constant][110], units=3072);
          %551 = add(%550, meta[relay.Constant][111]);
          %552 = reshape(%551, newshape=[50, 32, 3072]);
          %553 = power(%552, 3f16);
          %554 = multiply(%553, 0.044715f16);
          %555 = add(%552, %554);
          %556 = multiply(%555, 0.797885f16);
          %557 = tanh(%556);
          %558 = multiply(%552, 0.5f16);
          %559 = add(%557, 1f16);
          %560 = multiply(%558, %559);
          %561 = reshape(%560, newshape=[-1, 3072]);
          %562 = nn.dense(%561, meta[relay.Constant][112], units=768);
          %563 = add(%562, meta[relay.Constant][113]);
          %564 = reshape(%563, newshape=[50, 32, 768]);
          %565 = add(%539, %564);
          %566 = mean(%565, axis=[-1], keepdims=True);
          %567 = subtract(%565, %566);
          %568 = power(%567, 2f16);
          %569 = mean(%568, axis=[-1], keepdims=True);
          %570 = add(%569, 1e-05f16);
          %571 = sqrt(%570);
          %572 = divide(%567, %571);
          %573 = multiply(%572, meta[relay.Constant][114]);
          %574 = add(%573, meta[relay.Constant][115]);
          %575 = reshape(%574, newshape=[-1, 768]);
          %576 = nn.dense(%575, meta[relay.Constant][116], units=2304);
          %577 = add(%576, meta[relay.Constant][117]);
          %578 = reshape(%577, newshape=[50, 32, 2304]);
          %579 = split(%578, indices_or_sections=[768, 1536], axis=2);
          %580 = %579.0;
          %581 = reshape(%580, newshape=[50, 32, 12, 64]);
          %582 = transpose(%581, axes=[0, 2, 1, 3]);
          %583 = %579.1;
          %584 = reshape(%583, newshape=[50, 32, 12, 64]);
          %585 = transpose(%584, axes=[0, 2, 3, 1]);
          %586 = reshape(%585, newshape=[-1, 64, 32]);
          %587 = reshape(%582, newshape=[-1, 32, 64]);
          %588 = transpose(%586, axes=[0, 2, 1]);
          %589 = nn.batch_matmul(%587, %588, out_dtype="float16", transpose_b=True);
          %590 = reshape(%589, newshape=[50, 12, 32, 32]);
          %591 = divide(%590, 8f16);
          %592 = multiply(%591, meta[relay.Constant][118]);
          %593 = subtract(%592, meta[relay.Constant][119]);
          %594 = nn.softmax(%593, axis=3);
          %595 = %579.2;
          %596 = reshape(%595, newshape=[50, 32, 12, 64]);
          %597 = transpose(%596, axes=[0, 2, 1, 3]);
          %598 = reshape(%597, newshape=[-1, 32, 64]);
          %599 = reshape(%594, newshape=[-1, 32, 32]);
          %600 = transpose(%598, axes=[0, 2, 1]);
          %601 = nn.batch_matmul(%599, %600, out_dtype="float16", transpose_b=True);
          %602 = reshape(%601, newshape=[50, 12, 32, 64]);
          %603 = transpose(%602, axes=[0, 2, 1, 3]);
          %604 = reshape(%603, newshape=[50, 32, 768]);
          %605 = reshape(%604, newshape=[-1, 768]);
          %606 = nn.dense(%605, meta[relay.Constant][120], units=768);
          %607 = add(%606, meta[relay.Constant][121]);
          %608 = reshape(%607, newshape=[50, 32, 768]);
          %609 = add(%565, %608);
          %610 = mean(%609, axis=[-1], keepdims=True);
          %611 = subtract(%609, %610);
          %612 = power(%611, 2f16);
          %613 = mean(%612, axis=[-1], keepdims=True);
          %614 = add(%613, 1e-05f16);
          %615 = sqrt(%614);
          %616 = divide(%611, %615);
          %617 = multiply(%616, meta[relay.Constant][122]);
          %618 = add(%617, meta[relay.Constant][123]);
          %619 = reshape(%618, newshape=[-1, 768]);
          %620 = nn.dense(%619, meta[relay.Constant][124], units=3072);
          %621 = add(%620, meta[relay.Constant][125]);
          %622 = reshape(%621, newshape=[50, 32, 3072]);
          %623 = power(%622, 3f16);
          %624 = multiply(%623, 0.044715f16);
          %625 = add(%622, %624);
          %626 = multiply(%625, 0.797885f16);
          %627 = tanh(%626);
          %628 = multiply(%622, 0.5f16);
          %629 = add(%627, 1f16);
          %630 = multiply(%628, %629);
          %631 = reshape(%630, newshape=[-1, 3072]);
          %632 = nn.dense(%631, meta[relay.Constant][126], units=768);
          %633 = add(%632, meta[relay.Constant][127]);
          %634 = reshape(%633, newshape=[50, 32, 768]);
          %635 = add(%609, %634);
          %636 = mean(%635, axis=[-1], keepdims=True);
          %637 = subtract(%635, %636);
          %638 = power(%637, 2f16);
          %639 = mean(%638, axis=[-1], keepdims=True);
          %640 = add(%639, 1e-05f16);
          %641 = sqrt(%640);
          %642 = divide(%637, %641);
          %643 = multiply(%642, meta[relay.Constant][128]);
          %644 = add(%643, meta[relay.Constant][129]);
          %645 = reshape(%644, newshape=[-1, 768]);
          %646 = nn.dense(%645, meta[relay.Constant][130], units=2304);
          %647 = add(%646, meta[relay.Constant][131]);
          %648 = reshape(%647, newshape=[50, 32, 2304]);
          %649 = split(%648, indices_or_sections=[768, 1536], axis=2);
          %650 = %649.0;
          %651 = reshape(%650, newshape=[50, 32, 12, 64]);
          %652 = transpose(%651, axes=[0, 2, 1, 3]);
          %653 = %649.1;
          %654 = reshape(%653, newshape=[50, 32, 12, 64]);
          %655 = transpose(%654, axes=[0, 2, 3, 1]);
          %656 = reshape(%655, newshape=[-1, 64, 32]);
          %657 = reshape(%652, newshape=[-1, 32, 64]);
          %658 = transpose(%656, axes=[0, 2, 1]);
          %659 = nn.batch_matmul(%657, %658, out_dtype="float16", transpose_b=True);
          %660 = reshape(%659, newshape=[50, 12, 32, 32]);
          %661 = divide(%660, 8f16);
          %662 = multiply(%661, meta[relay.Constant][132]);
          %663 = subtract(%662, meta[relay.Constant][133]);
          %664 = nn.softmax(%663, axis=3);
          %665 = %649.2;
          %666 = reshape(%665, newshape=[50, 32, 12, 64]);
          %667 = transpose(%666, axes=[0, 2, 1, 3]);
          %668 = reshape(%667, newshape=[-1, 32, 64]);
          %669 = reshape(%664, newshape=[-1, 32, 32]);
          %670 = transpose(%668, axes=[0, 2, 1]);
          %671 = nn.batch_matmul(%669, %670, out_dtype="float16", transpose_b=True);
          %672 = reshape(%671, newshape=[50, 12, 32, 64]);
          %673 = transpose(%672, axes=[0, 2, 1, 3]);
          %674 = reshape(%673, newshape=[50, 32, 768]);
          %675 = reshape(%674, newshape=[-1, 768]);
          %676 = nn.dense(%675, meta[relay.Constant][134], units=768);
          %677 = add(%676, meta[relay.Constant][135]);
          %678 = reshape(%677, newshape=[50, 32, 768]);
          %679 = add(%635, %678);
          %680 = mean(%679, axis=[-1], keepdims=True);
          %681 = subtract(%679, %680);
          %682 = power(%681, 2f16);
          %683 = mean(%682, axis=[-1], keepdims=True);
          %684 = add(%683, 1e-05f16);
          %685 = sqrt(%684);
          %686 = divide(%681, %685);
          %687 = multiply(%686, meta[relay.Constant][136]);
          %688 = add(%687, meta[relay.Constant][137]);
          %689 = reshape(%688, newshape=[-1, 768]);
          %690 = nn.dense(%689, meta[relay.Constant][138], units=3072);
          %691 = add(%690, meta[relay.Constant][139]);
          %692 = reshape(%691, newshape=[50, 32, 3072]);
          %693 = power(%692, 3f16);
          %694 = multiply(%693, 0.044715f16);
          %695 = add(%692, %694);
          %696 = multiply(%695, 0.797885f16);
          %697 = tanh(%696);
          %698 = multiply(%692, 0.5f16);
          %699 = add(%697, 1f16);
          %700 = multiply(%698, %699);
          %701 = reshape(%700, newshape=[-1, 3072]);
          %702 = nn.dense(%701, meta[relay.Constant][140], units=768);
          %703 = add(%702, meta[relay.Constant][141]);
          %704 = reshape(%703, newshape=[50, 32, 768]);
          %705 = add(%679, %704);
          %706 = mean(%705, axis=[-1], keepdims=True);
          %707 = subtract(%705, %706);
          %708 = power(%707, 2f16);
          %709 = mean(%708, axis=[-1], keepdims=True);
          %710 = add(%709, 1e-05f16);
          %711 = sqrt(%710);
          %712 = divide(%707, %711);
          %713 = multiply(%712, meta[relay.Constant][142]);
          %714 = add(%713, meta[relay.Constant][143]);
          %715 = reshape(%714, newshape=[-1, 768]);
          %716 = nn.dense(%715, meta[relay.Constant][144], units=2304);
          %717 = add(%716, meta[relay.Constant][145]);
          %718 = reshape(%717, newshape=[50, 32, 2304]);
          %719 = split(%718, indices_or_sections=[768, 1536], axis=2);
          %720 = %719.0;
          %721 = reshape(%720, newshape=[50, 32, 12, 64]);
          %722 = transpose(%721, axes=[0, 2, 1, 3]);
          %723 = %719.1;
          %724 = reshape(%723, newshape=[50, 32, 12, 64]);
          %725 = transpose(%724, axes=[0, 2, 3, 1]);
          %726 = reshape(%725, newshape=[-1, 64, 32]);
          %727 = reshape(%722, newshape=[-1, 32, 64]);
          %728 = transpose(%726, axes=[0, 2, 1]);
          %729 = nn.batch_matmul(%727, %728, out_dtype="float16", transpose_b=True);
          %730 = reshape(%729, newshape=[50, 12, 32, 32]);
          %731 = divide(%730, 8f16);
          %732 = multiply(%731, meta[relay.Constant][146]);
          %733 = subtract(%732, meta[relay.Constant][147]);
          %734 = nn.softmax(%733, axis=3);
          %735 = %719.2;
          %736 = reshape(%735, newshape=[50, 32, 12, 64]);
          %737 = transpose(%736, axes=[0, 2, 1, 3]);
          %738 = reshape(%737, newshape=[-1, 32, 64]);
          %739 = reshape(%734, newshape=[-1, 32, 32]);
          %740 = transpose(%738, axes=[0, 2, 1]);
          %741 = nn.batch_matmul(%739, %740, out_dtype="float16", transpose_b=True);
          %742 = reshape(%741, newshape=[50, 12, 32, 64]);
          %743 = transpose(%742, axes=[0, 2, 1, 3]);
          %744 = reshape(%743, newshape=[50, 32, 768]);
          %745 = reshape(%744, newshape=[-1, 768]);
          %746 = nn.dense(%745, meta[relay.Constant][148], units=768);
          %747 = add(%746, meta[relay.Constant][149]);
          %748 = reshape(%747, newshape=[50, 32, 768]);
          %749 = add(%705, %748);
          %750 = mean(%749, axis=[-1], keepdims=True);
          %751 = subtract(%749, %750);
          %752 = power(%751, 2f16);
          %753 = mean(%752, axis=[-1], keepdims=True);
          %754 = add(%753, 1e-05f16);
          %755 = sqrt(%754);
          %756 = divide(%751, %755);
          %757 = multiply(%756, meta[relay.Constant][150]);
          %758 = add(%757, meta[relay.Constant][151]);
          %759 = reshape(%758, newshape=[-1, 768]);
          %760 = nn.dense(%759, meta[relay.Constant][152], units=3072);
          %761 = add(%760, meta[relay.Constant][153]);
          %762 = reshape(%761, newshape=[50, 32, 3072]);
          %763 = power(%762, 3f16);
          %764 = multiply(%763, 0.044715f16);
          %765 = add(%762, %764);
          %766 = multiply(%765, 0.797885f16);
          %767 = tanh(%766);
          %768 = multiply(%762, 0.5f16);
          %769 = add(%767, 1f16);
          %770 = multiply(%768, %769);
          %771 = reshape(%770, newshape=[-1, 3072]);
          %772 = nn.dense(%771, meta[relay.Constant][154], units=768);
          %773 = add(%772, meta[relay.Constant][155]);
          %774 = reshape(%773, newshape=[50, 32, 768]);
          %775 = add(%749, %774);
          %776 = mean(%775, axis=[-1], keepdims=True);
          %777 = subtract(%775, %776);
          %778 = power(%777, 2f16);
          %779 = mean(%778, axis=[-1], keepdims=True);
          %780 = add(%779, 1e-05f16);
          %781 = sqrt(%780);
          %782 = divide(%777, %781);
          %783 = multiply(%782, meta[relay.Constant][156]);
          %784 = add(%783, meta[relay.Constant][157]);
          %785 = reshape(%784, newshape=[-1, 768]);
          %786 = nn.dense(%785, meta[relay.Constant][158], units=2304);
          %787 = add(%786, meta[relay.Constant][159]);
          %788 = reshape(%787, newshape=[50, 32, 2304]);
          %789 = split(%788, indices_or_sections=[768, 1536], axis=2);
          %790 = %789.0;
          %791 = reshape(%790, newshape=[50, 32, 12, 64]);
          %792 = transpose(%791, axes=[0, 2, 1, 3]);
          %793 = %789.1;
          %794 = reshape(%793, newshape=[50, 32, 12, 64]);
          %795 = transpose(%794, axes=[0, 2, 3, 1]);
          %796 = reshape(%795, newshape=[-1, 64, 32]);
          %797 = reshape(%792, newshape=[-1, 32, 64]);
          %798 = transpose(%796, axes=[0, 2, 1]);
          %799 = nn.batch_matmul(%797, %798, out_dtype="float16", transpose_b=True);
          %800 = reshape(%799, newshape=[50, 12, 32, 32]);
          %801 = divide(%800, 8f16);
          %802 = multiply(%801, meta[relay.Constant][160]);
          %803 = subtract(%802, meta[relay.Constant][161]);
          %804 = nn.softmax(%803, axis=3);
          %805 = %789.2;
          %806 = reshape(%805, newshape=[50, 32, 12, 64]);
          %807 = transpose(%806, axes=[0, 2, 1, 3]);
          %808 = reshape(%807, newshape=[-1, 32, 64]);
          %809 = reshape(%804, newshape=[-1, 32, 32]);
          %810 = transpose(%808, axes=[0, 2, 1]);
          %811 = nn.batch_matmul(%809, %810, out_dtype="float16", transpose_b=True);
          %812 = reshape(%811, newshape=[50, 12, 32, 64]);
          %813 = transpose(%812, axes=[0, 2, 1, 3]);
          %814 = reshape(%813, newshape=[50, 32, 768]);
          %815 = reshape(%814, newshape=[-1, 768]);
          %816 = nn.dense(%815, meta[relay.Constant][162], units=768);
          %817 = add(%816, meta[relay.Constant][163]);
          %818 = reshape(%817, newshape=[50, 32, 768]);
          %819 = add(%775, %818);
          %820 = mean(%819, axis=[-1], keepdims=True);
          %821 = subtract(%819, %820);
          %822 = power(%821, 2f16);
          %823 = mean(%822, axis=[-1], keepdims=True);
          %824 = add(%823, 1e-05f16);
          %825 = sqrt(%824);
          %826 = divide(%821, %825);
          %827 = multiply(%826, meta[relay.Constant][164]);
          %828 = add(%827, meta[relay.Constant][165]);
          %829 = reshape(%828, newshape=[-1, 768]);
          %830 = nn.dense(%829, meta[relay.Constant][166], units=3072);
          %831 = add(%830, meta[relay.Constant][167]);
          %832 = reshape(%831, newshape=[50, 32, 3072]);
          %833 = power(%832, 3f16);
          %834 = multiply(%833, 0.044715f16);
          %835 = add(%832, %834);
          %836 = multiply(%835, 0.797885f16);
          %837 = tanh(%836);
          %838 = multiply(%832, 0.5f16);
          %839 = add(%837, 1f16);
          %840 = multiply(%838, %839);
          %841 = reshape(%840, newshape=[-1, 3072]);
          %842 = nn.dense(%841, meta[relay.Constant][168], units=768);
          %843 = add(%842, meta[relay.Constant][169]);
          %844 = reshape(%843, newshape=[50, 32, 768]);
          %845 = add(%819, %844);
          %846 = mean(%845, axis=[-1], keepdims=True);
          %847 = subtract(%845, %846);
          %848 = power(%847, 2f16);
          %849 = mean(%848, axis=[-1], keepdims=True);
          %850 = add(%849, 1e-05f16);
          %851 = sqrt(%850);
          %852 = divide(%847, %851);
          %853 = multiply(%852, meta[relay.Constant][170]);
          %854 = add(%853, meta[relay.Constant][171]);
          %855 = transpose(%24, axes=[0, 2, 1, 3]);
          %856 = expand_dims(%855, axis=0);
          %857 = expand_dims(%37, axis=0);
          %858 = (%856, %857);
          %859 = transpose(%94, axes=[0, 2, 1, 3]);
          %860 = expand_dims(%859, axis=0);
          %861 = expand_dims(%107, axis=0);
          %862 = (%860, %861);
          %863 = transpose(%164, axes=[0, 2, 1, 3]);
          %864 = expand_dims(%863, axis=0);
          %865 = expand_dims(%177, axis=0);
          %866 = (%864, %865);
          %867 = transpose(%234, axes=[0, 2, 1, 3]);
          %868 = expand_dims(%867, axis=0);
          %869 = expand_dims(%247, axis=0);
          %870 = (%868, %869);
          %871 = transpose(%304, axes=[0, 2, 1, 3]);
          %872 = expand_dims(%871, axis=0);
          %873 = expand_dims(%317, axis=0);
          %874 = (%872, %873);
          %875 = transpose(%374, axes=[0, 2, 1, 3]);
          %876 = expand_dims(%875, axis=0);
          %877 = expand_dims(%387, axis=0);
          %878 = (%876, %877);
          %879 = transpose(%444, axes=[0, 2, 1, 3]);
          %880 = expand_dims(%879, axis=0);
          %881 = expand_dims(%457, axis=0);
          %882 = (%880, %881);
          %883 = transpose(%514, axes=[0, 2, 1, 3]);
          %884 = expand_dims(%883, axis=0);
          %885 = expand_dims(%527, axis=0);
          %886 = (%884, %885);
          %887 = transpose(%584, axes=[0, 2, 1, 3]);
          %888 = expand_dims(%887, axis=0);
          %889 = expand_dims(%597, axis=0);
          %890 = (%888, %889);
          %891 = transpose(%654, axes=[0, 2, 1, 3]);
          %892 = expand_dims(%891, axis=0);
          %893 = expand_dims(%667, axis=0);
          %894 = (%892, %893);
          %895 = transpose(%724, axes=[0, 2, 1, 3]);
          %896 = expand_dims(%895, axis=0);
          %897 = expand_dims(%737, axis=0);
          %898 = (%896, %897);
          %899 = transpose(%794, axes=[0, 2, 1, 3]);
          %900 = expand_dims(%899, axis=0);
          %901 = expand_dims(%807, axis=0);
          %902 = (%900, %901);
          %903 = reshape(%854, newshape=[1, 50, 32, 768]);
          %904 = concatenate(%858);
          %905 = concatenate(%862);
          %906 = concatenate(%866);
          %907 = concatenate(%870);
          %908 = concatenate(%874);
          %909 = concatenate(%878);
          %910 = concatenate(%882);
          %911 = concatenate(%886);
          %912 = concatenate(%890);
          %913 = concatenate(%894);
          %914 = concatenate(%898);
          %915 = concatenate(%902);
          (%903, %904, %905, %906, %907, %908, %909, %910, %911, %912, %913, %914, %915)
        }
        """,
        "from_string",
        None,
        metatable,
    )
    return {
        "name": "gpt2_16",
        "input_shapes": {"x": [1, 50, 32]},
        "input_dtypes": {"x": "int64"},
        "mod": mod,
        "params": None,
        "main_dtype": "float16",
    }


def gpt2_extract_consts(dtype):
    return make_consts(
        dtype,
        [
            (768, 768),  # 0
            (768,),  # 1
            (768,),  # 2
            (768,),  # 3
            (3072, 768),  # 4
            (3072,),  # 5
            (1, 32, 768),  # 6
        ],
    )


def gpt2_extract():
    metatable = {"relay.Constant": gpt2_extract_consts("float32")}
    mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%x: Tensor[(1600, 768), float32]) -> Tensor[(50, 32, 3072), float32] {
            %46 = nn.dense(%x, meta[relay.Constant][0], units=768);
            %47 = add(%46, meta[relay.Constant][1]);
            %48 = reshape(%47, newshape=[50, 32, 768]);
            %49 = add(meta[relay.Constant][6], %48);
            %50 = mean(%49, axis=[-1], keepdims=True);
            %51 = subtract(%49, %50);
            %52 = power(%51, 2f);
            %53 = mean(%52, axis=[-1], keepdims=True);
            %54 = add(%53, 1e-05f);
            %55 = sqrt(%54);
            %56 = divide(%51, %55);
            %57 = multiply(%56, meta[relay.Constant][2]);
            %58 = add(%57, meta[relay.Constant][3]);
            %59 = reshape(%58, newshape=[-1, 768]);
            %60 = nn.dense(%59, meta[relay.Constant][4], units=3072);
            %61 = add(%60, meta[relay.Constant][5]);
            %62 = reshape(%61, newshape=[50, 32, 3072]);
            %63 = power(%62, 3f);
            %64 = multiply(%63, 0.044715f);
            %65 = add(%62, %64);
            %66 = multiply(%65, 0.797885f);
            %67 = tanh(%66);
            %68 = multiply(%62, 0.5f);
            %69 = add(%67, 1f);
            %70 = multiply(%68, %69);
            %70
        }
        """,
        "from_string",
        None,
        metatable,
    )
    return {
        "input_shapes": {"x": [1600, 768]},
        "input_dtypes": {"x": "float32"},
        "mod": mod,
        "params": None,
        "main_dtype": "float32",
    }


def gpt2_extract_16():
    metatable = {"relay.Constant": gpt2_extract_consts("float16")}
    mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%x: Tensor[(1600, 768), float16]) -> Tensor[(50, 32, 3072), float16] {
            %46 = nn.dense(%x, meta[relay.Constant][0], units=768);
            %47 = add(%46, meta[relay.Constant][1]);
            %48 = reshape(%47, newshape=[50, 32, 768]);
            %49 = add(meta[relay.Constant][6], %48);
            %50 = mean(%49, axis=[-1], keepdims=True);
            %51 = subtract(%49, %50);
            %52 = power(%51, 2f16);
            %53 = mean(%52, axis=[-1], keepdims=True);
            %54 = add(%53, 1e-05f16);
            %55 = sqrt(%54);
            %56 = divide(%51, %55);
            %57 = multiply(%56, meta[relay.Constant][2]);
            %58 = add(%57, meta[relay.Constant][3]);
            %59 = reshape(%58, newshape=[-1, 768]);
            %60 = nn.dense(%59, meta[relay.Constant][4], units=3072);
            %61 = add(%60, meta[relay.Constant][5]);
            %62 = reshape(%61, newshape=[50, 32, 3072]);
            %63 = power(%62, 3f16);
            %64 = multiply(%63, 0.044715f16);
            %65 = add(%62, %64);
            %66 = multiply(%65, 0.797885f16);
            %67 = tanh(%66);
            %68 = multiply(%62, 0.5f16);
            %69 = add(%67, 1f16);
            %70 = multiply(%68, %69);
            %70
        }
        """,
        "from_string",
        None,
        metatable,
    )
    return {
        "name": "gpt2_extract_16",
        "input_shapes": {"x": [1600, 768]},
        "input_dtypes": {"x": "float16"},
        "mod": mod,
        "params": None,
        "main_dtype": "float16",
    }


def gpt2_16_for_cutlass_extract_consts(dtype):
    return make_consts(
        "float16",
        [
            (2304, 768),  # 0
            (2304,),  # 1
            (600, 32, 64),  # 2
            (600, 32, 32),  # 3
        ],
    )


def gpt2_16_for_cutlass_extract():
    metatable = {"relay.Constant": gpt2_16_for_cutlass_extract_consts("float16")}
    mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%x0: Tensor[(1600, 768), float16],
                  %x3: Tensor[(600, 32, 64), float16])
            -> (Tensor[(1600, 2304), float16], Tensor[(1200, 32, 32), float16]) {
          %0 = nn.dense(%x0, meta[relay.Constant][0], units=2304);
          %1 = add(%0, meta[relay.Constant][1]);
          %2 = nn.batch_matmul(%x3, meta[relay.Constant][2], out_dtype="float16", transpose_b=True);
          %3 = (%2, meta[relay.Constant][3]);
          %4 = concatenate(%3);
          (%1, %4)
        }
        """,
        "from_string",
        None,
        metatable,
    )
    return {
        "name": "gpt2_16_for_cutlass_extract",
        "input_shapes": {"x0": (1600, 768), "x3": (600, 32, 64)},
        "input_dtypes": {"x0": "float16", "x3": "float16"},
        "mod": mod,
        "params": None,
        "main_dtype": "float16",
    }


def resnet50_consts(dtype):
    return make_consts(
        dtype,
        [
            (3,),  # 0
            (3,),  # 1
            (3,),  # 2
            (3,),  # 3
            (64, 3, 7, 7),  # 4
            (64,),  # 5
            (64,),  # 6
            (64,),  # 7
            (64,),  # 8
            (64,),  # 9
            (64,),  # 10
            (64,),  # 11
            (64,),  # 12
            (64, 64, 1, 1),  # 13
            (64,),  # 14
            (64,),  # 15
            (64,),  # 16
            (64,),  # 17
            (64, 64, 3, 3),  # 18
            (64,),  # 19
            (64,),  # 20
            (64,),  # 21
            (64,),  # 22
            (256, 64, 1, 1),  # 23
            (256, 64, 1, 1),  # 24
            (256,),  # 25
            (256,),  # 26
            (256,),  # 27
            (256,),  # 28
            (64, 256, 1, 1),  # 29
            (64,),  # 30
            (64,),  # 31
            (64,),  # 32
            (64,),  # 33
            (64, 64, 3, 3),  # 34
            (64,),  # 35
            (64,),  # 36
            (64,),  # 37
            (64,),  # 38
            (256, 64, 1, 1),  # 39
            (256,),  # 40
            (256,),  # 41
            (256,),  # 42
            (256,),  # 43
            (64, 256, 1, 1),  # 44
            (64,),  # 45
            (64,),  # 46
            (64,),  # 47
            (64,),  # 48
            (64, 64, 3, 3),  # 49
            (64,),  # 50
            (64,),  # 51
            (64,),  # 52
            (64,),  # 53
            (256, 64, 1, 1),  # 54
            (256,),  # 55
            (256,),  # 56
            (256,),  # 57
            (256,),  # 58
            (128, 256, 1, 1),  # 59
            (128,),  # 60
            (128,),  # 61
            (128,),  # 62
            (128,),  # 63
            (128, 128, 3, 3),  # 64
            (128,),  # 65
            (128,),  # 66
            (128,),  # 67
            (128,),  # 68
            (512, 128, 1, 1),  # 69
            (512, 256, 1, 1),  # 70
            (512,),  # 71
            (512,),  # 72
            (512,),  # 73
            (512,),  # 74
            (128, 512, 1, 1),  # 75
            (128,),  # 76
            (128,),  # 77
            (128,),  # 78
            (128,),  # 79
            (128, 128, 3, 3),  # 80
            (128,),  # 81
            (128,),  # 82
            (128,),  # 83
            (128,),  # 84
            (512, 128, 1, 1),  # 85
            (512,),  # 86
            (512,),  # 87
            (512,),  # 88
            (512,),  # 89
            (128, 512, 1, 1),  # 90
            (128,),  # 91
            (128,),  # 92
            (128,),  # 93
            (128,),  # 94
            (128, 128, 3, 3),  # 95
            (128,),  # 96
            (128,),  # 97
            (128,),  # 98
            (128,),  # 99
            (512, 128, 1, 1),  # 100
            (512,),  # 101
            (512,),  # 102
            (512,),  # 103
            (512,),  # 104
            (128, 512, 1, 1),  # 105
            (128,),  # 106
            (128,),  # 107
            (128,),  # 108
            (128,),  # 109
            (128, 128, 3, 3),  # 110
            (128,),  # 111
            (128,),  # 112
            (128,),  # 113
            (128,),  # 114
            (512, 128, 1, 1),  # 115
            (512,),  # 116
            (512,),  # 117
            (512,),  # 118
            (512,),  # 119
            (256, 512, 1, 1),  # 120
            (256,),  # 121
            (256,),  # 122
            (256,),  # 123
            (256,),  # 124
            (256, 256, 3, 3),  # 125
            (256,),  # 126
            (256,),  # 127
            (256,),  # 128
            (256,),  # 129
            (1024, 256, 1, 1),  # 130
            (1024, 512, 1, 1),  # 131
            (1024,),  # 132
            (1024,),  # 133
            (1024,),  # 134
            (1024,),  # 135
            (256, 1024, 1, 1),  # 136
            (256,),  # 137
            (256,),  # 138
            (256,),  # 139
            (256,),  # 140
            (256, 256, 3, 3),  # 141
            (256,),  # 142
            (256,),  # 143
            (256,),  # 144
            (256,),  # 145
            (1024, 256, 1, 1),  # 146
            (1024,),  # 147
            (1024,),  # 148
            (1024,),  # 149
            (1024,),  # 150
            (256, 1024, 1, 1),  # 151
            (256,),  # 152
            (256,),  # 153
            (256,),  # 154
            (256,),  # 155
            (256, 256, 3, 3),  # 156
            (256,),  # 157
            (256,),  # 158
            (256,),  # 159
            (256,),  # 160
            (1024, 256, 1, 1),  # 161
            (1024,),  # 162
            (1024,),  # 163
            (1024,),  # 164
            (1024,),  # 165
            (256, 1024, 1, 1),  # 166
            (256,),  # 167
            (256,),  # 168
            (256,),  # 169
            (256,),  # 170
            (256, 256, 3, 3),  # 171
            (256,),  # 172
            (256,),  # 173
            (256,),  # 174
            (256,),  # 175
            (1024, 256, 1, 1),  # 176
            (1024,),  # 177
            (1024,),  # 178
            (1024,),  # 179
            (1024,),  # 180
            (256, 1024, 1, 1),  # 181
            (256,),  # 182
            (256,),  # 183
            (256,),  # 184
            (256,),  # 185
            (256, 256, 3, 3),  # 186
            (256,),  # 187
            (256,),  # 188
            (256,),  # 189
            (256,),  # 190
            (1024, 256, 1, 1),  # 191
            (1024,),  # 192
            (1024,),  # 193
            (1024,),  # 194
            (1024,),  # 195
            (256, 1024, 1, 1),  # 196
            (256,),  # 197
            (256,),  # 198
            (256,),  # 199
            (256,),  # 200
            (256, 256, 3, 3),  # 201
            (256,),  # 202
            (256,),  # 203
            (256,),  # 204
            (256,),  # 205
            (1024, 256, 1, 1),  # 206
            (1024,),  # 207
            (1024,),  # 208
            (1024,),  # 209
            (1024,),  # 210
            (512, 1024, 1, 1),  # 211
            (512,),  # 212
            (512,),  # 213
            (512,),  # 214
            (512,),  # 215
            (512, 512, 3, 3),  # 216
            (512,),  # 217
            (512,),  # 218
            (512,),  # 219
            (512,),  # 220
            (2048, 512, 1, 1),  # 221
            (2048, 1024, 1, 1),  # 222
            (2048,),  # 223
            (2048,),  # 224
            (2048,),  # 225
            (2048,),  # 226
            (512, 2048, 1, 1),  # 227
            (512,),  # 228
            (512,),  # 229
            (512,),  # 230
            (512,),  # 231
            (512, 512, 3, 3),  # 232
            (512,),  # 233
            (512,),  # 234
            (512,),  # 235
            (512,),  # 236
            (2048, 512, 1, 1),  # 237
            (2048,),  # 238
            (2048,),  # 239
            (2048,),  # 240
            (2048,),  # 241
            (512, 2048, 1, 1),  # 242
            (512,),  # 243
            (512,),  # 244
            (512,),  # 245
            (512,),  # 246
            (512, 512, 3, 3),  # 247
            (512,),  # 248
            (512,),  # 249
            (512,),  # 250
            (512,),  # 251
            (2048, 512, 1, 1),  # 252
            (2048,),  # 253
            (2048,),  # 254
            (2048,),  # 255
            (2048,),  # 256
            (1000, 2048),  # 257
            (1000,),  # 258
        ],
    )


def resnet50():
    metatable = {"relay.Constant": resnet50_consts("float32")}
    mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%data: Tensor[(1, 3, 224, 224), float32]) -> Tensor[(1, 1000), float32] {
          %0 = nn.batch_norm(%data, meta[relay.Constant][0], meta[relay.Constant][1], meta[relay.Constant][2], meta[relay.Constant][3]);
          %1 = %0.0;
          %2 = nn.conv2d(%1, meta[relay.Constant][4], strides=[2, 2], padding=[3, 3, 3, 3], channels=64, kernel_size=[7, 7]);
          %3 = nn.batch_norm(%2, meta[relay.Constant][5], meta[relay.Constant][6], meta[relay.Constant][7], meta[relay.Constant][8]);
          %4 = %3.0;
          %5 = nn.relu(%4);
          %6 = nn.max_pool2d(%5, pool_size=[3, 3], strides=[2, 2], padding=[1, 1, 1, 1]);
          %7 = nn.batch_norm(%6, meta[relay.Constant][9], meta[relay.Constant][10], meta[relay.Constant][11], meta[relay.Constant][12]);
          %8 = %7.0;
          %9 = nn.relu(%8);
          %10 = nn.conv2d(%9, meta[relay.Constant][13], padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]);
          %11 = nn.batch_norm(%10, meta[relay.Constant][14], meta[relay.Constant][15], meta[relay.Constant][16], meta[relay.Constant][17]);
          %12 = %11.0;
          %13 = nn.relu(%12);
          %14 = nn.conv2d(%13, meta[relay.Constant][18], padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]);
          %15 = nn.batch_norm(%14, meta[relay.Constant][19], meta[relay.Constant][20], meta[relay.Constant][21], meta[relay.Constant][22]);
          %16 = %15.0;
          %17 = nn.relu(%16);
          %18 = nn.conv2d(%17, meta[relay.Constant][23], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %19 = nn.conv2d(%9, meta[relay.Constant][24], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %20 = add(%18, %19);
          %21 = nn.batch_norm(%20, meta[relay.Constant][25], meta[relay.Constant][26], meta[relay.Constant][27], meta[relay.Constant][28]);
          %22 = %21.0;
          %23 = nn.relu(%22);
          %24 = nn.conv2d(%23, meta[relay.Constant][29], padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]);
          %25 = nn.batch_norm(%24, meta[relay.Constant][30], meta[relay.Constant][31], meta[relay.Constant][32], meta[relay.Constant][33]);
          %26 = %25.0;
          %27 = nn.relu(%26);
          %28 = nn.conv2d(%27, meta[relay.Constant][34], padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]);
          %29 = nn.batch_norm(%28, meta[relay.Constant][35], meta[relay.Constant][36], meta[relay.Constant][37], meta[relay.Constant][38]);
          %30 = %29.0;
          %31 = nn.relu(%30);
          %32 = nn.conv2d(%31, meta[relay.Constant][39], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %33 = add(%32, %20);
          %34 = nn.batch_norm(%33, meta[relay.Constant][40], meta[relay.Constant][41], meta[relay.Constant][42], meta[relay.Constant][43]);
          %35 = %34.0;
          %36 = nn.relu(%35);
          %37 = nn.conv2d(%36, meta[relay.Constant][44], padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]);
          %38 = nn.batch_norm(%37, meta[relay.Constant][45], meta[relay.Constant][46], meta[relay.Constant][47], meta[relay.Constant][48]);
          %39 = %38.0;
          %40 = nn.relu(%39);
          %41 = nn.conv2d(%40, meta[relay.Constant][49], padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]);
          %42 = nn.batch_norm(%41, meta[relay.Constant][50], meta[relay.Constant][51], meta[relay.Constant][52], meta[relay.Constant][53]);
          %43 = %42.0;
          %44 = nn.relu(%43);
          %45 = nn.conv2d(%44, meta[relay.Constant][54], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %46 = add(%45, %33);
          %47 = nn.batch_norm(%46, meta[relay.Constant][55], meta[relay.Constant][56], meta[relay.Constant][57], meta[relay.Constant][58]);
          %48 = %47.0;
          %49 = nn.relu(%48);
          %50 = nn.conv2d(%49, meta[relay.Constant][59], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]);
          %51 = nn.batch_norm(%50, meta[relay.Constant][60], meta[relay.Constant][61], meta[relay.Constant][62], meta[relay.Constant][63]);
          %52 = %51.0;
          %53 = nn.relu(%52);
          %54 = nn.conv2d(%53, meta[relay.Constant][64], strides=[2, 2], padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]);
          %55 = nn.batch_norm(%54, meta[relay.Constant][65], meta[relay.Constant][66], meta[relay.Constant][67], meta[relay.Constant][68]);
          %56 = %55.0;
          %57 = nn.relu(%56);
          %58 = nn.conv2d(%57, meta[relay.Constant][69], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %59 = nn.conv2d(%49, meta[relay.Constant][70], strides=[2, 2], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %60 = add(%58, %59);
          %61 = nn.batch_norm(%60, meta[relay.Constant][71], meta[relay.Constant][72], meta[relay.Constant][73], meta[relay.Constant][74]);
          %62 = %61.0;
          %63 = nn.relu(%62);
          %64 = nn.conv2d(%63, meta[relay.Constant][75], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]);
          %65 = nn.batch_norm(%64, meta[relay.Constant][76], meta[relay.Constant][77], meta[relay.Constant][78], meta[relay.Constant][79]);
          %66 = %65.0;
          %67 = nn.relu(%66);
          %68 = nn.conv2d(%67, meta[relay.Constant][80], padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]);
          %69 = nn.batch_norm(%68, meta[relay.Constant][81], meta[relay.Constant][82], meta[relay.Constant][83], meta[relay.Constant][84]);
          %70 = %69.0;
          %71 = nn.relu(%70);
          %72 = nn.conv2d(%71, meta[relay.Constant][85], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %73 = add(%72, %60);
          %74 = nn.batch_norm(%73, meta[relay.Constant][86], meta[relay.Constant][87], meta[relay.Constant][88], meta[relay.Constant][89]);
          %75 = %74.0;
          %76 = nn.relu(%75);
          %77 = nn.conv2d(%76, meta[relay.Constant][90], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]);
          %78 = nn.batch_norm(%77, meta[relay.Constant][91], meta[relay.Constant][92], meta[relay.Constant][93], meta[relay.Constant][94]);
          %79 = %78.0;
          %80 = nn.relu(%79);
          %81 = nn.conv2d(%80, meta[relay.Constant][95], padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]);
          %82 = nn.batch_norm(%81, meta[relay.Constant][96], meta[relay.Constant][97], meta[relay.Constant][98], meta[relay.Constant][99]);
          %83 = %82.0;
          %84 = nn.relu(%83);
          %85 = nn.conv2d(%84, meta[relay.Constant][100], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %86 = add(%85, %73);
          %87 = nn.batch_norm(%86, meta[relay.Constant][101], meta[relay.Constant][102], meta[relay.Constant][103], meta[relay.Constant][104]);
          %88 = %87.0;
          %89 = nn.relu(%88);
          %90 = nn.conv2d(%89, meta[relay.Constant][105], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]);
          %91 = nn.batch_norm(%90, meta[relay.Constant][106], meta[relay.Constant][107], meta[relay.Constant][108], meta[relay.Constant][109]);
          %92 = %91.0;
          %93 = nn.relu(%92);
          %94 = nn.conv2d(%93, meta[relay.Constant][110], padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]);
          %95 = nn.batch_norm(%94, meta[relay.Constant][111], meta[relay.Constant][112], meta[relay.Constant][113], meta[relay.Constant][114]);
          %96 = %95.0;
          %97 = nn.relu(%96);
          %98 = nn.conv2d(%97, meta[relay.Constant][115], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %99 = add(%98, %86);
          %100 = nn.batch_norm(%99, meta[relay.Constant][116], meta[relay.Constant][117], meta[relay.Constant][118], meta[relay.Constant][119]);
          %101 = %100.0;
          %102 = nn.relu(%101);
          %103 = nn.conv2d(%102, meta[relay.Constant][120], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %104 = nn.batch_norm(%103, meta[relay.Constant][121], meta[relay.Constant][122], meta[relay.Constant][123], meta[relay.Constant][124]);
          %105 = %104.0;
          %106 = nn.relu(%105);
          %107 = nn.conv2d(%106, meta[relay.Constant][125], strides=[2, 2], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]);
          %108 = nn.batch_norm(%107, meta[relay.Constant][126], meta[relay.Constant][127], meta[relay.Constant][128], meta[relay.Constant][129]);
          %109 = %108.0;
          %110 = nn.relu(%109);
          %111 = nn.conv2d(%110, meta[relay.Constant][130], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %112 = nn.conv2d(%102, meta[relay.Constant][131], strides=[2, 2], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %113 = add(%111, %112);
          %114 = nn.batch_norm(%113, meta[relay.Constant][132], meta[relay.Constant][133], meta[relay.Constant][134], meta[relay.Constant][135]);
          %115 = %114.0;
          %116 = nn.relu(%115);
          %117 = nn.conv2d(%116, meta[relay.Constant][136], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %118 = nn.batch_norm(%117, meta[relay.Constant][137], meta[relay.Constant][138], meta[relay.Constant][139], meta[relay.Constant][140]);
          %119 = %118.0;
          %120 = nn.relu(%119);
          %121 = nn.conv2d(%120, meta[relay.Constant][141], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]);
          %122 = nn.batch_norm(%121, meta[relay.Constant][142], meta[relay.Constant][143], meta[relay.Constant][144], meta[relay.Constant][145]);
          %123 = %122.0;
          %124 = nn.relu(%123);
          %125 = nn.conv2d(%124, meta[relay.Constant][146], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %126 = add(%125, %113);
          %127 = nn.batch_norm(%126, meta[relay.Constant][147], meta[relay.Constant][148], meta[relay.Constant][149], meta[relay.Constant][150]);
          %128 = %127.0;
          %129 = nn.relu(%128);
          %130 = nn.conv2d(%129, meta[relay.Constant][151], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %131 = nn.batch_norm(%130, meta[relay.Constant][152], meta[relay.Constant][153], meta[relay.Constant][154], meta[relay.Constant][155]);
          %132 = %131.0;
          %133 = nn.relu(%132);
          %134 = nn.conv2d(%133, meta[relay.Constant][156], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]);
          %135 = nn.batch_norm(%134, meta[relay.Constant][157], meta[relay.Constant][158], meta[relay.Constant][159], meta[relay.Constant][160]);
          %136 = %135.0;
          %137 = nn.relu(%136);
          %138 = nn.conv2d(%137, meta[relay.Constant][161], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %139 = add(%138, %126);
          %140 = nn.batch_norm(%139, meta[relay.Constant][162], meta[relay.Constant][163], meta[relay.Constant][164], meta[relay.Constant][165]);
          %141 = %140.0;
          %142 = nn.relu(%141);
          %143 = nn.conv2d(%142, meta[relay.Constant][166], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %144 = nn.batch_norm(%143, meta[relay.Constant][167], meta[relay.Constant][168], meta[relay.Constant][169], meta[relay.Constant][170]);
          %145 = %144.0;
          %146 = nn.relu(%145);
          %147 = nn.conv2d(%146, meta[relay.Constant][171], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]);
          %148 = nn.batch_norm(%147, meta[relay.Constant][172], meta[relay.Constant][173], meta[relay.Constant][174], meta[relay.Constant][175]);
          %149 = %148.0;
          %150 = nn.relu(%149);
          %151 = nn.conv2d(%150, meta[relay.Constant][176], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %152 = add(%151, %139);
          %153 = nn.batch_norm(%152, meta[relay.Constant][177], meta[relay.Constant][178], meta[relay.Constant][179], meta[relay.Constant][180]);
          %154 = %153.0;
          %155 = nn.relu(%154);
          %156 = nn.conv2d(%155, meta[relay.Constant][181], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %157 = nn.batch_norm(%156, meta[relay.Constant][182], meta[relay.Constant][183], meta[relay.Constant][184], meta[relay.Constant][185]);
          %158 = %157.0;
          %159 = nn.relu(%158);
          %160 = nn.conv2d(%159, meta[relay.Constant][186], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]);
          %161 = nn.batch_norm(%160, meta[relay.Constant][187], meta[relay.Constant][188], meta[relay.Constant][189], meta[relay.Constant][190]);
          %162 = %161.0;
          %163 = nn.relu(%162);
          %164 = nn.conv2d(%163, meta[relay.Constant][191], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %165 = add(%164, %152);
          %166 = nn.batch_norm(%165, meta[relay.Constant][192], meta[relay.Constant][193], meta[relay.Constant][194], meta[relay.Constant][195]);
          %167 = %166.0;
          %168 = nn.relu(%167);
          %169 = nn.conv2d(%168, meta[relay.Constant][196], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %170 = nn.batch_norm(%169, meta[relay.Constant][197], meta[relay.Constant][198], meta[relay.Constant][199], meta[relay.Constant][200]);
          %171 = %170.0;
          %172 = nn.relu(%171);
          %173 = nn.conv2d(%172, meta[relay.Constant][201], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]);
          %174 = nn.batch_norm(%173, meta[relay.Constant][202], meta[relay.Constant][203], meta[relay.Constant][204], meta[relay.Constant][205]);
          %175 = %174.0;
          %176 = nn.relu(%175);
          %177 = nn.conv2d(%176, meta[relay.Constant][206], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %178 = add(%177, %165);
          %179 = nn.batch_norm(%178, meta[relay.Constant][207], meta[relay.Constant][208], meta[relay.Constant][209], meta[relay.Constant][210]);
          %180 = %179.0;
          %181 = nn.relu(%180);
          %182 = nn.conv2d(%181, meta[relay.Constant][211], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %183 = nn.batch_norm(%182, meta[relay.Constant][212], meta[relay.Constant][213], meta[relay.Constant][214], meta[relay.Constant][215]);
          %184 = %183.0;
          %185 = nn.relu(%184);
          %186 = nn.conv2d(%185, meta[relay.Constant][216], strides=[2, 2], padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]);
          %187 = nn.batch_norm(%186, meta[relay.Constant][217], meta[relay.Constant][218], meta[relay.Constant][219], meta[relay.Constant][220]);
          %188 = %187.0;
          %189 = nn.relu(%188);
          %190 = nn.conv2d(%189, meta[relay.Constant][221], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1]);
          %191 = nn.conv2d(%181, meta[relay.Constant][222], strides=[2, 2], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1]);
          %192 = add(%190, %191);
          %193 = nn.batch_norm(%192, meta[relay.Constant][223], meta[relay.Constant][224], meta[relay.Constant][225], meta[relay.Constant][226]);
          %194 = %193.0;
          %195 = nn.relu(%194);
          %196 = nn.conv2d(%195, meta[relay.Constant][227], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %197 = nn.batch_norm(%196, meta[relay.Constant][228], meta[relay.Constant][229], meta[relay.Constant][230], meta[relay.Constant][231]);
          %198 = %197.0;
          %199 = nn.relu(%198);
          %200 = nn.conv2d(%199, meta[relay.Constant][232], padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]);
          %201 = nn.batch_norm(%200, meta[relay.Constant][233], meta[relay.Constant][234], meta[relay.Constant][235], meta[relay.Constant][236]);
          %202 = %201.0;
          %203 = nn.relu(%202);
          %204 = nn.conv2d(%203, meta[relay.Constant][237], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1]);
          %205 = add(%204, %192);
          %206 = nn.batch_norm(%205, meta[relay.Constant][238], meta[relay.Constant][239], meta[relay.Constant][240], meta[relay.Constant][241]);
          %207 = %206.0;
          %208 = nn.relu(%207);
          %209 = nn.conv2d(%208, meta[relay.Constant][242], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %210 = nn.batch_norm(%209, meta[relay.Constant][243], meta[relay.Constant][244], meta[relay.Constant][245], meta[relay.Constant][246]);
          %211 = %210.0;
          %212 = nn.relu(%211);
          %213 = nn.conv2d(%212, meta[relay.Constant][247], padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]);
          %214 = nn.batch_norm(%213, meta[relay.Constant][248], meta[relay.Constant][249], meta[relay.Constant][250], meta[relay.Constant][251]);
          %215 = %214.0;
          %216 = nn.relu(%215);
          %217 = nn.conv2d(%216, meta[relay.Constant][252], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1]);
          %218 = add(%217, %205);
          %219 = nn.batch_norm(%218, meta[relay.Constant][253], meta[relay.Constant][254], meta[relay.Constant][255], meta[relay.Constant][256]);
          %220 = %219.0;
          %221 = nn.relu(%220);
          %222 = nn.global_avg_pool2d(%221);
          %223 = reshape(%222, newshape=[0, -1]);
          %224 = nn.dense(%223, meta[relay.Constant][257], units=1000);
          add(%224, meta[relay.Constant][258])
        }
        """,
        "from_string",
        None,
        metatable,
    )
    return {
        "name": "resnet50",
        "input_shapes": {"data": [1, 3, 224, 224]},
        "input_dtypes": {"data": "float32"},
        "mod": mod,
        "params": None,
        "main_dtype": "float32",
    }


def resnet50_16():
    metatable = {"relay.Constant": resnet50_consts("float16")}
    mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%data: Tensor[(1, 3, 224, 224), float16]) -> Tensor[(1, 1000), float16] {
          %0 = nn.batch_norm(%data, meta[relay.Constant][0], meta[relay.Constant][1], meta[relay.Constant][2], meta[relay.Constant][3]);
          %1 = %0.0;
          %2 = nn.conv2d(%1, meta[relay.Constant][4], strides=[2, 2], padding=[3, 3, 3, 3], channels=64, kernel_size=[7, 7]);
          %3 = nn.batch_norm(%2, meta[relay.Constant][5], meta[relay.Constant][6], meta[relay.Constant][7], meta[relay.Constant][8]);
          %4 = %3.0;
          %5 = nn.relu(%4);
          %6 = nn.max_pool2d(%5, pool_size=[3, 3], strides=[2, 2], padding=[1, 1, 1, 1]);
          %7 = nn.batch_norm(%6, meta[relay.Constant][9], meta[relay.Constant][10], meta[relay.Constant][11], meta[relay.Constant][12]);
          %8 = %7.0;
          %9 = nn.relu(%8);
          %10 = nn.conv2d(%9, meta[relay.Constant][13], padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]);
          %11 = nn.batch_norm(%10, meta[relay.Constant][14], meta[relay.Constant][15], meta[relay.Constant][16], meta[relay.Constant][17]);
          %12 = %11.0;
          %13 = nn.relu(%12);
          %14 = nn.conv2d(%13, meta[relay.Constant][18], padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]);
          %15 = nn.batch_norm(%14, meta[relay.Constant][19], meta[relay.Constant][20], meta[relay.Constant][21], meta[relay.Constant][22]);
          %16 = %15.0;
          %17 = nn.relu(%16);
          %18 = nn.conv2d(%17, meta[relay.Constant][23], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %19 = nn.conv2d(%9, meta[relay.Constant][24], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %20 = add(%18, %19);
          %21 = nn.batch_norm(%20, meta[relay.Constant][25], meta[relay.Constant][26], meta[relay.Constant][27], meta[relay.Constant][28]);
          %22 = %21.0;
          %23 = nn.relu(%22);
          %24 = nn.conv2d(%23, meta[relay.Constant][29], padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]);
          %25 = nn.batch_norm(%24, meta[relay.Constant][30], meta[relay.Constant][31], meta[relay.Constant][32], meta[relay.Constant][33]);
          %26 = %25.0;
          %27 = nn.relu(%26);
          %28 = nn.conv2d(%27, meta[relay.Constant][34], padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]);
          %29 = nn.batch_norm(%28, meta[relay.Constant][35], meta[relay.Constant][36], meta[relay.Constant][37], meta[relay.Constant][38]);
          %30 = %29.0;
          %31 = nn.relu(%30);
          %32 = nn.conv2d(%31, meta[relay.Constant][39], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %33 = add(%32, %20);
          %34 = nn.batch_norm(%33, meta[relay.Constant][40], meta[relay.Constant][41], meta[relay.Constant][42], meta[relay.Constant][43]);
          %35 = %34.0;
          %36 = nn.relu(%35);
          %37 = nn.conv2d(%36, meta[relay.Constant][44], padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]);
          %38 = nn.batch_norm(%37, meta[relay.Constant][45], meta[relay.Constant][46], meta[relay.Constant][47], meta[relay.Constant][48]);
          %39 = %38.0;
          %40 = nn.relu(%39);
          %41 = nn.conv2d(%40, meta[relay.Constant][49], padding=[1, 1, 1, 1], channels=64, kernel_size=[3, 3]);
          %42 = nn.batch_norm(%41, meta[relay.Constant][50], meta[relay.Constant][51], meta[relay.Constant][52], meta[relay.Constant][53]);
          %43 = %42.0;
          %44 = nn.relu(%43);
          %45 = nn.conv2d(%44, meta[relay.Constant][54], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %46 = add(%45, %33);
          %47 = nn.batch_norm(%46, meta[relay.Constant][55], meta[relay.Constant][56], meta[relay.Constant][57], meta[relay.Constant][58]);
          %48 = %47.0;
          %49 = nn.relu(%48);
          %50 = nn.conv2d(%49, meta[relay.Constant][59], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]);
          %51 = nn.batch_norm(%50, meta[relay.Constant][60], meta[relay.Constant][61], meta[relay.Constant][62], meta[relay.Constant][63]);
          %52 = %51.0;
          %53 = nn.relu(%52);
          %54 = nn.conv2d(%53, meta[relay.Constant][64], strides=[2, 2], padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]);
          %55 = nn.batch_norm(%54, meta[relay.Constant][65], meta[relay.Constant][66], meta[relay.Constant][67], meta[relay.Constant][68]);
          %56 = %55.0;
          %57 = nn.relu(%56);
          %58 = nn.conv2d(%57, meta[relay.Constant][69], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %59 = nn.conv2d(%49, meta[relay.Constant][70], strides=[2, 2], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %60 = add(%58, %59);
          %61 = nn.batch_norm(%60, meta[relay.Constant][71], meta[relay.Constant][72], meta[relay.Constant][73], meta[relay.Constant][74]);
          %62 = %61.0;
          %63 = nn.relu(%62);
          %64 = nn.conv2d(%63, meta[relay.Constant][75], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]);
          %65 = nn.batch_norm(%64, meta[relay.Constant][76], meta[relay.Constant][77], meta[relay.Constant][78], meta[relay.Constant][79]);
          %66 = %65.0;
          %67 = nn.relu(%66);
          %68 = nn.conv2d(%67, meta[relay.Constant][80], padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]);
          %69 = nn.batch_norm(%68, meta[relay.Constant][81], meta[relay.Constant][82], meta[relay.Constant][83], meta[relay.Constant][84]);
          %70 = %69.0;
          %71 = nn.relu(%70);
          %72 = nn.conv2d(%71, meta[relay.Constant][85], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %73 = add(%72, %60);
          %74 = nn.batch_norm(%73, meta[relay.Constant][86], meta[relay.Constant][87], meta[relay.Constant][88], meta[relay.Constant][89]);
          %75 = %74.0;
          %76 = nn.relu(%75);
          %77 = nn.conv2d(%76, meta[relay.Constant][90], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]);
          %78 = nn.batch_norm(%77, meta[relay.Constant][91], meta[relay.Constant][92], meta[relay.Constant][93], meta[relay.Constant][94]);
          %79 = %78.0;
          %80 = nn.relu(%79);
          %81 = nn.conv2d(%80, meta[relay.Constant][95], padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]);
          %82 = nn.batch_norm(%81, meta[relay.Constant][96], meta[relay.Constant][97], meta[relay.Constant][98], meta[relay.Constant][99]);
          %83 = %82.0;
          %84 = nn.relu(%83);
          %85 = nn.conv2d(%84, meta[relay.Constant][100], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %86 = add(%85, %73);
          %87 = nn.batch_norm(%86, meta[relay.Constant][101], meta[relay.Constant][102], meta[relay.Constant][103], meta[relay.Constant][104]);
          %88 = %87.0;
          %89 = nn.relu(%88);
          %90 = nn.conv2d(%89, meta[relay.Constant][105], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]);
          %91 = nn.batch_norm(%90, meta[relay.Constant][106], meta[relay.Constant][107], meta[relay.Constant][108], meta[relay.Constant][109]);
          %92 = %91.0;
          %93 = nn.relu(%92);
          %94 = nn.conv2d(%93, meta[relay.Constant][110], padding=[1, 1, 1, 1], channels=128, kernel_size=[3, 3]);
          %95 = nn.batch_norm(%94, meta[relay.Constant][111], meta[relay.Constant][112], meta[relay.Constant][113], meta[relay.Constant][114]);
          %96 = %95.0;
          %97 = nn.relu(%96);
          %98 = nn.conv2d(%97, meta[relay.Constant][115], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %99 = add(%98, %86);
          %100 = nn.batch_norm(%99, meta[relay.Constant][116], meta[relay.Constant][117], meta[relay.Constant][118], meta[relay.Constant][119]);
          %101 = %100.0;
          %102 = nn.relu(%101);
          %103 = nn.conv2d(%102, meta[relay.Constant][120], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %104 = nn.batch_norm(%103, meta[relay.Constant][121], meta[relay.Constant][122], meta[relay.Constant][123], meta[relay.Constant][124]);
          %105 = %104.0;
          %106 = nn.relu(%105);
          %107 = nn.conv2d(%106, meta[relay.Constant][125], strides=[2, 2], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]);
          %108 = nn.batch_norm(%107, meta[relay.Constant][126], meta[relay.Constant][127], meta[relay.Constant][128], meta[relay.Constant][129]);
          %109 = %108.0;
          %110 = nn.relu(%109);
          %111 = nn.conv2d(%110, meta[relay.Constant][130], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %112 = nn.conv2d(%102, meta[relay.Constant][131], strides=[2, 2], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %113 = add(%111, %112);
          %114 = nn.batch_norm(%113, meta[relay.Constant][132], meta[relay.Constant][133], meta[relay.Constant][134], meta[relay.Constant][135]);
          %115 = %114.0;
          %116 = nn.relu(%115);
          %117 = nn.conv2d(%116, meta[relay.Constant][136], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %118 = nn.batch_norm(%117, meta[relay.Constant][137], meta[relay.Constant][138], meta[relay.Constant][139], meta[relay.Constant][140]);
          %119 = %118.0;
          %120 = nn.relu(%119);
          %121 = nn.conv2d(%120, meta[relay.Constant][141], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]);
          %122 = nn.batch_norm(%121, meta[relay.Constant][142], meta[relay.Constant][143], meta[relay.Constant][144], meta[relay.Constant][145]);
          %123 = %122.0;
          %124 = nn.relu(%123);
          %125 = nn.conv2d(%124, meta[relay.Constant][146], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %126 = add(%125, %113);
          %127 = nn.batch_norm(%126, meta[relay.Constant][147], meta[relay.Constant][148], meta[relay.Constant][149], meta[relay.Constant][150]);
          %128 = %127.0;
          %129 = nn.relu(%128);
          %130 = nn.conv2d(%129, meta[relay.Constant][151], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %131 = nn.batch_norm(%130, meta[relay.Constant][152], meta[relay.Constant][153], meta[relay.Constant][154], meta[relay.Constant][155]);
          %132 = %131.0;
          %133 = nn.relu(%132);
          %134 = nn.conv2d(%133, meta[relay.Constant][156], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]);
          %135 = nn.batch_norm(%134, meta[relay.Constant][157], meta[relay.Constant][158], meta[relay.Constant][159], meta[relay.Constant][160]);
          %136 = %135.0;
          %137 = nn.relu(%136);
          %138 = nn.conv2d(%137, meta[relay.Constant][161], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %139 = add(%138, %126);
          %140 = nn.batch_norm(%139, meta[relay.Constant][162], meta[relay.Constant][163], meta[relay.Constant][164], meta[relay.Constant][165]);
          %141 = %140.0;
          %142 = nn.relu(%141);
          %143 = nn.conv2d(%142, meta[relay.Constant][166], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %144 = nn.batch_norm(%143, meta[relay.Constant][167], meta[relay.Constant][168], meta[relay.Constant][169], meta[relay.Constant][170]);
          %145 = %144.0;
          %146 = nn.relu(%145);
          %147 = nn.conv2d(%146, meta[relay.Constant][171], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]);
          %148 = nn.batch_norm(%147, meta[relay.Constant][172], meta[relay.Constant][173], meta[relay.Constant][174], meta[relay.Constant][175]);
          %149 = %148.0;
          %150 = nn.relu(%149);
          %151 = nn.conv2d(%150, meta[relay.Constant][176], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %152 = add(%151, %139);
          %153 = nn.batch_norm(%152, meta[relay.Constant][177], meta[relay.Constant][178], meta[relay.Constant][179], meta[relay.Constant][180]);
          %154 = %153.0;
          %155 = nn.relu(%154);
          %156 = nn.conv2d(%155, meta[relay.Constant][181], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %157 = nn.batch_norm(%156, meta[relay.Constant][182], meta[relay.Constant][183], meta[relay.Constant][184], meta[relay.Constant][185]);
          %158 = %157.0;
          %159 = nn.relu(%158);
          %160 = nn.conv2d(%159, meta[relay.Constant][186], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]);
          %161 = nn.batch_norm(%160, meta[relay.Constant][187], meta[relay.Constant][188], meta[relay.Constant][189], meta[relay.Constant][190]);
          %162 = %161.0;
          %163 = nn.relu(%162);
          %164 = nn.conv2d(%163, meta[relay.Constant][191], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %165 = add(%164, %152);
          %166 = nn.batch_norm(%165, meta[relay.Constant][192], meta[relay.Constant][193], meta[relay.Constant][194], meta[relay.Constant][195]);
          %167 = %166.0;
          %168 = nn.relu(%167);
          %169 = nn.conv2d(%168, meta[relay.Constant][196], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %170 = nn.batch_norm(%169, meta[relay.Constant][197], meta[relay.Constant][198], meta[relay.Constant][199], meta[relay.Constant][200]);
          %171 = %170.0;
          %172 = nn.relu(%171);
          %173 = nn.conv2d(%172, meta[relay.Constant][201], padding=[1, 1, 1, 1], channels=256, kernel_size=[3, 3]);
          %174 = nn.batch_norm(%173, meta[relay.Constant][202], meta[relay.Constant][203], meta[relay.Constant][204], meta[relay.Constant][205]);
          %175 = %174.0;
          %176 = nn.relu(%175);
          %177 = nn.conv2d(%176, meta[relay.Constant][206], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %178 = add(%177, %165);
          %179 = nn.batch_norm(%178, meta[relay.Constant][207], meta[relay.Constant][208], meta[relay.Constant][209], meta[relay.Constant][210]);
          %180 = %179.0;
          %181 = nn.relu(%180);
          %182 = nn.conv2d(%181, meta[relay.Constant][211], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %183 = nn.batch_norm(%182, meta[relay.Constant][212], meta[relay.Constant][213], meta[relay.Constant][214], meta[relay.Constant][215]);
          %184 = %183.0;
          %185 = nn.relu(%184);
          %186 = nn.conv2d(%185, meta[relay.Constant][216], strides=[2, 2], padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]);
          %187 = nn.batch_norm(%186, meta[relay.Constant][217], meta[relay.Constant][218], meta[relay.Constant][219], meta[relay.Constant][220]);
          %188 = %187.0;
          %189 = nn.relu(%188);
          %190 = nn.conv2d(%189, meta[relay.Constant][221], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1]);
          %191 = nn.conv2d(%181, meta[relay.Constant][222], strides=[2, 2], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1]);
          %192 = add(%190, %191);
          %193 = nn.batch_norm(%192, meta[relay.Constant][223], meta[relay.Constant][224], meta[relay.Constant][225], meta[relay.Constant][226]);
          %194 = %193.0;
          %195 = nn.relu(%194);
          %196 = nn.conv2d(%195, meta[relay.Constant][227], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %197 = nn.batch_norm(%196, meta[relay.Constant][228], meta[relay.Constant][229], meta[relay.Constant][230], meta[relay.Constant][231]);
          %198 = %197.0;
          %199 = nn.relu(%198);
          %200 = nn.conv2d(%199, meta[relay.Constant][232], padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]);
          %201 = nn.batch_norm(%200, meta[relay.Constant][233], meta[relay.Constant][234], meta[relay.Constant][235], meta[relay.Constant][236]);
          %202 = %201.0;
          %203 = nn.relu(%202);
          %204 = nn.conv2d(%203, meta[relay.Constant][237], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1]);
          %205 = add(%204, %192);
          %206 = nn.batch_norm(%205, meta[relay.Constant][238], meta[relay.Constant][239], meta[relay.Constant][240], meta[relay.Constant][241]);
          %207 = %206.0;
          %208 = nn.relu(%207);
          %209 = nn.conv2d(%208, meta[relay.Constant][242], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %210 = nn.batch_norm(%209, meta[relay.Constant][243], meta[relay.Constant][244], meta[relay.Constant][245], meta[relay.Constant][246]);
          %211 = %210.0;
          %212 = nn.relu(%211);
          %213 = nn.conv2d(%212, meta[relay.Constant][247], padding=[1, 1, 1, 1], channels=512, kernel_size=[3, 3]);
          %214 = nn.batch_norm(%213, meta[relay.Constant][248], meta[relay.Constant][249], meta[relay.Constant][250], meta[relay.Constant][251]);
          %215 = %214.0;
          %216 = nn.relu(%215);
          %217 = nn.conv2d(%216, meta[relay.Constant][252], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1]);
          %218 = add(%217, %205);
          %219 = nn.batch_norm(%218, meta[relay.Constant][253], meta[relay.Constant][254], meta[relay.Constant][255], meta[relay.Constant][256]);
          %220 = %219.0;
          %221 = nn.relu(%220);
          %222 = nn.global_avg_pool2d(%221);
          %223 = reshape(%222, newshape=[0, -1]);
          %224 = nn.dense(%223, meta[relay.Constant][257], units=1000);
          add(%224, meta[relay.Constant][258])
        }
        """,
        "from_string",
        None,
        metatable,
    )
    return {
        "name": "resnet50_16",
        "input_shapes": {"data": [1, 3, 224, 224]},
        "input_dtypes": {"data": "float16"},
        "mod": mod,
        "params": None,
        "main_dtype": "float16",
    }


def mobilenet_consts(dtype):
    return make_consts(
        dtype,
        [
            (32, 3, 3, 3),  # 0
            (32,),  # 1
            (32,),  # 2
            (32,),  # 3
            (32,),  # 4
            (32, 32, 1, 1),  # 5
            (32,),  # 6
            (32,),  # 7
            (32,),  # 8
            (32,),  # 9
            (32, 1, 3, 3),  # 10
            (32,),  # 11
            (32,),  # 12
            (32,),  # 13
            (32,),  # 14
            (16, 32, 1, 1),  # 15
            (16,),  # 16
            (16,),  # 17
            (16,),  # 18
            (16,),  # 19
            (96, 16, 1, 1),  # 20
            (96,),  # 21
            (96,),  # 22
            (96,),  # 23
            (96,),  # 24
            (96, 1, 3, 3),  # 25
            (96,),  # 26
            (96,),  # 27
            (96,),  # 28
            (96,),  # 29
            (24, 96, 1, 1),  # 30
            (24,),  # 31
            (24,),  # 32
            (24,),  # 33
            (24,),  # 34
            (144, 24, 1, 1),  # 35
            (144,),  # 36
            (144,),  # 37
            (144,),  # 38
            (144,),  # 39
            (144, 1, 3, 3),  # 40
            (144,),  # 41
            (144,),  # 42
            (144,),  # 43
            (144,),  # 44
            (24, 144, 1, 1),  # 45
            (24,),  # 46
            (24,),  # 47
            (24,),  # 48
            (24,),  # 49
            (144, 24, 1, 1),  # 50
            (144,),  # 51
            (144,),  # 52
            (144,),  # 53
            (144,),  # 54
            (144, 1, 3, 3),  # 55
            (144,),  # 56
            (144,),  # 57
            (144,),  # 58
            (144,),  # 59
            (32, 144, 1, 1),  # 60
            (32,),  # 61
            (32,),  # 62
            (32,),  # 63
            (32,),  # 64
            (192, 32, 1, 1),  # 65
            (192,),  # 66
            (192,),  # 67
            (192,),  # 68
            (192,),  # 69
            (192, 1, 3, 3),  # 70
            (192,),  # 71
            (192,),  # 72
            (192,),  # 73
            (192,),  # 74
            (32, 192, 1, 1),  # 75
            (32,),  # 76
            (32,),  # 77
            (32,),  # 78
            (32,),  # 79
            (192, 32, 1, 1),  # 80
            (192,),  # 81
            (192,),  # 82
            (192,),  # 83
            (192,),  # 84
            (192, 1, 3, 3),  # 85
            (192,),  # 86
            (192,),  # 87
            (192,),  # 88
            (192,),  # 89
            (32, 192, 1, 1),  # 90
            (32,),  # 91
            (32,),  # 92
            (32,),  # 93
            (32,),  # 94
            (192, 32, 1, 1),  # 95
            (192,),  # 96
            (192,),  # 97
            (192,),  # 98
            (192,),  # 99
            (192, 1, 3, 3),  # 100
            (192,),  # 101
            (192,),  # 102
            (192,),  # 103
            (192,),  # 104
            (64, 192, 1, 1),  # 105
            (64,),  # 106
            (64,),  # 107
            (64,),  # 108
            (64,),  # 109
            (384, 64, 1, 1),  # 110
            (384,),  # 111
            (384,),  # 112
            (384,),  # 113
            (384,),  # 114
            (384, 1, 3, 3),  # 115
            (384,),  # 116
            (384,),  # 117
            (384,),  # 118
            (384,),  # 119
            (64, 384, 1, 1),  # 120
            (64,),  # 121
            (64,),  # 122
            (64,),  # 123
            (64,),  # 124
            (384, 64, 1, 1),  # 125
            (384,),  # 126
            (384,),  # 127
            (384,),  # 128
            (384,),  # 129
            (384, 1, 3, 3),  # 130
            (384,),  # 131
            (384,),  # 132
            (384,),  # 133
            (384,),  # 134
            (64, 384, 1, 1),  # 135
            (64,),  # 136
            (64,),  # 137
            (64,),  # 138
            (64,),  # 139
            (384, 64, 1, 1),  # 140
            (384,),  # 141
            (384,),  # 142
            (384,),  # 143
            (384,),  # 144
            (384, 1, 3, 3),  # 145
            (384,),  # 146
            (384,),  # 147
            (384,),  # 148
            (384,),  # 149
            (64, 384, 1, 1),  # 150
            (64,),  # 151
            (64,),  # 152
            (64,),  # 153
            (64,),  # 154
            (384, 64, 1, 1),  # 155
            (384,),  # 156
            (384,),  # 157
            (384,),  # 158
            (384,),  # 159
            (384, 1, 3, 3),  # 160
            (384,),  # 161
            (384,),  # 162
            (384,),  # 163
            (384,),  # 164
            (96, 384, 1, 1),  # 165
            (96,),  # 166
            (96,),  # 167
            (96,),  # 168
            (96,),  # 169
            (576, 96, 1, 1),  # 170
            (576,),  # 171
            (576,),  # 172
            (576,),  # 173
            (576,),  # 174
            (576, 1, 3, 3),  # 175
            (576,),  # 176
            (576,),  # 177
            (576,),  # 178
            (576,),  # 179
            (96, 576, 1, 1),  # 180
            (96,),  # 181
            (96,),  # 182
            (96,),  # 183
            (96,),  # 184
            (576, 96, 1, 1),  # 185
            (576,),  # 186
            (576,),  # 187
            (576,),  # 188
            (576,),  # 189
            (576, 1, 3, 3),  # 190
            (576,),  # 191
            (576,),  # 192
            (576,),  # 193
            (576,),  # 194
            (96, 576, 1, 1),  # 195
            (96,),  # 196
            (96,),  # 197
            (96,),  # 198
            (96,),  # 199
            (576, 96, 1, 1),  # 200
            (576,),  # 201
            (576,),  # 202
            (576,),  # 203
            (576,),  # 204
            (576, 1, 3, 3),  # 205
            (576,),  # 206
            (576,),  # 207
            (576,),  # 208
            (576,),  # 209
            (160, 576, 1, 1),  # 210
            (160,),  # 211
            (160,),  # 212
            (160,),  # 213
            (160,),  # 214
            (960, 160, 1, 1),  # 215
            (960,),  # 216
            (960,),  # 217
            (960,),  # 218
            (960,),  # 219
            (960, 1, 3, 3),  # 220
            (960,),  # 221
            (960,),  # 222
            (960,),  # 223
            (960,),  # 224
            (160, 960, 1, 1),  # 225
            (160,),  # 226
            (160,),  # 227
            (160,),  # 228
            (160,),  # 229
            (960, 160, 1, 1),  # 230
            (960,),  # 231
            (960,),  # 232
            (960,),  # 233
            (960,),  # 234
            (960, 1, 3, 3),  # 235
            (960,),  # 236
            (960,),  # 237
            (960,),  # 238
            (960,),  # 239
            (160, 960, 1, 1),  # 240
            (160,),  # 241
            (160,),  # 242
            (160,),  # 243
            (160,),  # 244
            (960, 160, 1, 1),  # 245
            (960,),  # 246
            (960,),  # 247
            (960,),  # 248
            (960,),  # 249
            (960, 1, 3, 3),  # 250
            (960,),  # 251
            (960,),  # 252
            (960,),  # 253
            (960,),  # 254
            (320, 960, 1, 1),  # 255
            (320,),  # 256
            (320,),  # 257
            (320,),  # 258
            (320,),  # 259
            (1280, 320, 1, 1),  # 260
            (1280,),  # 261
            (1280,),  # 262
            (1280,),  # 263
            (1280,),  # 264
            (1000, 1280, 1, 1),  # 265
        ],
    )


def mobilenet():
    metatable = {"relay.Constant": mobilenet_consts("float32")}
    mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%data: Tensor[(1, 3, 224, 224), float32]) -> Tensor[(1, 1000), float32] {
          %0 = nn.conv2d(%data, meta[relay.Constant][0], strides=[2, 2], padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3]);
          %1 = nn.batch_norm(%0, meta[relay.Constant][1], meta[relay.Constant][2], meta[relay.Constant][3], meta[relay.Constant][4]);
          %2 = %1.0;
          %3 = nn.relu(%2);
          %4 = nn.conv2d(%3, meta[relay.Constant][5], padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1]);
          %5 = nn.batch_norm(%4, meta[relay.Constant][6], meta[relay.Constant][7], meta[relay.Constant][8], meta[relay.Constant][9]);
          %6 = %5.0;
          %7 = nn.relu(%6);
          %8 = nn.conv2d(%7, meta[relay.Constant][10], padding=[1, 1, 1, 1], groups=32, channels=32, kernel_size=[3, 3]);
          %9 = nn.batch_norm(%8, meta[relay.Constant][11], meta[relay.Constant][12], meta[relay.Constant][13], meta[relay.Constant][14]);
          %10 = %9.0;
          %11 = nn.relu(%10);
          %12 = nn.conv2d(%11, meta[relay.Constant][15], padding=[0, 0, 0, 0], channels=16, kernel_size=[1, 1]);
          %13 = nn.batch_norm(%12, meta[relay.Constant][16], meta[relay.Constant][17], meta[relay.Constant][18], meta[relay.Constant][19]);
          %14 = %13.0;
          %15 = nn.conv2d(%14, meta[relay.Constant][20], padding=[0, 0, 0, 0], channels=96, kernel_size=[1, 1]);
          %16 = nn.batch_norm(%15, meta[relay.Constant][21], meta[relay.Constant][22], meta[relay.Constant][23], meta[relay.Constant][24]);
          %17 = %16.0;
          %18 = nn.relu(%17);
          %19 = nn.conv2d(%18, meta[relay.Constant][25], strides=[2, 2], padding=[1, 1, 1, 1], groups=96, channels=96, kernel_size=[3, 3]);
          %20 = nn.batch_norm(%19, meta[relay.Constant][26], meta[relay.Constant][27], meta[relay.Constant][28], meta[relay.Constant][29]);
          %21 = %20.0;
          %22 = nn.relu(%21);
          %23 = nn.conv2d(%22, meta[relay.Constant][30], padding=[0, 0, 0, 0], channels=24, kernel_size=[1, 1]);
          %24 = nn.batch_norm(%23, meta[relay.Constant][31], meta[relay.Constant][32], meta[relay.Constant][33], meta[relay.Constant][34]);
          %25 = %24.0;
          %26 = nn.conv2d(%25, meta[relay.Constant][35], padding=[0, 0, 0, 0], channels=144, kernel_size=[1, 1]);
          %27 = nn.batch_norm(%26, meta[relay.Constant][36], meta[relay.Constant][37], meta[relay.Constant][38], meta[relay.Constant][39]);
          %28 = %27.0;
          %29 = nn.relu(%28);
          %30 = nn.conv2d(%29, meta[relay.Constant][40], padding=[1, 1, 1, 1], groups=144, channels=144, kernel_size=[3, 3]);
          %31 = nn.batch_norm(%30, meta[relay.Constant][41], meta[relay.Constant][42], meta[relay.Constant][43], meta[relay.Constant][44]);
          %32 = %31.0;
          %33 = nn.relu(%32);
          %34 = nn.conv2d(%33, meta[relay.Constant][45], padding=[0, 0, 0, 0], channels=24, kernel_size=[1, 1]);
          %35 = nn.batch_norm(%34, meta[relay.Constant][46], meta[relay.Constant][47], meta[relay.Constant][48], meta[relay.Constant][49]);
          %36 = %35.0;
          %37 = add(%36, %25);
          %38 = nn.conv2d(%37, meta[relay.Constant][50], padding=[0, 0, 0, 0], channels=144, kernel_size=[1, 1]);
          %39 = nn.batch_norm(%38, meta[relay.Constant][51], meta[relay.Constant][52], meta[relay.Constant][53], meta[relay.Constant][54]);
          %40 = %39.0;
          %41 = nn.relu(%40);
          %42 = nn.conv2d(%41, meta[relay.Constant][55], strides=[2, 2], padding=[1, 1, 1, 1], groups=144, channels=144, kernel_size=[3, 3]);
          %43 = nn.batch_norm(%42, meta[relay.Constant][56], meta[relay.Constant][57], meta[relay.Constant][58], meta[relay.Constant][59]);
          %44 = %43.0;
          %45 = nn.relu(%44);
          %46 = nn.conv2d(%45, meta[relay.Constant][60], padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1]);
          %47 = nn.batch_norm(%46, meta[relay.Constant][61], meta[relay.Constant][62], meta[relay.Constant][63], meta[relay.Constant][64]);
          %48 = %47.0;
          %49 = nn.conv2d(%48, meta[relay.Constant][65], padding=[0, 0, 0, 0], channels=192, kernel_size=[1, 1]);
          %50 = nn.batch_norm(%49, meta[relay.Constant][66], meta[relay.Constant][67], meta[relay.Constant][68], meta[relay.Constant][69]);
          %51 = %50.0;
          %52 = nn.relu(%51);
          %53 = nn.conv2d(%52, meta[relay.Constant][70], padding=[1, 1, 1, 1], groups=192, channels=192, kernel_size=[3, 3]);
          %54 = nn.batch_norm(%53, meta[relay.Constant][71], meta[relay.Constant][72], meta[relay.Constant][73], meta[relay.Constant][74]);
          %55 = %54.0;
          %56 = nn.relu(%55);
          %57 = nn.conv2d(%56, meta[relay.Constant][75], padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1]);
          %58 = nn.batch_norm(%57, meta[relay.Constant][76], meta[relay.Constant][77], meta[relay.Constant][78], meta[relay.Constant][79]);
          %59 = %58.0;
          %60 = add(%59, %48);
          %61 = nn.conv2d(%60, meta[relay.Constant][80], padding=[0, 0, 0, 0], channels=192, kernel_size=[1, 1]);
          %62 = nn.batch_norm(%61, meta[relay.Constant][81], meta[relay.Constant][82], meta[relay.Constant][83], meta[relay.Constant][84]);
          %63 = %62.0;
          %64 = nn.relu(%63);
          %65 = nn.conv2d(%64, meta[relay.Constant][85], padding=[1, 1, 1, 1], groups=192, channels=192, kernel_size=[3, 3]);
          %66 = nn.batch_norm(%65, meta[relay.Constant][86], meta[relay.Constant][87], meta[relay.Constant][88], meta[relay.Constant][89]);
          %67 = %66.0;
          %68 = nn.relu(%67);
          %69 = nn.conv2d(%68, meta[relay.Constant][90], padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1]);
          %70 = nn.batch_norm(%69, meta[relay.Constant][91], meta[relay.Constant][92], meta[relay.Constant][93], meta[relay.Constant][94]);
          %71 = %70.0;
          %72 = add(%71, %60);
          %73 = nn.conv2d(%72, meta[relay.Constant][95], padding=[0, 0, 0, 0], channels=192, kernel_size=[1, 1]);
          %74 = nn.batch_norm(%73, meta[relay.Constant][96], meta[relay.Constant][97], meta[relay.Constant][98], meta[relay.Constant][99]);
          %75 = %74.0;
          %76 = nn.relu(%75);
          %77 = nn.conv2d(%76, meta[relay.Constant][100], padding=[1, 1, 1, 1], groups=192, channels=192, kernel_size=[3, 3]);
          %78 = nn.batch_norm(%77, meta[relay.Constant][101], meta[relay.Constant][102], meta[relay.Constant][103], meta[relay.Constant][104]);
          %79 = %78.0;
          %80 = nn.relu(%79);
          %81 = nn.conv2d(%80, meta[relay.Constant][105], padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]);
          %82 = nn.batch_norm(%81, meta[relay.Constant][106], meta[relay.Constant][107], meta[relay.Constant][108], meta[relay.Constant][109]);
          %83 = %82.0;
          %84 = nn.conv2d(%83, meta[relay.Constant][110], padding=[0, 0, 0, 0], channels=384, kernel_size=[1, 1]);
          %85 = nn.batch_norm(%84, meta[relay.Constant][111], meta[relay.Constant][112], meta[relay.Constant][113], meta[relay.Constant][114]);
          %86 = %85.0;
          %87 = nn.relu(%86);
          %88 = nn.conv2d(%87, meta[relay.Constant][115], padding=[1, 1, 1, 1], groups=384, channels=384, kernel_size=[3, 3]);
          %89 = nn.batch_norm(%88, meta[relay.Constant][116], meta[relay.Constant][117], meta[relay.Constant][118], meta[relay.Constant][119]);
          %90 = %89.0;
          %91 = nn.relu(%90);
          %92 = nn.conv2d(%91, meta[relay.Constant][120], padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]);
          %93 = nn.batch_norm(%92, meta[relay.Constant][121], meta[relay.Constant][122], meta[relay.Constant][123], meta[relay.Constant][124]);
          %94 = %93.0;
          %95 = add(%94, %83);
          %96 = nn.conv2d(%95, meta[relay.Constant][125], padding=[0, 0, 0, 0], channels=384, kernel_size=[1, 1]);
          %97 = nn.batch_norm(%96, meta[relay.Constant][126], meta[relay.Constant][127], meta[relay.Constant][128], meta[relay.Constant][129]);
          %98 = %97.0;
          %99 = nn.relu(%98);
          %100 = nn.conv2d(%99, meta[relay.Constant][130], padding=[1, 1, 1, 1], groups=384, channels=384, kernel_size=[3, 3]);
          %101 = nn.batch_norm(%100, meta[relay.Constant][131], meta[relay.Constant][132], meta[relay.Constant][133], meta[relay.Constant][134]);
          %102 = %101.0;
          %103 = nn.relu(%102);
          %104 = nn.conv2d(%103, meta[relay.Constant][135], padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]);
          %105 = nn.batch_norm(%104, meta[relay.Constant][136], meta[relay.Constant][137], meta[relay.Constant][138], meta[relay.Constant][139]);
          %106 = %105.0;
          %107 = add(%106, %95);
          %108 = nn.conv2d(%107, meta[relay.Constant][140], padding=[0, 0, 0, 0], channels=384, kernel_size=[1, 1]);
          %109 = nn.batch_norm(%108, meta[relay.Constant][141], meta[relay.Constant][142], meta[relay.Constant][143], meta[relay.Constant][144]);
          %110 = %109.0;
          %111 = nn.relu(%110);
          %112 = nn.conv2d(%111, meta[relay.Constant][145], padding=[1, 1, 1, 1], groups=384, channels=384, kernel_size=[3, 3]);
          %113 = nn.batch_norm(%112, meta[relay.Constant][146], meta[relay.Constant][147], meta[relay.Constant][148], meta[relay.Constant][149]);
          %114 = %113.0;
          %115 = nn.relu(%114);
          %116 = nn.conv2d(%115, meta[relay.Constant][150], padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]);
          %117 = nn.batch_norm(%116, meta[relay.Constant][151], meta[relay.Constant][152], meta[relay.Constant][153], meta[relay.Constant][154]);
          %118 = %117.0;
          %119 = add(%118, %107);
          %120 = nn.conv2d(%119, meta[relay.Constant][155], padding=[0, 0, 0, 0], channels=384, kernel_size=[1, 1]);
          %121 = nn.batch_norm(%120, meta[relay.Constant][156], meta[relay.Constant][157], meta[relay.Constant][158], meta[relay.Constant][159]);
          %122 = %121.0;
          %123 = nn.relu(%122);
          %124 = nn.conv2d(%123, meta[relay.Constant][160], strides=[2, 2], padding=[1, 1, 1, 1], groups=384, channels=384, kernel_size=[3, 3]);
          %125 = nn.batch_norm(%124, meta[relay.Constant][161], meta[relay.Constant][162], meta[relay.Constant][163], meta[relay.Constant][164]);
          %126 = %125.0;
          %127 = nn.relu(%126);
          %128 = nn.conv2d(%127, meta[relay.Constant][165], padding=[0, 0, 0, 0], channels=96, kernel_size=[1, 1]);
          %129 = nn.batch_norm(%128, meta[relay.Constant][166], meta[relay.Constant][167], meta[relay.Constant][168], meta[relay.Constant][169]);
          %130 = %129.0;
          %131 = nn.conv2d(%130, meta[relay.Constant][170], padding=[0, 0, 0, 0], channels=576, kernel_size=[1, 1]);
          %132 = nn.batch_norm(%131, meta[relay.Constant][171], meta[relay.Constant][172], meta[relay.Constant][173], meta[relay.Constant][174]);
          %133 = %132.0;
          %134 = nn.relu(%133);
          %135 = nn.conv2d(%134, meta[relay.Constant][175], padding=[1, 1, 1, 1], groups=576, channels=576, kernel_size=[3, 3]);
          %136 = nn.batch_norm(%135, meta[relay.Constant][176], meta[relay.Constant][177], meta[relay.Constant][178], meta[relay.Constant][179]);
          %137 = %136.0;
          %138 = nn.relu(%137);
          %139 = nn.conv2d(%138, meta[relay.Constant][180], padding=[0, 0, 0, 0], channels=96, kernel_size=[1, 1]);
          %140 = nn.batch_norm(%139, meta[relay.Constant][181], meta[relay.Constant][182], meta[relay.Constant][183], meta[relay.Constant][184]);
          %141 = %140.0;
          %142 = add(%141, %130);
          %143 = nn.conv2d(%142, meta[relay.Constant][185], padding=[0, 0, 0, 0], channels=576, kernel_size=[1, 1]);
          %144 = nn.batch_norm(%143, meta[relay.Constant][186], meta[relay.Constant][187], meta[relay.Constant][188], meta[relay.Constant][189]);
          %145 = %144.0;
          %146 = nn.relu(%145);
          %147 = nn.conv2d(%146, meta[relay.Constant][190], padding=[1, 1, 1, 1], groups=576, channels=576, kernel_size=[3, 3]);
          %148 = nn.batch_norm(%147, meta[relay.Constant][191], meta[relay.Constant][192], meta[relay.Constant][193], meta[relay.Constant][194]);
          %149 = %148.0;
          %150 = nn.relu(%149);
          %151 = nn.conv2d(%150, meta[relay.Constant][195], padding=[0, 0, 0, 0], channels=96, kernel_size=[1, 1]);
          %152 = nn.batch_norm(%151, meta[relay.Constant][196], meta[relay.Constant][197], meta[relay.Constant][198], meta[relay.Constant][199]);
          %153 = %152.0;
          %154 = add(%153, %142);
          %155 = nn.conv2d(%154, meta[relay.Constant][200], padding=[0, 0, 0, 0], channels=576, kernel_size=[1, 1]);
          %156 = nn.batch_norm(%155, meta[relay.Constant][201], meta[relay.Constant][202], meta[relay.Constant][203], meta[relay.Constant][204]);
          %157 = %156.0;
          %158 = nn.relu(%157);
          %159 = nn.conv2d(%158, meta[relay.Constant][205], strides=[2, 2], padding=[1, 1, 1, 1], groups=576, channels=576, kernel_size=[3, 3]);
          %160 = nn.batch_norm(%159, meta[relay.Constant][206], meta[relay.Constant][207], meta[relay.Constant][208], meta[relay.Constant][209]);
          %161 = %160.0;
          %162 = nn.relu(%161);
          %163 = nn.conv2d(%162, meta[relay.Constant][210], padding=[0, 0, 0, 0], channels=160, kernel_size=[1, 1]);
          %164 = nn.batch_norm(%163, meta[relay.Constant][211], meta[relay.Constant][212], meta[relay.Constant][213], meta[relay.Constant][214]);
          %165 = %164.0;
          %166 = nn.conv2d(%165, meta[relay.Constant][215], padding=[0, 0, 0, 0], channels=960, kernel_size=[1, 1]);
          %167 = nn.batch_norm(%166, meta[relay.Constant][216], meta[relay.Constant][217], meta[relay.Constant][218], meta[relay.Constant][219]);
          %168 = %167.0;
          %169 = nn.relu(%168);
          %170 = nn.conv2d(%169, meta[relay.Constant][220], padding=[1, 1, 1, 1], groups=960, channels=960, kernel_size=[3, 3]);
          %171 = nn.batch_norm(%170, meta[relay.Constant][221], meta[relay.Constant][222], meta[relay.Constant][223], meta[relay.Constant][224]);
          %172 = %171.0;
          %173 = nn.relu(%172);
          %174 = nn.conv2d(%173, meta[relay.Constant][225], padding=[0, 0, 0, 0], channels=160, kernel_size=[1, 1]);
          %175 = nn.batch_norm(%174, meta[relay.Constant][226], meta[relay.Constant][227], meta[relay.Constant][228], meta[relay.Constant][229]);
          %176 = %175.0;
          %177 = add(%176, %165);
          %178 = nn.conv2d(%177, meta[relay.Constant][230], padding=[0, 0, 0, 0], channels=960, kernel_size=[1, 1]);
          %179 = nn.batch_norm(%178, meta[relay.Constant][231], meta[relay.Constant][232], meta[relay.Constant][233], meta[relay.Constant][234]);
          %180 = %179.0;
          %181 = nn.relu(%180);
          %182 = nn.conv2d(%181, meta[relay.Constant][235], padding=[1, 1, 1, 1], groups=960, channels=960, kernel_size=[3, 3]);
          %183 = nn.batch_norm(%182, meta[relay.Constant][236], meta[relay.Constant][237], meta[relay.Constant][238], meta[relay.Constant][239]);
          %184 = %183.0;
          %185 = nn.relu(%184);
          %186 = nn.conv2d(%185, meta[relay.Constant][240], padding=[0, 0, 0, 0], channels=160, kernel_size=[1, 1]);
          %187 = nn.batch_norm(%186, meta[relay.Constant][241], meta[relay.Constant][242], meta[relay.Constant][243], meta[relay.Constant][244]);
          %188 = %187.0;
          %189 = add(%188, %177);
          %190 = nn.conv2d(%189, meta[relay.Constant][245], padding=[0, 0, 0, 0], channels=960, kernel_size=[1, 1]);
          %191 = nn.batch_norm(%190, meta[relay.Constant][246], meta[relay.Constant][247], meta[relay.Constant][248], meta[relay.Constant][249]);
          %192 = %191.0;
          %193 = nn.relu(%192);
          %194 = nn.conv2d(%193, meta[relay.Constant][250], padding=[1, 1, 1, 1], groups=960, channels=960, kernel_size=[3, 3]);
          %195 = nn.batch_norm(%194, meta[relay.Constant][251], meta[relay.Constant][252], meta[relay.Constant][253], meta[relay.Constant][254]);
          %196 = %195.0;
          %197 = nn.relu(%196);
          %198 = nn.conv2d(%197, meta[relay.Constant][255], padding=[0, 0, 0, 0], channels=320, kernel_size=[1, 1]);
          %199 = nn.batch_norm(%198, meta[relay.Constant][256], meta[relay.Constant][257], meta[relay.Constant][258], meta[relay.Constant][259]);
          %200 = %199.0;
          %201 = nn.conv2d(%200, meta[relay.Constant][260], padding=[0, 0, 0, 0], channels=1280, kernel_size=[1, 1]);
          %202 = nn.batch_norm(%201, meta[relay.Constant][261], meta[relay.Constant][262], meta[relay.Constant][263], meta[relay.Constant][264]);
          %203 = %202.0;
          %204 = nn.relu(%203);
          %205 = nn.global_avg_pool2d(%204);
          %206 = nn.conv2d(%205, meta[relay.Constant][265], padding=[0, 0, 0, 0], channels=1000, kernel_size=[1, 1]);
          reshape(%206, newshape=[0, -1])
        }
        """,
        "from_string",
        None,
        metatable,
    )
    return {
        "name": "mobilenet",
        "input_shapes": {"data": [1, 3, 224, 224]},
        "input_dtypes": {"data": "float32"},
        "mod": mod,
        "params": None,
        "main_dtype": "float32",
    }


def mobilenet_16():
    metatable = {"relay.Constant": mobilenet_consts("float16")}
    mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%data: Tensor[(1, 3, 224, 224), float16]) -> Tensor[(1, 1000), float16] {
          %0 = nn.conv2d(%data, meta[relay.Constant][0], strides=[2, 2], padding=[1, 1, 1, 1], channels=32, kernel_size=[3, 3]);
          %1 = nn.batch_norm(%0, meta[relay.Constant][1], meta[relay.Constant][2], meta[relay.Constant][3], meta[relay.Constant][4]);
          %2 = %1.0;
          %3 = nn.relu(%2);
          %4 = nn.conv2d(%3, meta[relay.Constant][5], padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1]);
          %5 = nn.batch_norm(%4, meta[relay.Constant][6], meta[relay.Constant][7], meta[relay.Constant][8], meta[relay.Constant][9]);
          %6 = %5.0;
          %7 = nn.relu(%6);
          %8 = nn.conv2d(%7, meta[relay.Constant][10], padding=[1, 1, 1, 1], groups=32, channels=32, kernel_size=[3, 3]);
          %9 = nn.batch_norm(%8, meta[relay.Constant][11], meta[relay.Constant][12], meta[relay.Constant][13], meta[relay.Constant][14]);
          %10 = %9.0;
          %11 = nn.relu(%10);
          %12 = nn.conv2d(%11, meta[relay.Constant][15], padding=[0, 0, 0, 0], channels=16, kernel_size=[1, 1]);
          %13 = nn.batch_norm(%12, meta[relay.Constant][16], meta[relay.Constant][17], meta[relay.Constant][18], meta[relay.Constant][19]);
          %14 = %13.0;
          %15 = nn.conv2d(%14, meta[relay.Constant][20], padding=[0, 0, 0, 0], channels=96, kernel_size=[1, 1]);
          %16 = nn.batch_norm(%15, meta[relay.Constant][21], meta[relay.Constant][22], meta[relay.Constant][23], meta[relay.Constant][24]);
          %17 = %16.0;
          %18 = nn.relu(%17);
          %19 = nn.conv2d(%18, meta[relay.Constant][25], strides=[2, 2], padding=[1, 1, 1, 1], groups=96, channels=96, kernel_size=[3, 3]);
          %20 = nn.batch_norm(%19, meta[relay.Constant][26], meta[relay.Constant][27], meta[relay.Constant][28], meta[relay.Constant][29]);
          %21 = %20.0;
          %22 = nn.relu(%21);
          %23 = nn.conv2d(%22, meta[relay.Constant][30], padding=[0, 0, 0, 0], channels=24, kernel_size=[1, 1]);
          %24 = nn.batch_norm(%23, meta[relay.Constant][31], meta[relay.Constant][32], meta[relay.Constant][33], meta[relay.Constant][34]);
          %25 = %24.0;
          %26 = nn.conv2d(%25, meta[relay.Constant][35], padding=[0, 0, 0, 0], channels=144, kernel_size=[1, 1]);
          %27 = nn.batch_norm(%26, meta[relay.Constant][36], meta[relay.Constant][37], meta[relay.Constant][38], meta[relay.Constant][39]);
          %28 = %27.0;
          %29 = nn.relu(%28);
          %30 = nn.conv2d(%29, meta[relay.Constant][40], padding=[1, 1, 1, 1], groups=144, channels=144, kernel_size=[3, 3]);
          %31 = nn.batch_norm(%30, meta[relay.Constant][41], meta[relay.Constant][42], meta[relay.Constant][43], meta[relay.Constant][44]);
          %32 = %31.0;
          %33 = nn.relu(%32);
          %34 = nn.conv2d(%33, meta[relay.Constant][45], padding=[0, 0, 0, 0], channels=24, kernel_size=[1, 1]);
          %35 = nn.batch_norm(%34, meta[relay.Constant][46], meta[relay.Constant][47], meta[relay.Constant][48], meta[relay.Constant][49]);
          %36 = %35.0;
          %37 = add(%36, %25);
          %38 = nn.conv2d(%37, meta[relay.Constant][50], padding=[0, 0, 0, 0], channels=144, kernel_size=[1, 1]);
          %39 = nn.batch_norm(%38, meta[relay.Constant][51], meta[relay.Constant][52], meta[relay.Constant][53], meta[relay.Constant][54]);
          %40 = %39.0;
          %41 = nn.relu(%40);
          %42 = nn.conv2d(%41, meta[relay.Constant][55], strides=[2, 2], padding=[1, 1, 1, 1], groups=144, channels=144, kernel_size=[3, 3]);
          %43 = nn.batch_norm(%42, meta[relay.Constant][56], meta[relay.Constant][57], meta[relay.Constant][58], meta[relay.Constant][59]);
          %44 = %43.0;
          %45 = nn.relu(%44);
          %46 = nn.conv2d(%45, meta[relay.Constant][60], padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1]);
          %47 = nn.batch_norm(%46, meta[relay.Constant][61], meta[relay.Constant][62], meta[relay.Constant][63], meta[relay.Constant][64]);
          %48 = %47.0;
          %49 = nn.conv2d(%48, meta[relay.Constant][65], padding=[0, 0, 0, 0], channels=192, kernel_size=[1, 1]);
          %50 = nn.batch_norm(%49, meta[relay.Constant][66], meta[relay.Constant][67], meta[relay.Constant][68], meta[relay.Constant][69]);
          %51 = %50.0;
          %52 = nn.relu(%51);
          %53 = nn.conv2d(%52, meta[relay.Constant][70], padding=[1, 1, 1, 1], groups=192, channels=192, kernel_size=[3, 3]);
          %54 = nn.batch_norm(%53, meta[relay.Constant][71], meta[relay.Constant][72], meta[relay.Constant][73], meta[relay.Constant][74]);
          %55 = %54.0;
          %56 = nn.relu(%55);
          %57 = nn.conv2d(%56, meta[relay.Constant][75], padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1]);
          %58 = nn.batch_norm(%57, meta[relay.Constant][76], meta[relay.Constant][77], meta[relay.Constant][78], meta[relay.Constant][79]);
          %59 = %58.0;
          %60 = add(%59, %48);
          %61 = nn.conv2d(%60, meta[relay.Constant][80], padding=[0, 0, 0, 0], channels=192, kernel_size=[1, 1]);
          %62 = nn.batch_norm(%61, meta[relay.Constant][81], meta[relay.Constant][82], meta[relay.Constant][83], meta[relay.Constant][84]);
          %63 = %62.0;
          %64 = nn.relu(%63);
          %65 = nn.conv2d(%64, meta[relay.Constant][85], padding=[1, 1, 1, 1], groups=192, channels=192, kernel_size=[3, 3]);
          %66 = nn.batch_norm(%65, meta[relay.Constant][86], meta[relay.Constant][87], meta[relay.Constant][88], meta[relay.Constant][89]);
          %67 = %66.0;
          %68 = nn.relu(%67);
          %69 = nn.conv2d(%68, meta[relay.Constant][90], padding=[0, 0, 0, 0], channels=32, kernel_size=[1, 1]);
          %70 = nn.batch_norm(%69, meta[relay.Constant][91], meta[relay.Constant][92], meta[relay.Constant][93], meta[relay.Constant][94]);
          %71 = %70.0;
          %72 = add(%71, %60);
          %73 = nn.conv2d(%72, meta[relay.Constant][95], padding=[0, 0, 0, 0], channels=192, kernel_size=[1, 1]);
          %74 = nn.batch_norm(%73, meta[relay.Constant][96], meta[relay.Constant][97], meta[relay.Constant][98], meta[relay.Constant][99]);
          %75 = %74.0;
          %76 = nn.relu(%75);
          %77 = nn.conv2d(%76, meta[relay.Constant][100], padding=[1, 1, 1, 1], groups=192, channels=192, kernel_size=[3, 3]);
          %78 = nn.batch_norm(%77, meta[relay.Constant][101], meta[relay.Constant][102], meta[relay.Constant][103], meta[relay.Constant][104]);
          %79 = %78.0;
          %80 = nn.relu(%79);
          %81 = nn.conv2d(%80, meta[relay.Constant][105], padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]);
          %82 = nn.batch_norm(%81, meta[relay.Constant][106], meta[relay.Constant][107], meta[relay.Constant][108], meta[relay.Constant][109]);
          %83 = %82.0;
          %84 = nn.conv2d(%83, meta[relay.Constant][110], padding=[0, 0, 0, 0], channels=384, kernel_size=[1, 1]);
          %85 = nn.batch_norm(%84, meta[relay.Constant][111], meta[relay.Constant][112], meta[relay.Constant][113], meta[relay.Constant][114]);
          %86 = %85.0;
          %87 = nn.relu(%86);
          %88 = nn.conv2d(%87, meta[relay.Constant][115], padding=[1, 1, 1, 1], groups=384, channels=384, kernel_size=[3, 3]);
          %89 = nn.batch_norm(%88, meta[relay.Constant][116], meta[relay.Constant][117], meta[relay.Constant][118], meta[relay.Constant][119]);
          %90 = %89.0;
          %91 = nn.relu(%90);
          %92 = nn.conv2d(%91, meta[relay.Constant][120], padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]);
          %93 = nn.batch_norm(%92, meta[relay.Constant][121], meta[relay.Constant][122], meta[relay.Constant][123], meta[relay.Constant][124]);
          %94 = %93.0;
          %95 = add(%94, %83);
          %96 = nn.conv2d(%95, meta[relay.Constant][125], padding=[0, 0, 0, 0], channels=384, kernel_size=[1, 1]);
          %97 = nn.batch_norm(%96, meta[relay.Constant][126], meta[relay.Constant][127], meta[relay.Constant][128], meta[relay.Constant][129]);
          %98 = %97.0;
          %99 = nn.relu(%98);
          %100 = nn.conv2d(%99, meta[relay.Constant][130], padding=[1, 1, 1, 1], groups=384, channels=384, kernel_size=[3, 3]);
          %101 = nn.batch_norm(%100, meta[relay.Constant][131], meta[relay.Constant][132], meta[relay.Constant][133], meta[relay.Constant][134]);
          %102 = %101.0;
          %103 = nn.relu(%102);
          %104 = nn.conv2d(%103, meta[relay.Constant][135], padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]);
          %105 = nn.batch_norm(%104, meta[relay.Constant][136], meta[relay.Constant][137], meta[relay.Constant][138], meta[relay.Constant][139]);
          %106 = %105.0;
          %107 = add(%106, %95);
          %108 = nn.conv2d(%107, meta[relay.Constant][140], padding=[0, 0, 0, 0], channels=384, kernel_size=[1, 1]);
          %109 = nn.batch_norm(%108, meta[relay.Constant][141], meta[relay.Constant][142], meta[relay.Constant][143], meta[relay.Constant][144]);
          %110 = %109.0;
          %111 = nn.relu(%110);
          %112 = nn.conv2d(%111, meta[relay.Constant][145], padding=[1, 1, 1, 1], groups=384, channels=384, kernel_size=[3, 3]);
          %113 = nn.batch_norm(%112, meta[relay.Constant][146], meta[relay.Constant][147], meta[relay.Constant][148], meta[relay.Constant][149]);
          %114 = %113.0;
          %115 = nn.relu(%114);
          %116 = nn.conv2d(%115, meta[relay.Constant][150], padding=[0, 0, 0, 0], channels=64, kernel_size=[1, 1]);
          %117 = nn.batch_norm(%116, meta[relay.Constant][151], meta[relay.Constant][152], meta[relay.Constant][153], meta[relay.Constant][154]);
          %118 = %117.0;
          %119 = add(%118, %107);
          %120 = nn.conv2d(%119, meta[relay.Constant][155], padding=[0, 0, 0, 0], channels=384, kernel_size=[1, 1]);
          %121 = nn.batch_norm(%120, meta[relay.Constant][156], meta[relay.Constant][157], meta[relay.Constant][158], meta[relay.Constant][159]);
          %122 = %121.0;
          %123 = nn.relu(%122);
          %124 = nn.conv2d(%123, meta[relay.Constant][160], strides=[2, 2], padding=[1, 1, 1, 1], groups=384, channels=384, kernel_size=[3, 3]);
          %125 = nn.batch_norm(%124, meta[relay.Constant][161], meta[relay.Constant][162], meta[relay.Constant][163], meta[relay.Constant][164]);
          %126 = %125.0;
          %127 = nn.relu(%126);
          %128 = nn.conv2d(%127, meta[relay.Constant][165], padding=[0, 0, 0, 0], channels=96, kernel_size=[1, 1]);
          %129 = nn.batch_norm(%128, meta[relay.Constant][166], meta[relay.Constant][167], meta[relay.Constant][168], meta[relay.Constant][169]);
          %130 = %129.0;
          %131 = nn.conv2d(%130, meta[relay.Constant][170], padding=[0, 0, 0, 0], channels=576, kernel_size=[1, 1]);
          %132 = nn.batch_norm(%131, meta[relay.Constant][171], meta[relay.Constant][172], meta[relay.Constant][173], meta[relay.Constant][174]);
          %133 = %132.0;
          %134 = nn.relu(%133);
          %135 = nn.conv2d(%134, meta[relay.Constant][175], padding=[1, 1, 1, 1], groups=576, channels=576, kernel_size=[3, 3]);
          %136 = nn.batch_norm(%135, meta[relay.Constant][176], meta[relay.Constant][177], meta[relay.Constant][178], meta[relay.Constant][179]);
          %137 = %136.0;
          %138 = nn.relu(%137);
          %139 = nn.conv2d(%138, meta[relay.Constant][180], padding=[0, 0, 0, 0], channels=96, kernel_size=[1, 1]);
          %140 = nn.batch_norm(%139, meta[relay.Constant][181], meta[relay.Constant][182], meta[relay.Constant][183], meta[relay.Constant][184]);
          %141 = %140.0;
          %142 = add(%141, %130);
          %143 = nn.conv2d(%142, meta[relay.Constant][185], padding=[0, 0, 0, 0], channels=576, kernel_size=[1, 1]);
          %144 = nn.batch_norm(%143, meta[relay.Constant][186], meta[relay.Constant][187], meta[relay.Constant][188], meta[relay.Constant][189]);
          %145 = %144.0;
          %146 = nn.relu(%145);
          %147 = nn.conv2d(%146, meta[relay.Constant][190], padding=[1, 1, 1, 1], groups=576, channels=576, kernel_size=[3, 3]);
          %148 = nn.batch_norm(%147, meta[relay.Constant][191], meta[relay.Constant][192], meta[relay.Constant][193], meta[relay.Constant][194]);
          %149 = %148.0;
          %150 = nn.relu(%149);
          %151 = nn.conv2d(%150, meta[relay.Constant][195], padding=[0, 0, 0, 0], channels=96, kernel_size=[1, 1]);
          %152 = nn.batch_norm(%151, meta[relay.Constant][196], meta[relay.Constant][197], meta[relay.Constant][198], meta[relay.Constant][199]);
          %153 = %152.0;
          %154 = add(%153, %142);
          %155 = nn.conv2d(%154, meta[relay.Constant][200], padding=[0, 0, 0, 0], channels=576, kernel_size=[1, 1]);
          %156 = nn.batch_norm(%155, meta[relay.Constant][201], meta[relay.Constant][202], meta[relay.Constant][203], meta[relay.Constant][204]);
          %157 = %156.0;
          %158 = nn.relu(%157);
          %159 = nn.conv2d(%158, meta[relay.Constant][205], strides=[2, 2], padding=[1, 1, 1, 1], groups=576, channels=576, kernel_size=[3, 3]);
          %160 = nn.batch_norm(%159, meta[relay.Constant][206], meta[relay.Constant][207], meta[relay.Constant][208], meta[relay.Constant][209]);
          %161 = %160.0;
          %162 = nn.relu(%161);
          %163 = nn.conv2d(%162, meta[relay.Constant][210], padding=[0, 0, 0, 0], channels=160, kernel_size=[1, 1]);
          %164 = nn.batch_norm(%163, meta[relay.Constant][211], meta[relay.Constant][212], meta[relay.Constant][213], meta[relay.Constant][214]);
          %165 = %164.0;
          %166 = nn.conv2d(%165, meta[relay.Constant][215], padding=[0, 0, 0, 0], channels=960, kernel_size=[1, 1]);
          %167 = nn.batch_norm(%166, meta[relay.Constant][216], meta[relay.Constant][217], meta[relay.Constant][218], meta[relay.Constant][219]);
          %168 = %167.0;
          %169 = nn.relu(%168);
          %170 = nn.conv2d(%169, meta[relay.Constant][220], padding=[1, 1, 1, 1], groups=960, channels=960, kernel_size=[3, 3]);
          %171 = nn.batch_norm(%170, meta[relay.Constant][221], meta[relay.Constant][222], meta[relay.Constant][223], meta[relay.Constant][224]);
          %172 = %171.0;
          %173 = nn.relu(%172);
          %174 = nn.conv2d(%173, meta[relay.Constant][225], padding=[0, 0, 0, 0], channels=160, kernel_size=[1, 1]);
          %175 = nn.batch_norm(%174, meta[relay.Constant][226], meta[relay.Constant][227], meta[relay.Constant][228], meta[relay.Constant][229]);
          %176 = %175.0;
          %177 = add(%176, %165);
          %178 = nn.conv2d(%177, meta[relay.Constant][230], padding=[0, 0, 0, 0], channels=960, kernel_size=[1, 1]);
          %179 = nn.batch_norm(%178, meta[relay.Constant][231], meta[relay.Constant][232], meta[relay.Constant][233], meta[relay.Constant][234]);
          %180 = %179.0;
          %181 = nn.relu(%180);
          %182 = nn.conv2d(%181, meta[relay.Constant][235], padding=[1, 1, 1, 1], groups=960, channels=960, kernel_size=[3, 3]);
          %183 = nn.batch_norm(%182, meta[relay.Constant][236], meta[relay.Constant][237], meta[relay.Constant][238], meta[relay.Constant][239]);
          %184 = %183.0;
          %185 = nn.relu(%184);
          %186 = nn.conv2d(%185, meta[relay.Constant][240], padding=[0, 0, 0, 0], channels=160, kernel_size=[1, 1]);
          %187 = nn.batch_norm(%186, meta[relay.Constant][241], meta[relay.Constant][242], meta[relay.Constant][243], meta[relay.Constant][244]);
          %188 = %187.0;
          %189 = add(%188, %177);
          %190 = nn.conv2d(%189, meta[relay.Constant][245], padding=[0, 0, 0, 0], channels=960, kernel_size=[1, 1]);
          %191 = nn.batch_norm(%190, meta[relay.Constant][246], meta[relay.Constant][247], meta[relay.Constant][248], meta[relay.Constant][249]);
          %192 = %191.0;
          %193 = nn.relu(%192);
          %194 = nn.conv2d(%193, meta[relay.Constant][250], padding=[1, 1, 1, 1], groups=960, channels=960, kernel_size=[3, 3]);
          %195 = nn.batch_norm(%194, meta[relay.Constant][251], meta[relay.Constant][252], meta[relay.Constant][253], meta[relay.Constant][254]);
          %196 = %195.0;
          %197 = nn.relu(%196);
          %198 = nn.conv2d(%197, meta[relay.Constant][255], padding=[0, 0, 0, 0], channels=320, kernel_size=[1, 1]);
          %199 = nn.batch_norm(%198, meta[relay.Constant][256], meta[relay.Constant][257], meta[relay.Constant][258], meta[relay.Constant][259]);
          %200 = %199.0;
          %201 = nn.conv2d(%200, meta[relay.Constant][260], padding=[0, 0, 0, 0], channels=1280, kernel_size=[1, 1]);
          %202 = nn.batch_norm(%201, meta[relay.Constant][261], meta[relay.Constant][262], meta[relay.Constant][263], meta[relay.Constant][264]);
          %203 = %202.0;
          %204 = nn.relu(%203);
          %205 = nn.global_avg_pool2d(%204);
          %206 = nn.conv2d(%205, meta[relay.Constant][265], padding=[0, 0, 0, 0], channels=1000, kernel_size=[1, 1]);
          reshape(%206, newshape=[0, -1])
        }
        """,
        "from_string",
        None,
        metatable,
    )
    return {
        "name": "mobilenet_16",
        "input_shapes": {"data": [1, 3, 224, 224]},
        "input_dtypes": {"data": "float16"},
        "mod": mod,
        "params": None,
        "main_dtype": "float16",
    }


def batch_norm_extract():
    consts = make_consts(
        "float32",
        [
            (32,),  # 0
            (32,),  # 1
            (32,),  # 2
            (32,),  # 3
        ],
    )
    metatable = {"relay.Constant": consts}
    mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%FunctionVar_0: Tensor[(1, 32, 112, 112), float32]) -> Tensor[(1, 32, 112, 112), float32] {
          %3 = nn.batch_norm(%FunctionVar_0, meta[relay.Constant][0], meta[relay.Constant][1], meta[relay.Constant][2], meta[relay.Constant][3]);
          %3.0
        }
        """,
        "from_string",
        None,
        metatable,
    )
    return {
        "name": "batch_norm_extract",
        "input_shapes": {"FunctionVar_0": [1, 32, 112, 112]},
        "input_dtypes": {"FunctionVar_0": "float32"},
        "mod": mod,
        "params": None,
        "main_dtype": "float32",
    }


def resnext50_32x4d_consts(dtype):
    return make_consts(
        dtype,
        [
            (128, 64, 1, 1),  # 0
            (128, 4, 3, 3),  # 1
            (256, 128, 1, 1),  # 2
            (256, 64, 1, 1),  # 3
            (128, 256, 1, 1),  # 4
            (128, 4, 3, 3),  # 5
            (256, 128, 1, 1),  # 6
            (128, 256, 1, 1),  # 7
            (128, 4, 3, 3),  # 8
            (256, 128, 1, 1),  # 9
            (256, 256, 1, 1),  # 10
            (256, 8, 3, 3),  # 11
            (512, 256, 1, 1),  # 12
            (512, 256, 1, 1),  # 13
            (256, 512, 1, 1),  # 14
            (256, 8, 3, 3),  # 15
            (512, 256, 1, 1),  # 16
            (256, 512, 1, 1),  # 17
            (256, 8, 3, 3),  # 18
            (512, 256, 1, 1),  # 19
            (256, 512, 1, 1),  # 20
            (256, 8, 3, 3),  # 21
            (512, 256, 1, 1),  # 22
            (512, 512, 1, 1),  # 23
            (512, 16, 3, 3),  # 24
            (1024, 512, 1, 1),  # 25
            (1024, 512, 1, 1),  # 26
            (512, 1024, 1, 1),  # 27
            (512, 16, 3, 3),  # 28
            (1024, 512, 1, 1),  # 29
            (512, 1024, 1, 1),  # 30
            (512, 16, 3, 3),  # 31
            (1024, 512, 1, 1),  # 32
            (512, 1024, 1, 1),  # 33
            (512, 16, 3, 3),  # 34
            (1024, 512, 1, 1),  # 35
            (512, 1024, 1, 1),  # 36
            (512, 16, 3, 3),  # 37
            (1024, 512, 1, 1),  # 38
            (512, 1024, 1, 1),  # 39
            (512, 16, 3, 3),  # 40
            (1024, 512, 1, 1),  # 41
            (1024, 1024, 1, 1),  # 42
            (1024, 32, 3, 3),  # 43
            (2048, 1024, 1, 1),  # 44
            (2048, 1024, 1, 1),  # 45
            (1024, 2048, 1, 1),  # 46
            (1024, 32, 3, 3),  # 47
            (2048, 1024, 1, 1),  # 48
            (1024, 2048, 1, 1),  # 49
            (1024, 32, 3, 3),  # 50
            (2048, 1024, 1, 1),  # 51
        ],
    )


def resnext50_32x4d():
    metatable = {"relay.Constant": resnext50_32x4d_consts("float32")}
    mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%x: Tensor[(1, 64, 56, 56), float32]) {
          %0 = nn.conv2d(%x, meta[relay.Constant][0], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]);
          %1 = nn.relu(%0);
          %2 = nn.conv2d(%1, meta[relay.Constant][1], padding=[1, 1, 1, 1], groups=32, channels=128, kernel_size=[3, 3]);
          %3 = nn.relu(%2);
          %4 = nn.conv2d(%3, meta[relay.Constant][2], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %5 = nn.conv2d(%x, meta[relay.Constant][3], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %6 = add(%4, %5);
          %7 = nn.relu(%6);
          %8 = nn.conv2d(%7, meta[relay.Constant][4], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]);
          %9 = nn.relu(%8);
          %10 = nn.conv2d(%9, meta[relay.Constant][5], padding=[1, 1, 1, 1], groups=32, channels=128, kernel_size=[3, 3]);
          %11 = nn.relu(%10);
          %12 = nn.conv2d(%11, meta[relay.Constant][6], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %13 = add(%12, %7);
          %14 = nn.relu(%13);
          %15 = nn.conv2d(%14, meta[relay.Constant][7], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]);
          %16 = nn.relu(%15);
          %17 = nn.conv2d(%16, meta[relay.Constant][8], padding=[1, 1, 1, 1], groups=32, channels=128, kernel_size=[3, 3]);
          %18 = nn.relu(%17);
          %19 = nn.conv2d(%18, meta[relay.Constant][9], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %20 = add(%19, %14);
          %21 = nn.relu(%20);
          %22 = nn.conv2d(%21, meta[relay.Constant][10], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %23 = nn.relu(%22);
          %24 = nn.conv2d(%23, meta[relay.Constant][11], strides=[2, 2], padding=[1, 1, 1, 1], groups=32, channels=256, kernel_size=[3, 3]);
          %25 = nn.relu(%24);
          %26 = nn.conv2d(%25, meta[relay.Constant][12], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %27 = nn.conv2d(%21, meta[relay.Constant][13], strides=[2, 2], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %28 = add(%26, %27);
          %29 = nn.relu(%28);
          %30 = nn.conv2d(%29, meta[relay.Constant][14], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %31 = nn.relu(%30);
          %32 = nn.conv2d(%31, meta[relay.Constant][15], padding=[1, 1, 1, 1], groups=32, channels=256, kernel_size=[3, 3]);
          %33 = nn.relu(%32);
          %34 = nn.conv2d(%33, meta[relay.Constant][16], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %35 = add(%34, %29);
          %36 = nn.relu(%35);
          %37 = nn.conv2d(%36, meta[relay.Constant][17], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %38 = nn.relu(%37);
          %39 = nn.conv2d(%38, meta[relay.Constant][18], padding=[1, 1, 1, 1], groups=32, channels=256, kernel_size=[3, 3]);
          %40 = nn.relu(%39);
          %41 = nn.conv2d(%40, meta[relay.Constant][19], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %42 = add(%41, %36);
          %43 = nn.relu(%42);
          %44 = nn.conv2d(%43, meta[relay.Constant][20], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %45 = nn.relu(%44);
          %46 = nn.conv2d(%45, meta[relay.Constant][21], padding=[1, 1, 1, 1], groups=32, channels=256, kernel_size=[3, 3]);
          %47 = nn.relu(%46);
          %48 = nn.conv2d(%47, meta[relay.Constant][22], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %49 = add(%48, %43);
          %50 = nn.relu(%49);
          %51 = nn.conv2d(%50, meta[relay.Constant][23], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %52 = nn.relu(%51);
          %53 = nn.conv2d(%52, meta[relay.Constant][24], strides=[2, 2], padding=[1, 1, 1, 1], groups=32, channels=512, kernel_size=[3, 3]);
          %54 = nn.relu(%53);
          %55 = nn.conv2d(%54, meta[relay.Constant][25], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %56 = nn.conv2d(%50, meta[relay.Constant][26], strides=[2, 2], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %57 = add(%55, %56);
          %58 = nn.relu(%57);
          %59 = nn.conv2d(%58, meta[relay.Constant][27], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %60 = nn.relu(%59);
          %61 = nn.conv2d(%60, meta[relay.Constant][28], padding=[1, 1, 1, 1], groups=32, channels=512, kernel_size=[3, 3]);
          %62 = nn.relu(%61);
          %63 = nn.conv2d(%62, meta[relay.Constant][29], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %64 = add(%63, %58);
          %65 = nn.relu(%64);
          %66 = nn.conv2d(%65, meta[relay.Constant][30], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %67 = nn.relu(%66);
          %68 = nn.conv2d(%67, meta[relay.Constant][31], padding=[1, 1, 1, 1], groups=32, channels=512, kernel_size=[3, 3]);
          %69 = nn.relu(%68);
          %70 = nn.conv2d(%69, meta[relay.Constant][32], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %71 = add(%70, %65);
          %72 = nn.relu(%71);
          %73 = nn.conv2d(%72, meta[relay.Constant][33], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %74 = nn.relu(%73);
          %75 = nn.conv2d(%74, meta[relay.Constant][34], padding=[1, 1, 1, 1], groups=32, channels=512, kernel_size=[3, 3]);
          %76 = nn.relu(%75);
          %77 = nn.conv2d(%76, meta[relay.Constant][35], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %78 = add(%77, %72);
          %79 = nn.relu(%78);
          %80 = nn.conv2d(%79, meta[relay.Constant][36], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %81 = nn.relu(%80);
          %82 = nn.conv2d(%81, meta[relay.Constant][37], padding=[1, 1, 1, 1], groups=32, channels=512, kernel_size=[3, 3]);
          %83 = nn.relu(%82);
          %84 = nn.conv2d(%83, meta[relay.Constant][38], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %85 = add(%84, %79);
          %86 = nn.relu(%85);
          %87 = nn.conv2d(%86, meta[relay.Constant][39], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %88 = nn.relu(%87);
          %89 = nn.conv2d(%88, meta[relay.Constant][40], padding=[1, 1, 1, 1], groups=32, channels=512, kernel_size=[3, 3]);
          %90 = nn.relu(%89);
          %91 = nn.conv2d(%90, meta[relay.Constant][41], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %92 = add(%91, %86);
          %93 = nn.relu(%92);
          %94 = nn.conv2d(%93, meta[relay.Constant][42], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %95 = nn.relu(%94);
          %96 = nn.conv2d(%95, meta[relay.Constant][43], strides=[2, 2], padding=[1, 1, 1, 1], groups=32, channels=1024, kernel_size=[3, 3]);
          %97 = nn.relu(%96);
          %98 = nn.conv2d(%97, meta[relay.Constant][44], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1]);
          %99 = nn.conv2d(%93, meta[relay.Constant][45], strides=[2, 2], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1]);
          %100 = add(%98, %99);
          %101 = nn.relu(%100);
          %102 = nn.conv2d(%101, meta[relay.Constant][46], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %103 = nn.relu(%102);
          %104 = nn.conv2d(%103, meta[relay.Constant][47], padding=[1, 1, 1, 1], groups=32, channels=1024, kernel_size=[3, 3]);
          %105 = nn.relu(%104);
          %106 = nn.conv2d(%105, meta[relay.Constant][48], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1]);
          %107 = add(%106, %101);
          %108 = nn.relu(%107);
          %109 = nn.conv2d(%108, meta[relay.Constant][49], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %110 = nn.relu(%109);
          %111 = nn.conv2d(%110, meta[relay.Constant][50], padding=[1, 1, 1, 1], groups=32, channels=1024, kernel_size=[3, 3]);
          %112 = nn.relu(%111);
          %113 = nn.conv2d(%112, meta[relay.Constant][51], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1]);
          %114 = add(%113, %108);
          nn.relu(%114)
        }
        """,
        "from_string",
        None,
        metatable,
    )
    return {
        "name": "resnext50_32x4d",
        "input_shapes": {"x": [1, 64, 56, 56]},
        "input_dtypes": {"x": "float32"},
        "mod": mod,
        "params": None,
        "main_dtype": "float32",
    }


def resnext50_32x4d_16():
    metatable = {"relay.Constant": resnext50_32x4d_consts("float16")}
    mod = tvm.parser.parse(
        """
        #[version = "0.0.5"]
        def @main(%x: Tensor[(1, 64, 56, 56), float16]) {
          %0 = nn.conv2d(%x, meta[relay.Constant][0], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]);
          %1 = nn.relu(%0);
          %2 = nn.conv2d(%1, meta[relay.Constant][1], padding=[1, 1, 1, 1], groups=32, channels=128, kernel_size=[3, 3]);
          %3 = nn.relu(%2);
          %4 = nn.conv2d(%3, meta[relay.Constant][2], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %5 = nn.conv2d(%x, meta[relay.Constant][3], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %6 = add(%4, %5);
          %7 = nn.relu(%6);
          %8 = nn.conv2d(%7, meta[relay.Constant][4], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]);
          %9 = nn.relu(%8);
          %10 = nn.conv2d(%9, meta[relay.Constant][5], padding=[1, 1, 1, 1], groups=32, channels=128, kernel_size=[3, 3]);
          %11 = nn.relu(%10);
          %12 = nn.conv2d(%11, meta[relay.Constant][6], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %13 = add(%12, %7);
          %14 = nn.relu(%13);
          %15 = nn.conv2d(%14, meta[relay.Constant][7], padding=[0, 0, 0, 0], channels=128, kernel_size=[1, 1]);
          %16 = nn.relu(%15);
          %17 = nn.conv2d(%16, meta[relay.Constant][8], padding=[1, 1, 1, 1], groups=32, channels=128, kernel_size=[3, 3]);
          %18 = nn.relu(%17);
          %19 = nn.conv2d(%18, meta[relay.Constant][9], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %20 = add(%19, %14);
          %21 = nn.relu(%20);
          %22 = nn.conv2d(%21, meta[relay.Constant][10], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %23 = nn.relu(%22);
          %24 = nn.conv2d(%23, meta[relay.Constant][11], strides=[2, 2], padding=[1, 1, 1, 1], groups=32, channels=256, kernel_size=[3, 3]);
          %25 = nn.relu(%24);
          %26 = nn.conv2d(%25, meta[relay.Constant][12], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %27 = nn.conv2d(%21, meta[relay.Constant][13], strides=[2, 2], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %28 = add(%26, %27);
          %29 = nn.relu(%28);
          %30 = nn.conv2d(%29, meta[relay.Constant][14], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %31 = nn.relu(%30);
          %32 = nn.conv2d(%31, meta[relay.Constant][15], padding=[1, 1, 1, 1], groups=32, channels=256, kernel_size=[3, 3]);
          %33 = nn.relu(%32);
          %34 = nn.conv2d(%33, meta[relay.Constant][16], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %35 = add(%34, %29);
          %36 = nn.relu(%35);
          %37 = nn.conv2d(%36, meta[relay.Constant][17], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %38 = nn.relu(%37);
          %39 = nn.conv2d(%38, meta[relay.Constant][18], padding=[1, 1, 1, 1], groups=32, channels=256, kernel_size=[3, 3]);
          %40 = nn.relu(%39);
          %41 = nn.conv2d(%40, meta[relay.Constant][19], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %42 = add(%41, %36);
          %43 = nn.relu(%42);
          %44 = nn.conv2d(%43, meta[relay.Constant][20], padding=[0, 0, 0, 0], channels=256, kernel_size=[1, 1]);
          %45 = nn.relu(%44);
          %46 = nn.conv2d(%45, meta[relay.Constant][21], padding=[1, 1, 1, 1], groups=32, channels=256, kernel_size=[3, 3]);
          %47 = nn.relu(%46);
          %48 = nn.conv2d(%47, meta[relay.Constant][22], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %49 = add(%48, %43);
          %50 = nn.relu(%49);
          %51 = nn.conv2d(%50, meta[relay.Constant][23], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %52 = nn.relu(%51);
          %53 = nn.conv2d(%52, meta[relay.Constant][24], strides=[2, 2], padding=[1, 1, 1, 1], groups=32, channels=512, kernel_size=[3, 3]);
          %54 = nn.relu(%53);
          %55 = nn.conv2d(%54, meta[relay.Constant][25], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %56 = nn.conv2d(%50, meta[relay.Constant][26], strides=[2, 2], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %57 = add(%55, %56);
          %58 = nn.relu(%57);
          %59 = nn.conv2d(%58, meta[relay.Constant][27], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %60 = nn.relu(%59);
          %61 = nn.conv2d(%60, meta[relay.Constant][28], padding=[1, 1, 1, 1], groups=32, channels=512, kernel_size=[3, 3]);
          %62 = nn.relu(%61);
          %63 = nn.conv2d(%62, meta[relay.Constant][29], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %64 = add(%63, %58);
          %65 = nn.relu(%64);
          %66 = nn.conv2d(%65, meta[relay.Constant][30], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %67 = nn.relu(%66);
          %68 = nn.conv2d(%67, meta[relay.Constant][31], padding=[1, 1, 1, 1], groups=32, channels=512, kernel_size=[3, 3]);
          %69 = nn.relu(%68);
          %70 = nn.conv2d(%69, meta[relay.Constant][32], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %71 = add(%70, %65);
          %72 = nn.relu(%71);
          %73 = nn.conv2d(%72, meta[relay.Constant][33], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %74 = nn.relu(%73);
          %75 = nn.conv2d(%74, meta[relay.Constant][34], padding=[1, 1, 1, 1], groups=32, channels=512, kernel_size=[3, 3]);
          %76 = nn.relu(%75);
          %77 = nn.conv2d(%76, meta[relay.Constant][35], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %78 = add(%77, %72);
          %79 = nn.relu(%78);
          %80 = nn.conv2d(%79, meta[relay.Constant][36], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %81 = nn.relu(%80);
          %82 = nn.conv2d(%81, meta[relay.Constant][37], padding=[1, 1, 1, 1], groups=32, channels=512, kernel_size=[3, 3]);
          %83 = nn.relu(%82);
          %84 = nn.conv2d(%83, meta[relay.Constant][38], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %85 = add(%84, %79);
          %86 = nn.relu(%85);
          %87 = nn.conv2d(%86, meta[relay.Constant][39], padding=[0, 0, 0, 0], channels=512, kernel_size=[1, 1]);
          %88 = nn.relu(%87);
          %89 = nn.conv2d(%88, meta[relay.Constant][40], padding=[1, 1, 1, 1], groups=32, channels=512, kernel_size=[3, 3]);
          %90 = nn.relu(%89);
          %91 = nn.conv2d(%90, meta[relay.Constant][41], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %92 = add(%91, %86);
          %93 = nn.relu(%92);
          %94 = nn.conv2d(%93, meta[relay.Constant][42], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %95 = nn.relu(%94);
          %96 = nn.conv2d(%95, meta[relay.Constant][43], strides=[2, 2], padding=[1, 1, 1, 1], groups=32, channels=1024, kernel_size=[3, 3]);
          %97 = nn.relu(%96);
          %98 = nn.conv2d(%97, meta[relay.Constant][44], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1]);
          %99 = nn.conv2d(%93, meta[relay.Constant][45], strides=[2, 2], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1]);
          %100 = add(%98, %99);
          %101 = nn.relu(%100);
          %102 = nn.conv2d(%101, meta[relay.Constant][46], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %103 = nn.relu(%102);
          %104 = nn.conv2d(%103, meta[relay.Constant][47], padding=[1, 1, 1, 1], groups=32, channels=1024, kernel_size=[3, 3]);
          %105 = nn.relu(%104);
          %106 = nn.conv2d(%105, meta[relay.Constant][48], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1]);
          %107 = add(%106, %101);
          %108 = nn.relu(%107);
          %109 = nn.conv2d(%108, meta[relay.Constant][49], padding=[0, 0, 0, 0], channels=1024, kernel_size=[1, 1]);
          %110 = nn.relu(%109);
          %111 = nn.conv2d(%110, meta[relay.Constant][50], padding=[1, 1, 1, 1], groups=32, channels=1024, kernel_size=[3, 3]);
          %112 = nn.relu(%111);
          %113 = nn.conv2d(%112, meta[relay.Constant][51], padding=[0, 0, 0, 0], channels=2048, kernel_size=[1, 1]);
          %114 = add(%113, %108);
          nn.relu(%114)
        }
        """,
        "from_string",
        None,
        metatable,
    )
    return {
        "name": "resnext50_32x4d_16",
        "input_shapes": {"x": [1, 64, 56, 56]},
        "input_dtypes": {"x": "float16"},
        "mod": mod,
        "params": None,
        "main_dtype": "float16",
    }


def describe_onnx(name, filename):
    """Returns the description of the ONNX model at filename, which can be passed to from_onnx to actually load
    the model. Note that ? (ie unknown) shape dimensions must be manually changed to concrete dimensions
    which are consistent with the overall model."""
    onnx_model = onnx.load(MODEL_PREFIX + filename)
    input_shapes = {}
    input_dtypes = {}
    initializer_names = [n.name for n in onnx_model.graph.initializer]
    for input_info in onnx_model.graph.input:
        if input_info.name not in initializer_names:
            _, shape, dtype, _ = tvm.relay.frontend.onnx.get_info(input_info)
            if dtype is None:
                raise ValueError(f"Unknown dtype on input '{input_info.name}' is not supported.")
            input_shapes.update({input_info.name: shape})
            input_dtypes.update({input_info.name: dtype})
    print(
        f"{{'name': '{name}', 'filename': '{filename}', 'input_shapes': {input_shapes}, 'input_dtypes': {input_dtypes}, 'main_dtype': 'float32'}}"
    )


def from_onnx(model):
    logging.info("-------------------- BEGIN ONNX IMPORT --------------------")

    filename = MODEL_PREFIX + model["filename"]
    logging.info(f"Loading ONNX model from {filename}")

    onnx_model = onnx.load(filename)
    logging.info(f"Loaded model from {filename}")

    mod, params = tvm.relay.frontend.from_onnx(
        onnx_model, model["input_shapes"], freeze_params=True
    )
    mod = tvm.relay.transform.InferType()(mod)
    logging.info("-------------------- END ONNX IMPORT --------------------")

    logging.info(f"Imported model:\n{mod}")
    logging.info(f"Params:\n{params}")

    return {
        "name": model["name"],
        "input_shapes": model["input_shapes"],
        "input_dtypes": model["input_dtypes"],
        "mod": mod,
        "params": params,
        "main_dtype": model["main_dtype"],
    }


def to_onnx(model):
    logging.info("-------------------- BEGIN ONNX EXPORT --------------------")
    short_filename = model["name"] + ".onnx"
    filename = MODEL_PREFIX + short_filename
    logging.info(f"Saving ONNX model to {filename}")

    params = model["params"]
    if params is None:
        params = {}
    tvm.contrib.target.onnx.to_onnx(model["mod"], params, model["name"], path=filename)
    logging.info("-------------------- END ONNX EXPORT --------------------")

    return {
        "name": model["name"],
        "filename": short_filename,
        "input_shapes": model["input_shapes"],
        "input_dtypes": model["input_dtypes"],
        "main_dtype": model["main_dtype"],
    }
