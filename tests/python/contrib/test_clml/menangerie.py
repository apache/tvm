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
import numpy as np
import logging


def make_const(dtype, shape):
    return tvm.relay.const(np.random.rand(*shape).astype(dtype))


def make_consts(dtype, shapes):
    return [make_const(dtype, shape) for shape in shapes]


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
