/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Source: https://stackoverflow.com/questions/36212904/yuv-420-888-interpretation-on-samsung-galaxy-s7-camera2
#pragma version(1)
#pragma rs java_package_name(org.apache.tvm.android.androidcamerademo);
#pragma rs_fp_relaxed

int32_t width;
int32_t height;

uint picWidth, uvPixelStride, uvRowStride ;
rs_allocation ypsIn,uIn,vIn;

// The LaunchOptions ensure that the Kernel does not enter the padding  zone of Y, so yRowStride can be ignored WITHIN the Kernel.
uchar4 __attribute__((kernel)) doConvert(uint32_t x, uint32_t y) {

    // index for accessing the uIn's and vIn's
    uint uvIndex=  uvPixelStride * (x/2) + uvRowStride*(y/2);

    // get the y,u,v values
    uchar yps= rsGetElementAt_uchar(ypsIn, x, y);
    uchar u= rsGetElementAt_uchar(uIn, uvIndex);
    uchar v= rsGetElementAt_uchar(vIn, uvIndex);

    // calc argb
    int4 argb;
    argb.r = yps + v * 1436 / 1024 - 179;
    argb.g =  yps -u * 46549 / 131072 + 44 -v * 93604 / 131072 + 91;
    argb.b = yps +u * 1814 / 1024 - 227;
    argb.a = 255;

    uchar4 out = convert_uchar4(clamp(argb, 0, 255));
    return out;
}
