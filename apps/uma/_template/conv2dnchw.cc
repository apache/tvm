/*
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
*/
#include <stdlib.h>

// TODO(mjklaiber): leverage pragma import_c in the future
#ifdef __cplusplus
extern "C"
#endif

    /*!
     * \brief Conv2D function for mock-accelerator examples. Limited to same-padded Conv2D with
     * stride (1,1) and datatype float. \param ifmap Pointer to input feature map data of size
     * iw*ih*ic*sizeof(float). \param weights Pointer to weight data of size
     * kh*kw*ic**oc*sizeof(float). \param result Pointer to output feature map data of size
     * iw*ih*oc*sizeof(float). \param oc Number of channels of output feature map. \param iw Width
     * of input feature map, ifmap. \param ih Height of input feature map, ifmap. \param ic Number
     * of channels of input feature map. \param kh Height of convolution kernels. \param kw Width of
     * convolution kernels.
     *
     * \return error code
     *
     */
    int
    my_ai_hw_conv2dnchw(float* ifmap, float* weights, float* result, int oc, int iw, int ih, int ic,
                        int kh, int kw) {

  int kw_low = kw / 2;
  int kh_low = kh / 2;
  int kw_high = iw + kw / 2;
  int kh_high = ih + kh / 2;

  int padded_iw = iw + 2 * kw_low;
  int padded_ih = ih + 2 * kh_low;

  // This is only example code. A real hardware accelerator would call a device specific malloc
  // function.
  float* pad_temp = (float*)malloc(
      (((ic * padded_iw * padded_ih) + (padded_ih * padded_iw)) + padded_iw) * sizeof(float));

  if (pad_temp == NULL) {
    return -1;
  }

  for (int i1 = 0; i1 < ic; ++i1) {
    for (int i2 = 0; i2 < padded_ih; ++i2) {
      for (int i3 = 0; i3 < padded_iw; ++i3) {
        ((float*)pad_temp)[(((i1 * padded_iw * padded_ih) + (i2 * padded_iw)) + i3)] =
            (((((kh_low <= i2) && (i2 < kh_high)) && (kw_low <= i3)) && (i3 < kw_high))
                 ? ifmap[((((i1 * iw * ih) + ((i2 - kh_low) * iw)) + i3 - kw_low))]
                 : 0.000000e+00f);
      }
    }
  }
  for (int i11 = 0; i11 < oc; ++i11) {
    for (int i21 = 0; i21 < ih; ++i21) {
      for (int i31 = 0; i31 < iw; ++i31) {
        for (int i4 = 0; i4 < ic; ++i4) {
          for (int i5 = 0; i5 < kh; ++i5) {
            for (int i6 = 0; i6 < kw; ++i6) {
              int cse_var_1 = (((i11 * iw * ih) + (i21 * iw)) + i31);
              if (((i4 == 0) && (i5 == 0)) && (i6 == 0)) {
                result[cse_var_1] = 0.000000e+00f;
              }
              result[cse_var_1] =
                  (result[cse_var_1] +
                   (((float*)
                         pad_temp)[i4 * padded_iw * padded_ih + (i21 + i5) * padded_iw + i31 + i6] *
                    weights[((((i11 * ic * kh * kw) + (i4 * kh * kw)) + (i5 * kw)) + i6)]));
            }
          }
        }
      }
    }
  }

  // This is only example code. A real hardware accelerator would call a device specific free
  // function.
  free(pad_temp);
  return 0;
}
