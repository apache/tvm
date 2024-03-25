#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <stdbool.h>

#include <stdio.h>

#ifdef __cplusplus
extern "C"
#endif


int q_vanilla_accelerator_conv2dnchw(int8_t* q_vanilla_accelerator_0_i0, int8_t* q_vanilla_accelerator_0_i1, int32_t* bias_data, int32_t* compute,
                                      int32_t oc, int32_t iw, int32_t ih, int32_t ic, int32_t kh, int32_t kw, int32_t i_zp, int32_t k_zp) {


  int kw_low = kw / 2;
  int kh_low = kh / 2;
  int kw_high = iw + kw / 2;
  int kh_high = ih + kh / 2;

  int padded_iw = iw + 2 * kw_low;
  int padded_ih = ih + 2 * kh_low;

  int32_t* data_pad_let = (int32_t*)malloc(
      (((ic * padded_iw * padded_ih) + (padded_ih * padded_iw)) + padded_iw) * sizeof(int32_t));

  int32_t* compute_let = (int32_t*)malloc((oc * ic * kh * kw) * sizeof(int32_t));



  for (int32_t i1_1 = 0; i1_1 < ic; ++i1_1) {
    for (int32_t i2_1 = 0; i2_1 < padded_ih; ++i2_1) {
      for (int32_t i3_1 = 0; i3_1 < padded_iw; ++i3_1) {
        data_pad_let[(((i1_1 * padded_iw * padded_ih) + (i2_1 * padded_iw)) + i3_1)] = (((((kh_low <= i2_1) && (i2_1 < kh_high)) && (kw_low <= i3_1)) && (i3_1 < kw_high)) 
        ? ((int32_t)q_vanilla_accelerator_0_i0[(((i1_1 * iw * ih) + ((i2_1 - kh_low) * iw) + i3_1 - kw_low))] - (i_zp)) 
        : 0);
        
      }
    }
  }
  


  for (int32_t i0 = 0; i0 < oc; ++i0) {
    for (int32_t i1_2 = 0; i1_2 < ic; ++i1_2) {
      for (int32_t i2_2 = 0; i2_2 < kh; ++i2_2) {
        for (int32_t i3_2 = 0; i3_2 < kw; ++i3_2) {
          int32_t cse_var_2 = ((((i0 * ic * kh * kw) + (i1_2 * kw * kh)) + (i2_2 * kw)) + i3_2);
          compute_let[cse_var_2] = (((int32_t)q_vanilla_accelerator_0_i1[cse_var_2]) - k_zp);
        }
      }
    }
  }


  for (int32_t oc_ = 0; oc_ < oc; ++oc_) {
    for (int32_t oh = 0; oh < ih; ++oh) {
      for (int32_t ow = 0; ow < iw; ++ow) {
        int32_t cse_var_3 = (((oc_ * ih * iw) + (oh * iw)) + ow);
        for (int32_t ic_ = 0; ic_ < ic; ++ic_) {
          for (int32_t kh_ = 0; kh_ < kh; ++kh_) {
            for (int32_t kw_ = 0; kw_ < kh; ++kw_) {
              if (((ic_ == 0) && (kh_ == 0)) && (kw_ == 0)) {
                compute[cse_var_3] = 0;
              }
              compute[cse_var_3] = (compute[cse_var_3] + ((data_pad_let)[(((((ic_ * padded_iw * padded_ih) + (oh * padded_iw)) + (kh_ * padded_iw)) + ow) + kw_)] * (compute_let)[((((oc_ * ic * kh * kw) + (ic_ * kh * kw)) + (kh_ * kw)) + kw_)]));
            }
          }
        }
        compute[cse_var_3] = compute[cse_var_3] + bias_data[oc_]; //bias_add
      }
    }
  }
  free(data_pad_let);
  free(compute_let);
  return 0;
}

