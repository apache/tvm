// tvm target: c -keys=cpu -link-params=1
#define TVM_EXPORTS
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
#include <math.h>
void* __tvm_module_ctx = NULL;

#ifdef __cplusplus
extern "C" {
#endif
static const double __tvm_param__p1[81] = {
    -0x1.11eaa9e4d9b01p+1023, 0x1.1810ae97e4b43p+1021, -0x1.9fdc65fb7041dp+1023, 0x1.482627f5d6e4ap+1023, 
    -0x1.0772953561f07p+1020, 0x1.48a84e6cecff5p+1023, -0x1.26e28d17f635fp+1019, 0x1.7768fb3cff3c2p+1023, 
    -0x1.3c2fe44e75b9ap+1023, 0x1.44fb53557d3efp+1019, -0x1.5ee5fc8e28f1fp+1019, 0x1.0eb0a8c81c2a7p+1021, 
    -0x1.751ab22340c33p+1023, 0x1.d2c2afd7bd2a3p+1021, -0x1.be8801836629fp+1020, 0x1.f04f446513665p+1022, 
    -0x1.352ef6c4f0945p+1023, 0x1.5604c1ff524cap+1023, -0x1.3a5c224658ffep+1023, 0x1.44ac607a3d647p+1020, 
    -0x1.b882a496cdd23p+1021, 0x1.c0c6f66b80247p+1023, -0x1.ae594846fb8dep+1023, 0x1.596ed369fe11bp+1023, 
    -0x1.963dc3980f599p+1022, 0x1.d00f9052bfd4ap+1023, -0x1.6c00017c39815p+1022, 0x1.e9cce03ca9067p+1022, 
    -0x1.ba66095631057p+1022, 0x1.cd60aa80b3167p+1022, -0x1.9c69db1dec421p+1022, 0x1.52b9a71c13bb6p+1023, 
    -0x1.0df52422dae5bp+1022, 0x1.5ce9afae3d1dp+1023, -0x1.00b94a2dbe44bp+1023, 0x1.ec10a44afe9a7p+1021, 
    -0x1.cd70a5e91633ep+1023, 0x1.490108a32c333p+1022, -0x1.d09fd23a2744bp+1023, 0x1.cc8db1ba019e9p+1022, 
    -0x1.91ce17c01a327p+1022, 0x1.3e579d5d1070bp+1022, -0x1.23f0631410215p+1023, 0x1.d5f35bfe175efp+1023, 
    -0x1.571752d7f1f98p+1023, 0x1.01c8bd32f5264p+1023, -0x1.9d526c6f2eb1fp+1018, 0x1.6db67c710f181p+1022, 
    -0x1.79bfcc89df706p+1023, 0x1.81462f31a7f53p+1022, -0x1.e3e4f58fb370dp+1023, 0x1.36cf2d16c35a8p+1023, 
    -0x1.d7633decbb7e9p+1023, 0x1.9bfc0d68f0c77p+1020, -0x1.2ab53ffafc01p+1023, 0x1.620951c440317p+1023, 
    -0x1.549904988fcaep+1023, 0x1.6f7b8dd75aeb7p+1023, -0x1.1eedf60463e45p+1022, 0x1.45fa6ed55ef4fp+1020, 
    -0x1.d55a8ffb931efp+1021, 0x1.603eb928cf331p+1022, -0x1.f0122563885dep+1023, 0x1.3e048e5d5ae8cp+1023, 
    -0x1.aea9dcde66ee6p+1023, 0x1.8cfffa466229ap+1023, -0x1.81c661c48b6c3p+1021, 0x1.5313d6955731dp+1023, 
    -0x1.78dc9c04a57b1p+1023, 0x1.8692f83761623p+1021, -0x1.ae67377e5a8d3p+1021, 0x1.8b025daa1d0cfp+1023, 
    -0x1.51b58718ddaeap+1023, 0x1.dcece6dd29963p+1023, -0x1.d86ef9cfb2dcbp+1022, 0x1.4ec982c572341p+1022, 
    -0x1.4fcb41ca2c159p+1022, 0x1.43fb3809efdc5p+1022, -0x1.1bb063e8c0906p+1023, 0x1.3dbf5a1b6dc8fp+1023, 
    -0x1.f2e148163392p+1023
};
#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __cplusplus
extern "C" {
#endif
static const double __tvm_param__p0[81] = {
    -0x1.4dd47c51b72c8p+1023, 0x1.d9786a2b4d522p+1023, -0x1.fa07659bbc332p+1023, 0x1.6a62d30e83376p+1023, 
    -0x1.61089010741afp+1023, 0x1.656acf3ee1f7fp+1018, -0x1.2db66cc2a2f89p+1022, 0x1.e5cff3f6e202fp+1021, 
    -0x1.fd1a7adf445e9p+1022, 0x1.e260ae3916fb7p+1020, -0x1.1733d4372196cp+1023, 0x1.1d3650da0488dp+1023, 
    -0x1.9429cbc9a0347p+1020, 0x1.fc2c4446caedfp+1022, -0x1.06ad552e50b7bp+1022, 0x1.a0d419cf3706fp+1021, 
    -0x1.c8d619d83c835p+1022, 0x1.3dee9d58c177dp+1023, -0x1.bd6154f334a5cp+1023, 0x1.6146fc6d83fbfp+1023, 
    -0x1.cee100d78ceebp+1022, 0x1.156c5c9703e2ep+1023, -0x1.c0d33df619823p+1023, 0x1.e778228bfda2dp+1023, 
    -0x1.b71d84a1818fdp+1022, 0x1.aadb7c9781ddfp+1019, -0x1.257d2093da1f4p+1023, 0x1.3bc9cabe81393p+1023, 
    -0x1.1ef74a333e921p+1023, 0x1.729faa7d44b87p+1023, -0x1.938c107565401p+1022, 0x1.61fe4a0fd703fp+1023, 
    -0x1.d3f042f02589bp+1021, 0x1.394c485132ed6p+1023, -0x1.9c0dc79f151b1p+1022, 0x1.d08cd5c2f2f14p+1023, 
    -0x1.7e0dd67569d46p+1023, 0x1.f40c7704a373fp+1019, -0x1.c65642f30faf3p+1022, 0x1.673c607f8063p+1023, 
    -0x1.ed604120af0bfp+1020, 0x1.e52bb5083a4f8p+1023, -0x1.dd0485959554ap+1023, 0x1.2073becbe0e7fp+1018, 
    -0x1.93c95a54c07bdp+1022, 0x1.b338edb0a925dp+1022, -0x1.8af01f1153728p+1023, 0x1.5690181b1cddfp+1021, 
    -0x1.4211eaa65bf6fp+1019, 0x1.2fd8941825303p+1022, -0x1.1d8bf2b895a22p+1023, 0x1.ace68f96b68f3p+1022, 
    -0x1.6b2ad59d1d04dp+1023, 0x1.16479152b0a65p+1022, -0x1.74b3a2915b9cep+1023, 0x1.289a824cb08dfp+1021, 
    -0x1.71a2e9804bdfdp+1023, 0x1.e3b2df87e1868p+1023, -0x1.f5561ee543c3ep+1023, 0x1.c2da5999d986fp+1019, 
    -0x1.09513fd29ecafp+1019, 0x1.cf733cf0fb9bap+1023, -0x1.7cb5b76cfc722p+1023, 0x1.761f94b3f9df7p+1022, 
    -0x1.5ab3ed78fd4d3p+1023, 0x1.c74b6fe00f673p+1022, -0x1.5ccc5ce2a41fap+1023, 0x1.ad2225037bfefp+1021, 
    -0x1.b51724e855eebp+1021, 0x1.e56f375dda8bfp+1017, -0x1.199148aafc6c7p+1022, 0x1.bc4e7440f731ep+1023, 
    -0x1.7e43d1934185dp+1023, 0x1.ffd35b2d4e39fp+1021, -0x1.0454b56089466p+1023, 0x1.85d74b4c7757fp+1018, 
    -0x1.c8c44749af163p+1021, 0x1.d836a0252c82cp+1023, -0x1.8d503b03a94d9p+1022, 0x1.35c61ffc07cefp+1023, 
    -0x1.5912c93559c8ep+1023
};
#ifdef __cplusplus
}  // extern "C"
#endif
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_contrib_conv2d_NCHWc_154(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* conv2d_NCHWc = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  for (int32_t n_oc_chunk_fused_oh_fused = 0; n_oc_chunk_fused_oh_fused < 12; ++n_oc_chunk_fused_oh_fused) {
    double conv2d_NCHWc_global[36];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 3; ++oc_block_c_init) {
      conv2d_NCHWc_global[(oc_block_c_init)] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 3; ++oc_block_c_init1) {
      conv2d_NCHWc_global[((oc_block_c_init1 + 3))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 3; ++oc_block_c_init2) {
      conv2d_NCHWc_global[((oc_block_c_init2 + 6))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 3; ++oc_block_c_init3) {
      conv2d_NCHWc_global[((oc_block_c_init3 + 9))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 3; ++oc_block_c_init4) {
      conv2d_NCHWc_global[((oc_block_c_init4 + 12))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 3; ++oc_block_c_init5) {
      conv2d_NCHWc_global[((oc_block_c_init5 + 15))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 3; ++oc_block_c_init6) {
      conv2d_NCHWc_global[((oc_block_c_init6 + 18))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 3; ++oc_block_c_init7) {
      conv2d_NCHWc_global[((oc_block_c_init7 + 21))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init8 = 0; oc_block_c_init8 < 3; ++oc_block_c_init8) {
      conv2d_NCHWc_global[((oc_block_c_init8 + 24))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init9 = 0; oc_block_c_init9 < 3; ++oc_block_c_init9) {
      conv2d_NCHWc_global[((oc_block_c_init9 + 27))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init10 = 0; oc_block_c_init10 < 3; ++oc_block_c_init10) {
      conv2d_NCHWc_global[((oc_block_c_init10 + 30))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init11 = 0; oc_block_c_init11 < 3; ++oc_block_c_init11) {
      conv2d_NCHWc_global[((oc_block_c_init11 + 33))] = 0.000000e+00;
    }
    for (int32_t kh = 0; kh < 3; ++kh) {
      for (int32_t kw = 0; kw < 3; ++kw) {
        for (int32_t ic_inner = 0; ic_inner < 3; ++ic_inner) {
          for (int32_t oc_block_c = 0; oc_block_c < 3; ++oc_block_c) {
            conv2d_NCHWc_global[(oc_block_c)] = (conv2d_NCHWc_global[(oc_block_c)] + (((double*)placeholder)[(((((kh * 42) + (n_oc_chunk_fused_oh_fused * 42)) + (kw * 3)) + ic_inner))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c))]));
          }
          for (int32_t oc_block_c1 = 0; oc_block_c1 < 3; ++oc_block_c1) {
            conv2d_NCHWc_global[((oc_block_c1 + 3))] = (conv2d_NCHWc_global[((oc_block_c1 + 3))] + (((double*)placeholder)[((((((kh * 42) + (n_oc_chunk_fused_oh_fused * 42)) + (kw * 3)) + ic_inner) + 3))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c1))]));
          }
          for (int32_t oc_block_c2 = 0; oc_block_c2 < 3; ++oc_block_c2) {
            conv2d_NCHWc_global[((oc_block_c2 + 6))] = (conv2d_NCHWc_global[((oc_block_c2 + 6))] + (((double*)placeholder)[((((((kh * 42) + (n_oc_chunk_fused_oh_fused * 42)) + (kw * 3)) + ic_inner) + 6))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c2))]));
          }
          for (int32_t oc_block_c3 = 0; oc_block_c3 < 3; ++oc_block_c3) {
            conv2d_NCHWc_global[((oc_block_c3 + 9))] = (conv2d_NCHWc_global[((oc_block_c3 + 9))] + (((double*)placeholder)[((((((kh * 42) + (n_oc_chunk_fused_oh_fused * 42)) + (kw * 3)) + ic_inner) + 9))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c3))]));
          }
          for (int32_t oc_block_c4 = 0; oc_block_c4 < 3; ++oc_block_c4) {
            conv2d_NCHWc_global[((oc_block_c4 + 12))] = (conv2d_NCHWc_global[((oc_block_c4 + 12))] + (((double*)placeholder)[((((((kh * 42) + (n_oc_chunk_fused_oh_fused * 42)) + (kw * 3)) + ic_inner) + 12))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c4))]));
          }
          for (int32_t oc_block_c5 = 0; oc_block_c5 < 3; ++oc_block_c5) {
            conv2d_NCHWc_global[((oc_block_c5 + 15))] = (conv2d_NCHWc_global[((oc_block_c5 + 15))] + (((double*)placeholder)[((((((kh * 42) + (n_oc_chunk_fused_oh_fused * 42)) + (kw * 3)) + ic_inner) + 15))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c5))]));
          }
          for (int32_t oc_block_c6 = 0; oc_block_c6 < 3; ++oc_block_c6) {
            conv2d_NCHWc_global[((oc_block_c6 + 18))] = (conv2d_NCHWc_global[((oc_block_c6 + 18))] + (((double*)placeholder)[((((((kh * 42) + (n_oc_chunk_fused_oh_fused * 42)) + (kw * 3)) + ic_inner) + 18))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c6))]));
          }
          for (int32_t oc_block_c7 = 0; oc_block_c7 < 3; ++oc_block_c7) {
            conv2d_NCHWc_global[((oc_block_c7 + 21))] = (conv2d_NCHWc_global[((oc_block_c7 + 21))] + (((double*)placeholder)[((((((kh * 42) + (n_oc_chunk_fused_oh_fused * 42)) + (kw * 3)) + ic_inner) + 21))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c7))]));
          }
          for (int32_t oc_block_c8 = 0; oc_block_c8 < 3; ++oc_block_c8) {
            conv2d_NCHWc_global[((oc_block_c8 + 24))] = (conv2d_NCHWc_global[((oc_block_c8 + 24))] + (((double*)placeholder)[((((((kh * 42) + (n_oc_chunk_fused_oh_fused * 42)) + (kw * 3)) + ic_inner) + 24))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c8))]));
          }
          for (int32_t oc_block_c9 = 0; oc_block_c9 < 3; ++oc_block_c9) {
            conv2d_NCHWc_global[((oc_block_c9 + 27))] = (conv2d_NCHWc_global[((oc_block_c9 + 27))] + (((double*)placeholder)[((((((kh * 42) + (n_oc_chunk_fused_oh_fused * 42)) + (kw * 3)) + ic_inner) + 27))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c9))]));
          }
          for (int32_t oc_block_c10 = 0; oc_block_c10 < 3; ++oc_block_c10) {
            conv2d_NCHWc_global[((oc_block_c10 + 30))] = (conv2d_NCHWc_global[((oc_block_c10 + 30))] + (((double*)placeholder)[((((((kh * 42) + (n_oc_chunk_fused_oh_fused * 42)) + (kw * 3)) + ic_inner) + 30))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c10))]));
          }
          for (int32_t oc_block_c11 = 0; oc_block_c11 < 3; ++oc_block_c11) {
            conv2d_NCHWc_global[((oc_block_c11 + 33))] = (conv2d_NCHWc_global[((oc_block_c11 + 33))] + (((double*)placeholder)[((((((kh * 42) + (n_oc_chunk_fused_oh_fused * 42)) + (kw * 3)) + ic_inner) + 33))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c11))]));
          }
        }
      }
    }
    for (int32_t ow_inner = 0; ow_inner < 12; ++ow_inner) {
      for (int32_t oc_block = 0; oc_block < 3; ++oc_block) {
        ((double*)conv2d_NCHWc)[((((n_oc_chunk_fused_oh_fused * 36) + (ow_inner * 3)) + oc_block))] = conv2d_NCHWc_global[(((ow_inner * 3) + oc_block))];
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_contrib_conv2d_NCHWc_153(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* conv2d_NCHWc = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  for (int32_t n_oc_chunk_fused_oh_fused = 0; n_oc_chunk_fused_oh_fused < 10; ++n_oc_chunk_fused_oh_fused) {
    double conv2d_NCHWc_global[30];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 3; ++oc_block_c_init) {
      conv2d_NCHWc_global[(oc_block_c_init)] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 3; ++oc_block_c_init1) {
      conv2d_NCHWc_global[((oc_block_c_init1 + 3))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 3; ++oc_block_c_init2) {
      conv2d_NCHWc_global[((oc_block_c_init2 + 6))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 3; ++oc_block_c_init3) {
      conv2d_NCHWc_global[((oc_block_c_init3 + 9))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 3; ++oc_block_c_init4) {
      conv2d_NCHWc_global[((oc_block_c_init4 + 12))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 3; ++oc_block_c_init5) {
      conv2d_NCHWc_global[((oc_block_c_init5 + 15))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 3; ++oc_block_c_init6) {
      conv2d_NCHWc_global[((oc_block_c_init6 + 18))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 3; ++oc_block_c_init7) {
      conv2d_NCHWc_global[((oc_block_c_init7 + 21))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init8 = 0; oc_block_c_init8 < 3; ++oc_block_c_init8) {
      conv2d_NCHWc_global[((oc_block_c_init8 + 24))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init9 = 0; oc_block_c_init9 < 3; ++oc_block_c_init9) {
      conv2d_NCHWc_global[((oc_block_c_init9 + 27))] = 0.000000e+00;
    }
    for (int32_t kh = 0; kh < 3; ++kh) {
      for (int32_t kw = 0; kw < 3; ++kw) {
        for (int32_t ic_inner = 0; ic_inner < 3; ++ic_inner) {
          for (int32_t oc_block_c = 0; oc_block_c < 3; ++oc_block_c) {
            conv2d_NCHWc_global[(oc_block_c)] = (conv2d_NCHWc_global[(oc_block_c)] + (((double*)placeholder)[(((((kh * 36) + (n_oc_chunk_fused_oh_fused * 36)) + (kw * 3)) + ic_inner))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c))]));
          }
          for (int32_t oc_block_c1 = 0; oc_block_c1 < 3; ++oc_block_c1) {
            conv2d_NCHWc_global[((oc_block_c1 + 3))] = (conv2d_NCHWc_global[((oc_block_c1 + 3))] + (((double*)placeholder)[((((((kh * 36) + (n_oc_chunk_fused_oh_fused * 36)) + (kw * 3)) + ic_inner) + 3))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c1))]));
          }
          for (int32_t oc_block_c2 = 0; oc_block_c2 < 3; ++oc_block_c2) {
            conv2d_NCHWc_global[((oc_block_c2 + 6))] = (conv2d_NCHWc_global[((oc_block_c2 + 6))] + (((double*)placeholder)[((((((kh * 36) + (n_oc_chunk_fused_oh_fused * 36)) + (kw * 3)) + ic_inner) + 6))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c2))]));
          }
          for (int32_t oc_block_c3 = 0; oc_block_c3 < 3; ++oc_block_c3) {
            conv2d_NCHWc_global[((oc_block_c3 + 9))] = (conv2d_NCHWc_global[((oc_block_c3 + 9))] + (((double*)placeholder)[((((((kh * 36) + (n_oc_chunk_fused_oh_fused * 36)) + (kw * 3)) + ic_inner) + 9))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c3))]));
          }
          for (int32_t oc_block_c4 = 0; oc_block_c4 < 3; ++oc_block_c4) {
            conv2d_NCHWc_global[((oc_block_c4 + 12))] = (conv2d_NCHWc_global[((oc_block_c4 + 12))] + (((double*)placeholder)[((((((kh * 36) + (n_oc_chunk_fused_oh_fused * 36)) + (kw * 3)) + ic_inner) + 12))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c4))]));
          }
          for (int32_t oc_block_c5 = 0; oc_block_c5 < 3; ++oc_block_c5) {
            conv2d_NCHWc_global[((oc_block_c5 + 15))] = (conv2d_NCHWc_global[((oc_block_c5 + 15))] + (((double*)placeholder)[((((((kh * 36) + (n_oc_chunk_fused_oh_fused * 36)) + (kw * 3)) + ic_inner) + 15))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c5))]));
          }
          for (int32_t oc_block_c6 = 0; oc_block_c6 < 3; ++oc_block_c6) {
            conv2d_NCHWc_global[((oc_block_c6 + 18))] = (conv2d_NCHWc_global[((oc_block_c6 + 18))] + (((double*)placeholder)[((((((kh * 36) + (n_oc_chunk_fused_oh_fused * 36)) + (kw * 3)) + ic_inner) + 18))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c6))]));
          }
          for (int32_t oc_block_c7 = 0; oc_block_c7 < 3; ++oc_block_c7) {
            conv2d_NCHWc_global[((oc_block_c7 + 21))] = (conv2d_NCHWc_global[((oc_block_c7 + 21))] + (((double*)placeholder)[((((((kh * 36) + (n_oc_chunk_fused_oh_fused * 36)) + (kw * 3)) + ic_inner) + 21))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c7))]));
          }
          for (int32_t oc_block_c8 = 0; oc_block_c8 < 3; ++oc_block_c8) {
            conv2d_NCHWc_global[((oc_block_c8 + 24))] = (conv2d_NCHWc_global[((oc_block_c8 + 24))] + (((double*)placeholder)[((((((kh * 36) + (n_oc_chunk_fused_oh_fused * 36)) + (kw * 3)) + ic_inner) + 24))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c8))]));
          }
          for (int32_t oc_block_c9 = 0; oc_block_c9 < 3; ++oc_block_c9) {
            conv2d_NCHWc_global[((oc_block_c9 + 27))] = (conv2d_NCHWc_global[((oc_block_c9 + 27))] + (((double*)placeholder)[((((((kh * 36) + (n_oc_chunk_fused_oh_fused * 36)) + (kw * 3)) + ic_inner) + 27))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c9))]));
          }
        }
      }
    }
    for (int32_t ow_inner = 0; ow_inner < 10; ++ow_inner) {
      for (int32_t oc_block = 0; oc_block < 3; ++oc_block) {
        ((double*)conv2d_NCHWc)[((((n_oc_chunk_fused_oh_fused * 30) + (ow_inner * 3)) + oc_block))] = conv2d_NCHWc_global[(((ow_inner * 3) + oc_block))];
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_layout_transform_115(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* T_layout_trans = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  for (int32_t ax0_ax1_fused = 0; ax0_ax1_fused < 3; ++ax0_ax1_fused) {
    for (int32_t ax2 = 0; ax2 < 8; ++ax2) {
      for (int32_t ax3_inner = 0; ax3_inner < 8; ++ax3_inner) {
        ((double*)T_layout_trans)[((((ax0_ax1_fused * 64) + (ax2 * 8)) + ax3_inner))] = ((double*)placeholder)[((((ax2 * 24) + (ax3_inner * 3)) + ax0_ax1_fused))];
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_contrib_conv2d_NCHWc_152(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* conv2d_NCHWc = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  for (int32_t n_oc_chunk_fused_oh_fused = 0; n_oc_chunk_fused_oh_fused < 8; ++n_oc_chunk_fused_oh_fused) {
    double conv2d_NCHWc_global[24];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 3; ++oc_block_c_init) {
      conv2d_NCHWc_global[(oc_block_c_init)] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 3; ++oc_block_c_init1) {
      conv2d_NCHWc_global[((oc_block_c_init1 + 3))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 3; ++oc_block_c_init2) {
      conv2d_NCHWc_global[((oc_block_c_init2 + 6))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 3; ++oc_block_c_init3) {
      conv2d_NCHWc_global[((oc_block_c_init3 + 9))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 3; ++oc_block_c_init4) {
      conv2d_NCHWc_global[((oc_block_c_init4 + 12))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 3; ++oc_block_c_init5) {
      conv2d_NCHWc_global[((oc_block_c_init5 + 15))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 3; ++oc_block_c_init6) {
      conv2d_NCHWc_global[((oc_block_c_init6 + 18))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 3; ++oc_block_c_init7) {
      conv2d_NCHWc_global[((oc_block_c_init7 + 21))] = 0.000000e+00;
    }
    for (int32_t kh = 0; kh < 3; ++kh) {
      for (int32_t kw = 0; kw < 3; ++kw) {
        for (int32_t ic_inner = 0; ic_inner < 3; ++ic_inner) {
          for (int32_t oc_block_c = 0; oc_block_c < 3; ++oc_block_c) {
            conv2d_NCHWc_global[(oc_block_c)] = (conv2d_NCHWc_global[(oc_block_c)] + (((double*)placeholder)[(((((kh * 30) + (n_oc_chunk_fused_oh_fused * 30)) + (kw * 3)) + ic_inner))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c))]));
          }
          for (int32_t oc_block_c1 = 0; oc_block_c1 < 3; ++oc_block_c1) {
            conv2d_NCHWc_global[((oc_block_c1 + 3))] = (conv2d_NCHWc_global[((oc_block_c1 + 3))] + (((double*)placeholder)[((((((kh * 30) + (n_oc_chunk_fused_oh_fused * 30)) + (kw * 3)) + ic_inner) + 3))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c1))]));
          }
          for (int32_t oc_block_c2 = 0; oc_block_c2 < 3; ++oc_block_c2) {
            conv2d_NCHWc_global[((oc_block_c2 + 6))] = (conv2d_NCHWc_global[((oc_block_c2 + 6))] + (((double*)placeholder)[((((((kh * 30) + (n_oc_chunk_fused_oh_fused * 30)) + (kw * 3)) + ic_inner) + 6))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c2))]));
          }
          for (int32_t oc_block_c3 = 0; oc_block_c3 < 3; ++oc_block_c3) {
            conv2d_NCHWc_global[((oc_block_c3 + 9))] = (conv2d_NCHWc_global[((oc_block_c3 + 9))] + (((double*)placeholder)[((((((kh * 30) + (n_oc_chunk_fused_oh_fused * 30)) + (kw * 3)) + ic_inner) + 9))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c3))]));
          }
          for (int32_t oc_block_c4 = 0; oc_block_c4 < 3; ++oc_block_c4) {
            conv2d_NCHWc_global[((oc_block_c4 + 12))] = (conv2d_NCHWc_global[((oc_block_c4 + 12))] + (((double*)placeholder)[((((((kh * 30) + (n_oc_chunk_fused_oh_fused * 30)) + (kw * 3)) + ic_inner) + 12))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c4))]));
          }
          for (int32_t oc_block_c5 = 0; oc_block_c5 < 3; ++oc_block_c5) {
            conv2d_NCHWc_global[((oc_block_c5 + 15))] = (conv2d_NCHWc_global[((oc_block_c5 + 15))] + (((double*)placeholder)[((((((kh * 30) + (n_oc_chunk_fused_oh_fused * 30)) + (kw * 3)) + ic_inner) + 15))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c5))]));
          }
          for (int32_t oc_block_c6 = 0; oc_block_c6 < 3; ++oc_block_c6) {
            conv2d_NCHWc_global[((oc_block_c6 + 18))] = (conv2d_NCHWc_global[((oc_block_c6 + 18))] + (((double*)placeholder)[((((((kh * 30) + (n_oc_chunk_fused_oh_fused * 30)) + (kw * 3)) + ic_inner) + 18))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c6))]));
          }
          for (int32_t oc_block_c7 = 0; oc_block_c7 < 3; ++oc_block_c7) {
            conv2d_NCHWc_global[((oc_block_c7 + 21))] = (conv2d_NCHWc_global[((oc_block_c7 + 21))] + (((double*)placeholder)[((((((kh * 30) + (n_oc_chunk_fused_oh_fused * 30)) + (kw * 3)) + ic_inner) + 21))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c7))]));
          }
        }
      }
    }
    for (int32_t ow_inner = 0; ow_inner < 8; ++ow_inner) {
      for (int32_t oc_block = 0; oc_block < 3; ++oc_block) {
        ((double*)conv2d_NCHWc)[((((n_oc_chunk_fused_oh_fused * 24) + (ow_inner * 3)) + oc_block))] = conv2d_NCHWc_global[(((ow_inner * 3) + oc_block))];
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_layout_transform_116(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* T_layout_trans = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  for (int32_t ax0_ax1_fused_ax2_fused = 0; ax0_ax1_fused_ax2_fused < 16; ++ax0_ax1_fused_ax2_fused) {
    for (int32_t ax3 = 0; ax3 < 16; ++ax3) {
      for (int32_t ax4_inner = 0; ax4_inner < 3; ++ax4_inner) {
        ((double*)T_layout_trans)[((((ax0_ax1_fused_ax2_fused * 48) + (ax3 * 3)) + ax4_inner))] = ((double*)placeholder)[((((ax4_inner * 256) + (ax0_ax1_fused_ax2_fused * 16)) + ax3))];
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fused_nn_contrib_conv2d_NCHWc_155(void* args, void* arg_type_ids, int32_t num_args, void* out_ret_value, void* out_ret_tcode, void* resource_handle) {
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = ((int32_t*)arg_type_ids)[(0)];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = ((int32_t*)arg_type_ids)[(1)];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = ((int32_t*)arg_type_ids)[(2)];
  void* placeholder = (((DLTensor*)arg0)[0].data);
  void* arg0_shape = (((DLTensor*)arg0)[0].shape);
  void* arg0_strides = (((DLTensor*)arg0)[0].strides);
  int32_t dev_id = (((DLTensor*)arg0)[0].device.device_id);
  void* placeholder1 = (((DLTensor*)arg1)[0].data);
  void* arg1_shape = (((DLTensor*)arg1)[0].shape);
  void* arg1_strides = (((DLTensor*)arg1)[0].strides);
  void* conv2d_NCHWc = (((DLTensor*)arg2)[0].data);
  void* arg2_shape = (((DLTensor*)arg2)[0].shape);
  void* arg2_strides = (((DLTensor*)arg2)[0].strides);
  if (!(arg0_strides == NULL)) {
  }
  if (!(arg1_strides == NULL)) {
  }
  if (!(arg2_strides == NULL)) {
  }
  for (int32_t n_oc_chunk_fused_oh_fused = 0; n_oc_chunk_fused_oh_fused < 14; ++n_oc_chunk_fused_oh_fused) {
    double conv2d_NCHWc_global[42];
    for (int32_t oc_block_c_init = 0; oc_block_c_init < 3; ++oc_block_c_init) {
      conv2d_NCHWc_global[(oc_block_c_init)] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init1 = 0; oc_block_c_init1 < 3; ++oc_block_c_init1) {
      conv2d_NCHWc_global[((oc_block_c_init1 + 3))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init2 = 0; oc_block_c_init2 < 3; ++oc_block_c_init2) {
      conv2d_NCHWc_global[((oc_block_c_init2 + 6))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init3 = 0; oc_block_c_init3 < 3; ++oc_block_c_init3) {
      conv2d_NCHWc_global[((oc_block_c_init3 + 9))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init4 = 0; oc_block_c_init4 < 3; ++oc_block_c_init4) {
      conv2d_NCHWc_global[((oc_block_c_init4 + 12))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init5 = 0; oc_block_c_init5 < 3; ++oc_block_c_init5) {
      conv2d_NCHWc_global[((oc_block_c_init5 + 15))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init6 = 0; oc_block_c_init6 < 3; ++oc_block_c_init6) {
      conv2d_NCHWc_global[((oc_block_c_init6 + 18))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init7 = 0; oc_block_c_init7 < 3; ++oc_block_c_init7) {
      conv2d_NCHWc_global[((oc_block_c_init7 + 21))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init8 = 0; oc_block_c_init8 < 3; ++oc_block_c_init8) {
      conv2d_NCHWc_global[((oc_block_c_init8 + 24))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init9 = 0; oc_block_c_init9 < 3; ++oc_block_c_init9) {
      conv2d_NCHWc_global[((oc_block_c_init9 + 27))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init10 = 0; oc_block_c_init10 < 3; ++oc_block_c_init10) {
      conv2d_NCHWc_global[((oc_block_c_init10 + 30))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init11 = 0; oc_block_c_init11 < 3; ++oc_block_c_init11) {
      conv2d_NCHWc_global[((oc_block_c_init11 + 33))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init12 = 0; oc_block_c_init12 < 3; ++oc_block_c_init12) {
      conv2d_NCHWc_global[((oc_block_c_init12 + 36))] = 0.000000e+00;
    }
    for (int32_t oc_block_c_init13 = 0; oc_block_c_init13 < 3; ++oc_block_c_init13) {
      conv2d_NCHWc_global[((oc_block_c_init13 + 39))] = 0.000000e+00;
    }
    for (int32_t kh = 0; kh < 3; ++kh) {
      for (int32_t kw = 0; kw < 3; ++kw) {
        for (int32_t ic_inner = 0; ic_inner < 3; ++ic_inner) {
          for (int32_t oc_block_c = 0; oc_block_c < 3; ++oc_block_c) {
            conv2d_NCHWc_global[(oc_block_c)] = (conv2d_NCHWc_global[(oc_block_c)] + (((double*)placeholder)[(((((kh * 48) + (n_oc_chunk_fused_oh_fused * 48)) + (kw * 3)) + ic_inner))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c))]));
          }
          for (int32_t oc_block_c1 = 0; oc_block_c1 < 3; ++oc_block_c1) {
            conv2d_NCHWc_global[((oc_block_c1 + 3))] = (conv2d_NCHWc_global[((oc_block_c1 + 3))] + (((double*)placeholder)[((((((kh * 48) + (n_oc_chunk_fused_oh_fused * 48)) + (kw * 3)) + ic_inner) + 3))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c1))]));
          }
          for (int32_t oc_block_c2 = 0; oc_block_c2 < 3; ++oc_block_c2) {
            conv2d_NCHWc_global[((oc_block_c2 + 6))] = (conv2d_NCHWc_global[((oc_block_c2 + 6))] + (((double*)placeholder)[((((((kh * 48) + (n_oc_chunk_fused_oh_fused * 48)) + (kw * 3)) + ic_inner) + 6))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c2))]));
          }
          for (int32_t oc_block_c3 = 0; oc_block_c3 < 3; ++oc_block_c3) {
            conv2d_NCHWc_global[((oc_block_c3 + 9))] = (conv2d_NCHWc_global[((oc_block_c3 + 9))] + (((double*)placeholder)[((((((kh * 48) + (n_oc_chunk_fused_oh_fused * 48)) + (kw * 3)) + ic_inner) + 9))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c3))]));
          }
          for (int32_t oc_block_c4 = 0; oc_block_c4 < 3; ++oc_block_c4) {
            conv2d_NCHWc_global[((oc_block_c4 + 12))] = (conv2d_NCHWc_global[((oc_block_c4 + 12))] + (((double*)placeholder)[((((((kh * 48) + (n_oc_chunk_fused_oh_fused * 48)) + (kw * 3)) + ic_inner) + 12))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c4))]));
          }
          for (int32_t oc_block_c5 = 0; oc_block_c5 < 3; ++oc_block_c5) {
            conv2d_NCHWc_global[((oc_block_c5 + 15))] = (conv2d_NCHWc_global[((oc_block_c5 + 15))] + (((double*)placeholder)[((((((kh * 48) + (n_oc_chunk_fused_oh_fused * 48)) + (kw * 3)) + ic_inner) + 15))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c5))]));
          }
          for (int32_t oc_block_c6 = 0; oc_block_c6 < 3; ++oc_block_c6) {
            conv2d_NCHWc_global[((oc_block_c6 + 18))] = (conv2d_NCHWc_global[((oc_block_c6 + 18))] + (((double*)placeholder)[((((((kh * 48) + (n_oc_chunk_fused_oh_fused * 48)) + (kw * 3)) + ic_inner) + 18))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c6))]));
          }
          for (int32_t oc_block_c7 = 0; oc_block_c7 < 3; ++oc_block_c7) {
            conv2d_NCHWc_global[((oc_block_c7 + 21))] = (conv2d_NCHWc_global[((oc_block_c7 + 21))] + (((double*)placeholder)[((((((kh * 48) + (n_oc_chunk_fused_oh_fused * 48)) + (kw * 3)) + ic_inner) + 21))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c7))]));
          }
          for (int32_t oc_block_c8 = 0; oc_block_c8 < 3; ++oc_block_c8) {
            conv2d_NCHWc_global[((oc_block_c8 + 24))] = (conv2d_NCHWc_global[((oc_block_c8 + 24))] + (((double*)placeholder)[((((((kh * 48) + (n_oc_chunk_fused_oh_fused * 48)) + (kw * 3)) + ic_inner) + 24))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c8))]));
          }
          for (int32_t oc_block_c9 = 0; oc_block_c9 < 3; ++oc_block_c9) {
            conv2d_NCHWc_global[((oc_block_c9 + 27))] = (conv2d_NCHWc_global[((oc_block_c9 + 27))] + (((double*)placeholder)[((((((kh * 48) + (n_oc_chunk_fused_oh_fused * 48)) + (kw * 3)) + ic_inner) + 27))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c9))]));
          }
          for (int32_t oc_block_c10 = 0; oc_block_c10 < 3; ++oc_block_c10) {
            conv2d_NCHWc_global[((oc_block_c10 + 30))] = (conv2d_NCHWc_global[((oc_block_c10 + 30))] + (((double*)placeholder)[((((((kh * 48) + (n_oc_chunk_fused_oh_fused * 48)) + (kw * 3)) + ic_inner) + 30))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c10))]));
          }
          for (int32_t oc_block_c11 = 0; oc_block_c11 < 3; ++oc_block_c11) {
            conv2d_NCHWc_global[((oc_block_c11 + 33))] = (conv2d_NCHWc_global[((oc_block_c11 + 33))] + (((double*)placeholder)[((((((kh * 48) + (n_oc_chunk_fused_oh_fused * 48)) + (kw * 3)) + ic_inner) + 33))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c11))]));
          }
          for (int32_t oc_block_c12 = 0; oc_block_c12 < 3; ++oc_block_c12) {
            conv2d_NCHWc_global[((oc_block_c12 + 36))] = (conv2d_NCHWc_global[((oc_block_c12 + 36))] + (((double*)placeholder)[((((((kh * 48) + (n_oc_chunk_fused_oh_fused * 48)) + (kw * 3)) + ic_inner) + 36))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c12))]));
          }
          for (int32_t oc_block_c13 = 0; oc_block_c13 < 3; ++oc_block_c13) {
            conv2d_NCHWc_global[((oc_block_c13 + 39))] = (conv2d_NCHWc_global[((oc_block_c13 + 39))] + (((double*)placeholder)[((((((kh * 48) + (n_oc_chunk_fused_oh_fused * 48)) + (kw * 3)) + ic_inner) + 39))] * ((double*)placeholder1)[(((((kh * 27) + (kw * 9)) + (ic_inner * 3)) + oc_block_c13))]));
          }
        }
      }
    }
    for (int32_t ow_inner = 0; ow_inner < 14; ++ow_inner) {
      for (int32_t oc_block = 0; oc_block < 3; ++oc_block) {
        ((double*)conv2d_NCHWc)[((((n_oc_chunk_fused_oh_fused * 42) + (ow_inner * 3)) + oc_block))] = conv2d_NCHWc_global[(((ow_inner * 3) + oc_block))];
      }
    }
  }
  return 0;
}

#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t _lookup_linked_param(void* args, int* arg_type_ids, int num_args, void* out_ret_value, int* out_ret_tcode, void* resource_handle) {
    switch (((int64_t*) args)[0]) {
    default:
        out_ret_tcode[0] = 4;
        return 0;
    case 4:
        ((uint64_t*)out_ret_value)[0] = (uint64_t) (uintptr_t) __tvm_param__p1;
        out_ret_tcode[0] = 3;
        return 0;
    case 2:
        ((uint64_t*)out_ret_value)[0] = (uint64_t) (uintptr_t) __tvm_param__p0;
        out_ret_tcode[0] = 3;
        return 0;
    }
}
