/*!
 * \brief This is an all in one file for ROCM runtime library.
 *
 * This is used to create a RPC module library that can be
 * safely passed to rocm
 */

#define TVM_ROCM_RUNTIME 1
#define TVM_USE_MIOPEN 1
#define __HIP_PLATFORM_HCC__ 1

#include "../../src/runtime/rocm/rocm_device_api.cc"
#include "../../src/runtime/rocm/rocm_module.cc"
#include "../../src/contrib/miopen/conv_forward.cc"
#include "../../src/contrib/miopen/miopen_utils.cc"
