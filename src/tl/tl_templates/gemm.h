#pragma once
#if (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 900))
#include "gemm_sm90.h"
#elif (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 750))
#include "gemm_sm80.h"
#elif (defined(__CUDA_ARCH_LIST__) && (__CUDA_ARCH_LIST__ >= 700))
#include "gemm_sm70.h"
#else


#if defined(__gfx1100__)
#include "gemm_gfx1100.h"
#elif defined(__gfx908__)
#include "gemm_gfx908.h"
#elif defined(__gfx906__)
#include "gemm_gfx906.h"
#endif

#endif
