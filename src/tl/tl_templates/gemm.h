#pragma once
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
#include "gemm_sm90.h"
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 750))
#include "gemm_sm80.h"
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700))
#include "gemm_sm70.h"
#else

#endif
