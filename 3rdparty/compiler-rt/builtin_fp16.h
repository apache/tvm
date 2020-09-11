/*
 * Copyright (c) 2009-2015 by llvm/compiler-rt contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * \file builtin_fp16.h
 * \brief Functions for conversion between fp32 and fp16, adopted from compiler-rt.
 */
#ifndef COMPILER_RT_BUILTIN_FP16_H_
#define COMPILER_RT_BUILTIN_FP16_H_

#ifdef _MSC_VER
#pragma warning(disable : 4305 4805)
#endif

#include <cstdint>

static inline uint32_t __clz(uint32_t x) {
  // count leading zeros
  int n = 32;
  uint32_t y;

  y = x >> 16;
  if (y) {
    n = n - 16;
    x = y;
  }
  y = x >> 8;
  if (y) {
    n = n - 8;
    x = y;
  }
  y = x >> 4;
  if (y) {
    n = n - 4;
    x = y;
  }
  y = x >> 2;
  if (y) {
    n = n - 2;
    x = y;
  }
  y = x >> 1;
  if (y) return n - 2;
  return n - x;
}

template <typename SRC_T, typename SRC_REP_T, int SRC_SIG_BITS, typename DST_T, typename DST_REP_T,
          int DST_SIG_BITS>
static inline DST_T __truncXfYf2__(SRC_T a) {
  // Various constants whose values follow from the type parameters.
  // Any reasonable optimizer will fold and propagate all of these.
  const int srcBits = sizeof(SRC_T) * 8;
  const int srcExpBits = srcBits - SRC_SIG_BITS - 1;
  const int srcInfExp = (1 << srcExpBits) - 1;
  const int srcExpBias = srcInfExp >> 1;

  const SRC_REP_T srcMinNormal = SRC_REP_T(1) << SRC_SIG_BITS;
  const SRC_REP_T srcSignificandMask = srcMinNormal - 1;
  const SRC_REP_T srcInfinity = (SRC_REP_T)srcInfExp << SRC_SIG_BITS;
  const SRC_REP_T srcSignMask = SRC_REP_T(1) << (SRC_SIG_BITS + srcExpBits);
  const SRC_REP_T srcAbsMask = srcSignMask - 1;
  const SRC_REP_T roundMask = (SRC_REP_T(1) << (SRC_SIG_BITS - DST_SIG_BITS)) - 1;
  const SRC_REP_T halfway = SRC_REP_T(1) << (SRC_SIG_BITS - DST_SIG_BITS - 1);
  const SRC_REP_T srcQNaN = SRC_REP_T(1) << (SRC_SIG_BITS - 1);
  const SRC_REP_T srcNaNCode = srcQNaN - 1;

  const int dstBits = sizeof(DST_T) * 8;
  const int dstExpBits = dstBits - DST_SIG_BITS - 1;
  const int dstInfExp = (1 << dstExpBits) - 1;
  const int dstExpBias = dstInfExp >> 1;

  const int underflowExponent = srcExpBias + 1 - dstExpBias;
  const int overflowExponent = srcExpBias + dstInfExp - dstExpBias;
  const SRC_REP_T underflow = (SRC_REP_T)underflowExponent << SRC_SIG_BITS;
  const SRC_REP_T overflow = (SRC_REP_T)overflowExponent << SRC_SIG_BITS;

  const DST_REP_T dstQNaN = DST_REP_T(1) << (DST_SIG_BITS - 1);
  const DST_REP_T dstNaNCode = dstQNaN - 1;

  // Break a into a sign and representation of the absolute value
  union SrcExchangeType {
    SRC_T f;
    SRC_REP_T i;
  };
  SrcExchangeType src_rep;
  src_rep.f = a;
  const SRC_REP_T aRep = src_rep.i;
  const SRC_REP_T aAbs = aRep & srcAbsMask;
  const SRC_REP_T sign = aRep & srcSignMask;
  DST_REP_T absResult;

  if (aAbs - underflow < aAbs - overflow) {
    // The exponent of a is within the range of normal numbers in the
    // destination format.  We can convert by simply right-shifting with
    // rounding and adjusting the exponent.
    absResult = aAbs >> (SRC_SIG_BITS - DST_SIG_BITS);
    absResult -= (DST_REP_T)(srcExpBias - dstExpBias) << DST_SIG_BITS;

    const SRC_REP_T roundBits = aAbs & roundMask;
    // Round to nearest
    if (roundBits > halfway) absResult++;
    // Ties to even
    else if (roundBits == halfway)
      absResult += absResult & 1;
  } else if (aAbs > srcInfinity) {
    // a is NaN.
    // Conjure the result by beginning with infinity, setting the qNaN
    // bit and inserting the (truncated) trailing NaN field.
    absResult = (DST_REP_T)dstInfExp << DST_SIG_BITS;
    absResult |= dstQNaN;
    absResult |= ((aAbs & srcNaNCode) >> (SRC_SIG_BITS - DST_SIG_BITS)) & dstNaNCode;
  } else if (aAbs >= overflow) {
    // a overflows to infinity.
    absResult = (DST_REP_T)dstInfExp << DST_SIG_BITS;
  } else {
    // a underflows on conversion to the destination type or is an exact
    // zero.  The result may be a denormal or zero.  Extract the exponent
    // to get the shift amount for the denormalization.
    const int aExp = aAbs >> SRC_SIG_BITS;
    const int shift = srcExpBias - dstExpBias - aExp + 1;

    const SRC_REP_T significand = (aRep & srcSignificandMask) | srcMinNormal;

    // Right shift by the denormalization amount with sticky.
    if (shift > SRC_SIG_BITS) {
      absResult = 0;
    } else {
      const bool sticky = significand << (srcBits - shift);
      SRC_REP_T denormalizedSignificand = significand >> shift | sticky;
      absResult = denormalizedSignificand >> (SRC_SIG_BITS - DST_SIG_BITS);
      const SRC_REP_T roundBits = denormalizedSignificand & roundMask;
      // Round to nearest
      if (roundBits > halfway) absResult++;
      // Ties to even
      else if (roundBits == halfway)
        absResult += absResult & 1;
    }
  }

  // Apply the signbit to (DST_T)abs(a).
  const DST_REP_T result = absResult | sign >> (srcBits - dstBits);
  union DstExchangeType {
    DST_T f;
    DST_REP_T i;
  };
  DstExchangeType dst_rep;
  dst_rep.i = result;
  return dst_rep.f;
}

template <typename SRC_T, typename SRC_REP_T, int SRC_SIG_BITS, typename DST_T, typename DST_REP_T,
          int DST_SIG_BITS>
static inline DST_T __extendXfYf2__(SRC_T a) {
  // Various constants whose values follow from the type parameters.
  // Any reasonable optimizer will fold and propagate all of these.
  const int srcBits = sizeof(SRC_T) * 8;
  const int srcExpBits = srcBits - SRC_SIG_BITS - 1;
  const int srcInfExp = (1 << srcExpBits) - 1;
  const int srcExpBias = srcInfExp >> 1;

  const SRC_REP_T srcMinNormal = SRC_REP_T(1) << SRC_SIG_BITS;
  const SRC_REP_T srcInfinity = (SRC_REP_T)srcInfExp << SRC_SIG_BITS;
  const SRC_REP_T srcSignMask = SRC_REP_T(1) << (SRC_SIG_BITS + srcExpBits);
  const SRC_REP_T srcAbsMask = srcSignMask - 1;
  const SRC_REP_T srcQNaN = SRC_REP_T(1) << (SRC_SIG_BITS - 1);
  const SRC_REP_T srcNaNCode = srcQNaN - 1;

  const int dstBits = sizeof(DST_T) * 8;
  const int dstExpBits = dstBits - DST_SIG_BITS - 1;
  const int dstInfExp = (1 << dstExpBits) - 1;
  const int dstExpBias = dstInfExp >> 1;

  const DST_REP_T dstMinNormal = DST_REP_T(1) << DST_SIG_BITS;

  // Break a into a sign and representation of the absolute value
  union SrcExchangeType {
    SRC_T f;
    SRC_REP_T i;
  };
  SrcExchangeType src_rep;
  src_rep.f = a;
  const SRC_REP_T aRep = src_rep.i;
  const SRC_REP_T aAbs = aRep & srcAbsMask;
  const SRC_REP_T sign = aRep & srcSignMask;
  DST_REP_T absResult;

  // If sizeof(SRC_REP_T) < sizeof(int), the subtraction result is promoted
  // to (signed) int.  To avoid that, explicitly cast to SRC_REP_T.
  if ((SRC_REP_T)(aAbs - srcMinNormal) < srcInfinity - srcMinNormal) {
    // a is a normal number.
    // Extend to the destination type by shifting the significand and
    // exponent into the proper position and rebiasing the exponent.
    absResult = (DST_REP_T)aAbs << (DST_SIG_BITS - SRC_SIG_BITS);
    absResult += (DST_REP_T)(dstExpBias - srcExpBias) << DST_SIG_BITS;
  }

  else if (aAbs >= srcInfinity) {
    // a is NaN or infinity.
    // Conjure the result by beginning with infinity, then setting the qNaN
    // bit (if needed) and right-aligning the rest of the trailing NaN
    // payload field.
    absResult = (DST_REP_T)dstInfExp << DST_SIG_BITS;
    absResult |= (DST_REP_T)(aAbs & srcQNaN) << (DST_SIG_BITS - SRC_SIG_BITS);
    absResult |= (DST_REP_T)(aAbs & srcNaNCode) << (DST_SIG_BITS - SRC_SIG_BITS);
  } else if (aAbs) {
    // a is denormal.
    // renormalize the significand and clear the leading bit, then insert
    // the correct adjusted exponent in the destination type.
    const int scale = __clz(aAbs) - __clz(srcMinNormal);
    absResult = (DST_REP_T)aAbs << (DST_SIG_BITS - SRC_SIG_BITS + scale);
    absResult ^= dstMinNormal;
    const int resultExponent = dstExpBias - srcExpBias - scale + 1;
    absResult |= (DST_REP_T)resultExponent << DST_SIG_BITS;
  } else {
    // a is zero.
    absResult = 0;
  }

  // Apply the signbit to (DST_T)abs(a).
  const DST_REP_T result = absResult | (DST_REP_T)sign << (dstBits - srcBits);
  union DstExchangeType {
    DST_T f;
    DST_REP_T i;
  };
  DstExchangeType dst_rep;
  dst_rep.i = result;
  return dst_rep.f;
}

#endif  // COMPILER_RT_BUILTIN_FP16_H_
