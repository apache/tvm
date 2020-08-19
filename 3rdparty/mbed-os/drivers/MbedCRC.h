/* mbed Microcontroller Library
 * Copyright (c) 2018 ARM Limited
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MBED_CRC_API_H
#define MBED_CRC_API_H

#ifdef __cplusplus

#include "crc_api.h"

namespace tvm {
namespace runtime {

/** CRC mode selection
 */
enum class CrcMode {
    TABLE,      /// Use table-based computation (if table available), else bitwise
    BITWISE     /// Always use bitwise manual computation
};

#ifndef DOXYGEN_ONLY
namespace impl {
template<uint32_t polynomial, uint8_t width, CrcMode mode>
class MbedCRC;

constexpr bool have_crc_table(uint32_t polynomial, uint8_t width)
{
#if MBED_CRC_TABLE_SIZE > 0
    return (polynomial == POLY_32BIT_ANSI && width == 32) ||
           (polynomial == POLY_16BIT_IBM && width == 16) ||
           (polynomial == POLY_16BIT_CCITT && width == 16) ||
           (polynomial == POLY_8BIT_CCITT && width == 8) ||
           (polynomial == POLY_7BIT_SD && width == 7);
#else
    return false;
#endif
}

constexpr CrcMode choose_crc_mode(uint32_t polynomial, uint8_t width, CrcMode mode_limit)
{
    return
        mode_limit <= CrcMode::TABLE && have_crc_table(polynomial, width) ? CrcMode::TABLE :
        CrcMode::BITWISE;
}
#endif // DOXYGEN_ONLY

} // namespace impl

/** CRC object provides CRC generation through software
 *
 *  CRC sums can be generated using two different methods: software ROM tables
 *  and bitwise computation. The mode used is normally selected automatically based on required
 *  polynomial and hardware capabilities. Any polynomial in standard form (`x^3 + x + 1`)
 *  can be used for computation, but custom ones can affect the performance.
 *
 *  First choice is the ROM polynomial tables (you can find list of supported polynomials here
 *  ::crc_polynomial). If the selected configuration is supported, it will accelerate the software
 *  computations. If ROM tables are not available for the selected polynomial, then CRC is computed
 *  at run time bit by bit for all data input.
 *
 *  If desired, the mode can be manually limited for a given instance by specifying the mode_limit
 *  template parameter. This might be appropriate to ensure a table is not pulled in for a
 *  non-speed-critical CRC, or to avoid the hardware set-up overhead if you know you will be
 *  calling `compute` with very small data sizes.
 *
 *  @note Synchronization level: Thread safe
 *
 *  @tparam  polynomial CRC polynomial value in hex
 *  @tparam  width      CRC polynomial width
 *  @tparam  mode_limit Maximum amount of acceleration to use
 *
 * Example: Compute CRC data
 * @code
 *
 *  #include "mbed.h"
 *
 *  int main() {
 *      MbedCRC<POLY_32BIT_ANSI, 32> ct;
 *
 *      char  test[] = "123456789";
 *      uint32_t crc = 0;
 *
 *      printf("\nPolynomial = 0x%lx  Width = %d \n", ct.get_polynomial(), ct.get_width());
 *
 *      ct.compute((void *)test, strlen((const char*)test), &crc);
 *
 *      printf("The CRC of data \"123456789\" is : 0x%lx\n", crc);
 *      return 0;
 *  }
 * @endcode
 * Example: Compute CRC with data available in parts
 * @code
 *
 *  #include "mbed.h"
 *  int main() {
 *      MbedCRC<POLY_32BIT_ANSI, 32> ct;
 *
 *      char  test[] = "123456789";
 *      uint32_t crc = 0;
 *
 *      printf("\nPolynomial = 0x%lx  Width = %d \n", ct.get_polynomial(), ct.get_width());
 *      ct.compute_partial_start(&crc);
 *      ct.compute_partial((void *)&test, 4, &crc);
 *      ct.compute_partial((void *)&test[4], 5, &crc);
 *      ct.compute_partial_stop(&crc);
 *      printf("The CRC of data \"123456789\" is : 0x%lx\n", crc);
 *      return 0;
 *  }
 * @endcode
 */
template <uint32_t polynomial = POLY_32BIT_ANSI, uint8_t width = 32, CrcMode mode_limit = CrcMode::HARDWARE>
class MbedCRC  {
    impl::MbedCRC<polynomial, width, impl::choose_crc_mode(polynomial, width, mode_limit)> crc_impl;

public:
    /* Backwards compatibility */
    enum CrcMode {
        TABLE       = int(::mbed::CrcMode::TABLE),
        BITWISE     = int(::mbed::CrcMode::BITWISE)
    };

    typedef size_t crc_data_size_t;

    /** Lifetime of CRC object
     *
     *  @param  initial_xor  Initial value/seed to Xor
     *  @param  final_xor  Final Xor value
     *  @param  reflect_data
     *  @param  reflect_remainder
     *  @note   Default constructor without any arguments is valid only for supported CRC polynomials. :: crc_polynomial_t
     *          MbedCRC <POLY_7BIT_SD, 7> ct; --- Valid POLY_7BIT_SD
     *          MbedCRC <0x1021, 16> ct; --- Valid POLY_16BIT_CCITT
     *          MbedCRC <POLY_16BIT_CCITT, 32> ct; --- Invalid, compilation error
     *          MbedCRC <POLY_16BIT_CCITT, 32> ct (i,f,rd,rr) Constructor can be used for not supported polynomials
     *          MbedCRC<POLY_16BIT_CCITT, 16> sd(0, 0, false, false); Constructor can also be used for supported
     *             polynomials with different initial/final/reflect values
     *
     */
    constexpr
    MbedCRC(uint32_t initial_xor, uint32_t final_xor, bool reflect_data, bool reflect_remainder) :
        crc_impl(initial_xor, final_xor, reflect_data, reflect_remainder)
    {
    }

    /* Default values for different types of polynomials
    */
    // *INDENT-OFF*
    template<uint32_t poly = polynomial, std::enable_if_t<poly == POLY_32BIT_ANSI && width == 32, int> = 0>
    constexpr MbedCRC() : MbedCRC(0xFFFFFFFF, 0xFFFFFFFF, true, true)
    {
    }

    template<uint32_t poly = polynomial, std::enable_if_t<poly == POLY_16BIT_IBM && width == 16, int> = 0>
    constexpr MbedCRC() : MbedCRC(0, 0, true, true)
    {
    }

    template<uint32_t poly = polynomial, std::enable_if_t<poly == POLY_16BIT_CCITT && width == 16, int> = 0>
    constexpr MbedCRC() : MbedCRC(0xFFFF, 0, false, false)
    {
    }

    template<uint32_t poly = polynomial, std::enable_if_t<poly == POLY_7BIT_SD && width == 7, int> = 0>
    constexpr MbedCRC() : MbedCRC(0, 0, false, false)
    {
    }

    template<uint32_t poly = polynomial, std::enable_if_t<poly == POLY_8BIT_CCITT && width == 8, int> = 0>
    constexpr MbedCRC() : MbedCRC(0, 0, false, false)
    {
    }
    // *INDENT-ON*

    /** Compute CRC for the data input
     *  Compute CRC performs the initialization, computation and collection of
     *  final CRC.
     *
     *  @param  buffer  Data bytes
     *  @param  size  Size of data
     *  @param  crc  CRC is the output value
     *  @return  0 on success, negative error code on failure
     */
    int32_t compute(const void *buffer, crc_data_size_t size, uint32_t *crc)
    {
        return crc_impl.compute(buffer, size, crc);
    }

    /** Compute partial CRC for the data input.
     *
     *  CRC data if not available fully, CRC can be computed in parts with available data.
     *
     *  In case of hardware, intermediate values and states are saved by hardware. Mutex
     *  locking is used to serialize access to hardware CRC.
     *
     *  In case of software CRC, previous CRC output should be passed as argument to the
     *  current compute_partial call. Please note the intermediate CRC value is maintained by
     *  application and not the driver.
     *
     *  @pre: Call `compute_partial_start` to start the partial CRC calculation.
     *  @post: Call `compute_partial_stop` to get the final CRC value.
     *
     *  @param  buffer  Data bytes
     *  @param  size  Size of data
     *  @param  crc  CRC value is intermediate CRC value filled by API.
     *  @return  0  on success or a negative error code on failure
     *  @note: CRC as output in compute_partial is not final CRC value, call `compute_partial_stop`
     *         to get final correct CRC value.
     */
    int32_t compute_partial(const void *buffer, crc_data_size_t size, uint32_t *crc)
    {
        return crc_impl.compute_partial(buffer, size, crc);
    }

    /** Compute partial start, indicate start of partial computation.
     *
     *  This API should be called before performing any partial computation
     *  with compute_partial API.
     *
     *  @param  crc  Initial CRC value set by the API
     *  @return  0  on success or a negative in case of failure
     *  @note: CRC is an out parameter and must be reused with compute_partial
     *         and `compute_partial_stop` without any modifications in application.
     */
    int32_t compute_partial_start(uint32_t *crc)
    {
        return crc_impl.compute_partial_start(crc);
    }

    /** Get the final CRC value of partial computation.
     *
     *  CRC value available in partial computation is not correct CRC, as some
     *  algorithms require remainder to be reflected and final value to be XORed
     *  This API is used to perform final computation to get correct CRC value.
     *
     *  @param crc  CRC result
     *  @return  0  on success or a negative in case of failure.
     */
    int32_t compute_partial_stop(uint32_t *crc)
    {
        return crc_impl.compute_partial_stop(crc);
    }

    /** Get the current CRC polynomial.
     *
     * @return  Polynomial value
     */
    static constexpr uint32_t get_polynomial()
    {
        return polynomial;
    }

    /** Get the current CRC width
     *
     * @return  CRC width
     */
    static constexpr uint8_t get_width()
    {
        return width;
    }
};

#if !defined(DOXYGEN_ONLY)
/* Internal implementation - basically same as public, but actual mode locked in */
namespace impl {

template <uint32_t polynomial, uint8_t width, CrcMode mode>
class MbedCRC {
public:
    typedef size_t crc_data_size_t;

    constexpr
    MbedCRC(uint32_t initial_xor, uint32_t final_xor, bool reflect_data, bool reflect_remainder) :
        _initial_value(adjust_initial_value(initial_xor, reflect_data)),
        _final_xor(final_xor),
        _reflect_data(reflect_data),
        _reflect_remainder(reflect_remainder)
    {
        static_assert(width <= 32, "Max 32-bit CRC supported");
    }

    /** Compute CRC for the data input
     *  Compute CRC performs the initialization, computation and collection of
     *  final CRC.
     *
     *  @param  buffer  Data bytes
     *  @param  size  Size of data
     *  @param  crc  CRC is the output value
     *  @return  0 on success, negative error code on failure
     */
    int32_t compute(const void *buffer, crc_data_size_t size, uint32_t *crc)
    {
        int32_t status;

        status = compute_partial_start(crc);
        if (0 != status) {
            return status;
        }

        status = compute_partial(buffer, size, crc);
        if (0 != status) {
            return status;
        }

        status = compute_partial_stop(crc);
        return status;
    }

    /** Compute partial CRC for the data input.
     *
     *  CRC data if not available fully, CRC can be computed in parts with available data.
     *
     *  In case of hardware, intermediate values and states are saved by hardware. Mutex
     *  locking is used to serialize access to hardware CRC.
     *
     *  In case of software CRC, previous CRC output should be passed as argument to the
     *  current compute_partial call. Please note the intermediate CRC value is maintained by
     *  application and not the driver.
     *
     *  @pre: Call `compute_partial_start` to start the partial CRC calculation.
     *  @post: Call `compute_partial_stop` to get the final CRC value.
     *
     *  @param  buffer  Data bytes
     *  @param  size  Size of data
     *  @param  crc  CRC value is intermediate CRC value filled by API.
     *  @return  0  on success or a negative error code on failure
     *  @note: CRC as output in compute_partial is not final CRC value, call `compute_partial_stop`
     *         to get final correct CRC value.
     */
    int32_t compute_partial(const void *buffer, crc_data_size_t size, uint32_t *crc)
    {
        const uint8_t *data = static_cast<const uint8_t *>(buffer);
        return do_compute_partial(data, size, crc);
    }

    /** Compute partial start, indicate start of partial computation.
     *
     *  This API should be called before performing any partial computation
     *  with compute_partial API.
     *
     *  @param  crc  Initial CRC value set by the API
     *  @return  0  on success or a negative in case of failure
     *  @note: CRC is an out parameter and must be reused with compute_partial
     *         and `compute_partial_stop` without any modifications in application.
     */
    int32_t compute_partial_start(uint32_t *crc)
    {

        *crc = _initial_value;
        return 0;
    }

    /** Get the final CRC value of partial computation.
     *
     *  CRC value available in partial computation is not correct CRC, as some
     *  algorithms require remainder to be reflected and final value to be XORed
     *  This API is used to perform final computation to get correct CRC value.
     *
     *  @param crc  CRC result
     *  @return  0  on success or a negative in case of failure.
     */
    int32_t compute_partial_stop(uint32_t *crc)
    {
        uint_fast32_t p_crc = *crc;
        if (mode == CrcMode::BITWISE) {
            if (_reflect_data) {
                /* CRC has MSB in bottom bit of register */
                if (!_reflect_remainder) {
                    p_crc = reflect_crc(p_crc);
                }
            } else {
                /* CRC has MSB in top bit of register */
                p_crc = _reflect_remainder ? reflect(p_crc) : shift_right(p_crc);
            }
        } else { // TABLE
            /* CRC has MSB in bottom bit of register */
            if (!_reflect_remainder) {
                p_crc = reflect_crc(p_crc);
            }
        }

        p_crc ^= _final_xor;
        p_crc &= get_crc_mask();
        *crc = p_crc;

        return 0;
    }

private:
    /** Guaranteed constexpr reflection (all toolchains)
     *
     * @note   This should never be run-time evaluated - very inefficient
     * @param  Register value to be reflected (full 32-bit value)
     * @return Reflected value (full 32-bit value)
     */
    static constexpr uint32_t reflect_constant(uint32_t data)
    {
        /* Doing this hard way to keep it C++11 constexpr and hence ARM C 5 compatible */
        return ((data & 0x00000001) << 31) |
               ((data & 0x00000002) << 29) |
               ((data & 0x00000004) << 27) |
               ((data & 0x00000008) << 25) |
               ((data & 0x00000010) << 23) |
               ((data & 0x00000020) << 21) |
               ((data & 0x00000040) << 19) |
               ((data & 0x00000080) << 17) |
               ((data & 0x00000100) << 15) |
               ((data & 0x00000200) << 13) |
               ((data & 0x00000400) << 11) |
               ((data & 0x00000800) <<  9) |
               ((data & 0x00001000) <<  7) |
               ((data & 0x00002000) <<  5) |
               ((data & 0x00004000) <<  3) |
               ((data & 0x00008000) <<  1) |
               ((data & 0x00010000) >>  1) |
               ((data & 0x00020000) >>  3) |
               ((data & 0x00040000) >>  5) |
               ((data & 0x00080000) >>  7) |
               ((data & 0x00100000) >>  9) |
               ((data & 0x00200000) >> 11) |
               ((data & 0x00400000) >> 13) |
               ((data & 0x00800000) >> 15) |
               ((data & 0x01000000) >> 17) |
               ((data & 0x02000000) >> 19) |
               ((data & 0x04000000) >> 21) |
               ((data & 0x08000000) >> 23) |
               ((data & 0x10000000) >> 25) |
               ((data & 0x20000000) >> 27) |
               ((data & 0x40000000) >> 29) |
               ((data & 0x80000000) >> 31);
    }

    /** General reflection
     *
     * @note This is used when we may need to perform run-time computation, so
     * we need the possibility to produce the optimal run-time RBIT instruction. But
     * if the compiler doesn't treat RBIT as a built-in, it's useful to have a C fallback
     * for the constant case, avoiding runtime RBIT(0) computations. This is an
     * optimization only available for some toolchains; others will always use runtime
     * RBIT. If we require a constant expression, use reflect_constant instead.
     *
     * @param  Register value to be reflected (full 32-bit value)
     * @return Reflected value (full 32-bit value)
     */
    static uint32_t reflect(uint32_t data)
    {
        return __RBIT(data);
    }

    /** Data bytes may need to be reflected.
     *
     * @param  data value to be reflected (bottom 8 bits)
     * @return Reflected value (bottom 8 bits)
     */
    static MSTD_CONSTEXPR_IF_HAS_IS_CONSTANT_EVALUATED
    uint_fast32_t reflect_byte(uint_fast32_t data)
    {
        return reflect(data) >> 24;
    }

    /** Get the current CRC polynomial, reflected at bottom of register.
     *
     * @return  Reflected polynomial value (so x^width term would be at bit -1)
     */
    static constexpr uint32_t get_reflected_polynomial()
    {
        return shift_right(reflect_constant(polynomial));
    }

    /** Get the current CRC polynomial, at top of register.
     *
     * @return  Shifted polynomial value (so x^width term would be at bit 32)
     */
    static constexpr uint32_t get_top_polynomial()
    {
        return shift_left(polynomial);
    }

    const uint32_t _initial_value;
    const uint32_t _final_xor;
    const bool _reflect_data;
    const bool _reflect_remainder;

    // *INDENT-OFF*
    using crc_table_t = std::conditional_t<width <= 8,  uint8_t,
                        std::conditional_t<width <= 16, uint16_t,
                                                        uint32_t
                                          >>;
    // *INDENT-ON*

#if MBED_CRC_TABLE_SIZE > 0
    /* Tables only actually defined for mode == TABLE, and certain polynomials - see below */
    static const crc_table_t _crc_table[MBED_CRC_TABLE_SIZE];
#endif

    static constexpr uint32_t adjust_initial_value(uint32_t initial_xor, bool reflect_data)
    {
        if (mode == CrcMode::BITWISE) {
            /* For bitwise calculation, CRC register is reflected if data is, to match input.
             * (MSB at bottom of register). If not reflected, it is at the top of the register
             * (MSB at top of register).
             */
            return reflect_data ? reflect_crc(initial_xor) : shift_left(initial_xor);
        } else if (mode == CrcMode::TABLE) {
            /* For table calculation, CRC value is reflected, to match tables.
             * (MSB at bottom of register). */
            return reflect_crc(initial_xor);
        } else { // CrcMode::HARDWARE
            return initial_xor;
        }
    }

    /** Acquire exclusive access to CRC hardware/software.
     */
    static void lock()
    {
    }

    /** Release exclusive access to CRC hardware/software.
     */
    static void unlock()
    {
    }

    /** Get the CRC data mask.
     *
     * @return  CRC data mask is generated based on current CRC width
     */
    static constexpr uint32_t get_crc_mask()
    {
        return (uint32_t)((uint32_t)2U << (width - 1)) - 1U;
    }

    /** CRC values may need to be reflected.
     *
     * @param  CRC value to be reflected (width bits at bottom of 32-bit word)
     * @return Reflected value (still at bottom of 32-bit word)
     */
    static MSTD_CONSTEXPR_IF_HAS_IS_CONSTANT_EVALUATED
    uint32_t reflect_crc(uint32_t data)
    {
        return reflect(data) >> (32 - width);
    }

    /** Register values may need to be shifted left.
     *
     * @param  Register value to be shifted up (in bottom width bits)
     * @return Shifted value (in top width bits)
     */
    static constexpr uint32_t shift_left(uint32_t data)
    {
        return data << (32 - width);
    }

    /** Register values may need to be shifted right.
     *
     * @param  Register value to be shifted right (in top width bits)
     * @return  Shifted value (in bottom width bits)
     */
    static constexpr uint32_t shift_right(uint32_t data)
    {
        return data >> (32 - width);
    }

    /* Check to see if we can do assembler optimizations */
#if ((defined __GNUC__ || defined __clang__) && !defined __CC_ARM) && \
    (defined __arm__ || defined __ARM_ARCH)
#if (__ARM_ARCH_7M__      == 1U) || \
    (__ARM_ARCH_7EM__     == 1U) || \
    (__ARM_ARCH_8M_MAIN__ == 1U) || \
    (__ARM_ARCH_7A__      == 1U)
    /* ARM that has Thumb-2 - same unified assembly is good for either ARM or Thumb state (LSRS; IT CS; EORCS reg/imm) */
#define MBED_CRC_ARM_THUMB2     1
#define MBED_CRC_THUMB1         0
#elif (__ARM_ARCH_6M__      == 1U) || \
      (__ARM_ARCH_8M_BASE__ == 1U)
    /* Thumb-1-only ARM-M device - use Thumb-1 compatible assembly with branch (LSRS; BCC; EORS reg) */
#define MBED_CRC_ARM_THUMB2     0
#define MBED_CRC_THUMB1         1
#else // __ARM_ARCH_xxx
#error "Unknown ARM architecture for CRC optimization"
#endif // __ARM_ARCH_xxx
#else // __arm__ || defined __ICC_ARM__ || defined __ARM_ARCH
    /* Seem to be compiling for non-ARM, or an unsupported toolchain, so stick with C implementations */
#define MBED_CRC_ARM_THUMB2     0
#define MBED_CRC_THUMB1         0
#endif

    // *INDENT-OFF*
    /** Process 1 bit of non-reflected CRC
     *
     * Shift the p_crc register left 1 bit - if a one is shifted
     * out, exclusive-or with the polynomial mask.
     *
     * Assembler optimizations can be applied here, to make
     * use of the CPU's carry output from shifts.
     *
     * @param  p_crc input register value
     * @return updated register value
     */
    static uint_fast32_t do_1_bit_normal(uint_fast32_t p_crc)
    {
#if MBED_CRC_ARM_THUMB2
        __asm(".syntax unified\n\t"
              "LSLS"  "\t%[p_crc], %[p_crc], #1\n\t"
              "IT"    "\tCS\n\t"
              "EORCS" "\t%[p_crc], %[poly]"
              : [p_crc] "+&r" (p_crc)
              : [poly] "rI" (get_top_polynomial())
              : "cc");
#elif MBED_CRC_THUMB1
        __asm(".syntax unified\n\t"
              "LSLS"  "\t%[p_crc], %[p_crc], #1\n\t"
              "BCC"   "\t%=f\n\t"
              "EORS"  "\t%[p_crc], %[poly]\n"
              "%=:"
              : [p_crc] "+&l" (p_crc)
              : [poly] "l" (get_top_polynomial())
              : "cc");
#else
        if (p_crc & 0x80000000) {
            p_crc = (p_crc << 1) ^ get_top_polynomial();
        } else {
            p_crc = (p_crc << 1);
        }
#endif
        return p_crc;
    }

    /** Process 1 bit of reflected CRC
     *
     * Shift the p_crc register right 1 bit - if a one is shifted
     * out, exclusive-or with the polynomial mask.
     *
     * Assembler optimizations can be applied here, to make
     * use of the CPU's carry output from shifts.
     *
     * @param  p_crc input register value
     * @return updated register value
     */
    static uint_fast32_t do_1_bit_reflected(uint_fast32_t p_crc)
    {
#if MBED_CRC_ARM_THUMB2
        __asm(".syntax unified\n\t"
              "LSRS"  "\t%[p_crc], %[p_crc], #1\n\t"
              "IT"    "\tCS\n\t"
              "EORCS" "\t%[p_crc], %[poly]"
              : [p_crc] "+&r" (p_crc)
              : [poly] "rI" (get_reflected_polynomial())
              : "cc");
#elif MBED_CRC_THUMB1
        __asm(".syntax unified\n\t"
              "LSRS"  "\t%[p_crc], %[p_crc], #1\n\t"
              "BCC"   "\t%=f\n\t"
              "EORS"  "\t%[p_crc], %[poly]\n"
              "%=:"
              : [p_crc] "+&l" (p_crc)
              : [poly] "l" (get_reflected_polynomial())
              : "cc");
#else
        if (p_crc & 1) {
            p_crc = (p_crc >> 1) ^ get_reflected_polynomial();
        } else {
            p_crc = (p_crc >> 1);
        }
#endif
        return p_crc;
    }
    // *INDENT-ON*

    /** Bitwise CRC computation.
     *
     * @param  buffer  data buffer
     * @param  size  size of the data
     * @param  crc  CRC value is filled in, but the value is not the final
     * @return  0  on success or a negative error code on failure
     */
    template<CrcMode mode_ = mode>
    std::enable_if_t<mode_ == CrcMode::BITWISE, int32_t>
    do_compute_partial(const uint8_t *data, crc_data_size_t size, uint32_t *crc) const
    {
        uint_fast32_t p_crc = *crc;

        if (_reflect_data) {
            /* Everything is reflected to match data - MSB of polynomial at bottom of 32-bit register */
            for (crc_data_size_t byte = 0; byte < size; byte++) {
                p_crc ^= data[byte];

                // Perform modulo-2 division, a bit at a time
                for (unsigned int bit = 8; bit > 0; --bit) {
                    p_crc = do_1_bit_reflected(p_crc);
                }
            }
        } else {
            /* Polynomial is shifted to put MSB of polynomial at top of 32-bit register */
            for (crc_data_size_t byte = 0; byte < size; byte++) {
                p_crc ^= (uint_fast32_t) data[byte] << 24;

                // Perform modulo-2 division, a bit at a time
                for (unsigned int bit = 8; bit > 0; --bit) {
                    p_crc = do_1_bit_normal(p_crc);
                }
            }
        }

        *crc = p_crc;

        return 0;
    }

#if MBED_CRC_TABLE_SIZE > 0
    /** CRC computation using ROM tables.
    *
    * @param  buffer  data buffer
    * @param  size  size of the data
    * @param  crc  CRC value is filled in, but the value is not the final
    * @return  0  on success or a negative error code on failure
    */
    template<CrcMode mode_ = mode>
    std::enable_if_t<mode_ == CrcMode::TABLE, int32_t>
    do_compute_partial(const uint8_t *data, crc_data_size_t size, uint32_t *crc) const
    {
        uint_fast32_t p_crc = *crc;
        // GCC has been observed to not hoist the load of _reflect_data out of the loop
        // Note the inversion because table and CRC are reflected - data must be
        bool reflect = !_reflect_data;

        for (crc_data_size_t byte = 0; byte < size; byte++) {
            uint_fast32_t data_byte = data[byte];
            if (reflect) {
                data_byte = reflect_byte(data_byte);
            }
#if MBED_CRC_TABLE_SIZE == 16
            p_crc = _crc_table[(data_byte ^ p_crc) & 0xF] ^ (p_crc >> 4);
            data_byte >>= 4;
            p_crc = _crc_table[(data_byte ^ p_crc) & 0xF] ^ (p_crc >> 4);
#else
            p_crc = _crc_table[(data_byte ^ p_crc) & 0xFF] ^ (p_crc >> 8);
#endif
        }
        *crc = p_crc;
        return 0;
    }
#endif


};

#if MBED_CRC_TABLE_SIZE > 0
/* Declarations of the tables we provide. (Not strictly needed, but compilers
 * can warn if they see us using the template without a generic definition, so
 * let it know we have provided these specialisations.)
 */
template<>
const uint8_t MbedCRC<POLY_7BIT_SD, 7, CrcMode::TABLE>::_crc_table[MBED_CRC_TABLE_SIZE];

template<>
const uint8_t MbedCRC<POLY_8BIT_CCITT, 8, CrcMode::TABLE>::_crc_table[MBED_CRC_TABLE_SIZE];

template<>
const uint16_t MbedCRC<POLY_16BIT_CCITT, 16, CrcMode::TABLE>::_crc_table[MBED_CRC_TABLE_SIZE];

template<>
const uint16_t MbedCRC<POLY_16BIT_IBM, 16, CrcMode::TABLE>::_crc_table[MBED_CRC_TABLE_SIZE];

template<>
const uint32_t MbedCRC<POLY_32BIT_ANSI, 32, CrcMode::TABLE>::_crc_table[MBED_CRC_TABLE_SIZE];

#endif // MBED_CRC_TABLE_SIZE > 0

} // namespace impl

#endif // !defined(DOXYGEN_ONLY)

/** @}*/
/** @}*/

} // namespace runtime
} // namespace tvm

#endif // __cplusplus

/* Internal helper for mbed_error.c crash recovery */
#ifdef __cplusplus
extern "C"
#endif
uint32_t mbed_tiny_compute_crc32(const void *data, int datalen);

#endif
