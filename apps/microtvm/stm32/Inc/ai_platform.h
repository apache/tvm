/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file ai_platform.h
 * \brief Definitions of AI platform public APIs types
 */

// LINT_C_FILE

#ifndef AI_PLATFORM_H
#define AI_PLATFORM_H
#pragma once

#include <stdint.h>
#include <stddef.h>
#include <inttypes.h>

#include "ai_runtime_api.h"

#ifndef AI_PLATFORM_API_MAJOR
#define AI_PLATFORM_API_MAJOR           (0)
#endif
#ifndef AI_PLATFORM_API_MINOR
#define AI_PLATFORM_API_MINOR           (6)
#endif
#ifndef AI_PLATFORM_API_MICRO
#define AI_PLATFORM_API_MICRO           (0)
#endif

#define AI_PLATFORM_API_VERSION \
  AI_VERSION(AI_PLATFORM_API_MAJOR, \
             AI_PLATFORM_API_MINOR, \
             AI_PLATFORM_API_MICRO)


#ifndef AI_TOOLS_API_VERSION_MAJOR
#define AI_TOOLS_API_VERSION_MAJOR      (1)
#endif
#ifndef AI_TOOLS_API_VERSION_MINOR
#define AI_TOOLS_API_VERSION_MINOR      (4)
#endif
#ifndef AI_TOOLS_API_VERSION_MICRO
#define AI_TOOLS_API_VERSION_MICRO      (0)
#endif

#define AI_TOOLS_API_VERSION \
  AI_VERSION(AI_TOOLS_API_VERSION_MAJOR, \
             AI_TOOLS_API_VERSION_MINOR, \
             AI_TOOLS_API_VERSION_MICRO)

#define AI_TOOLS_API_VERSION_1_3 \
  AI_VERSION(1, 3, 0)

/******************************************************************************/
#ifdef __cplusplus
#define AI_API_DECLARE_BEGIN extern "C" {
#define AI_API_DECLARE_END }
#else
#include <stdbool.h>
#define AI_API_DECLARE_BEGIN    /* AI_API_DECLARE_BEGIN */
#define AI_API_DECLARE_END      /* AI_API_DECLARE_END   */
#endif

/******************************************************************************/

#define AI_HANDLE_FUNC_PTR(func)      ((ai_handle_func)(func))

#define AI_NETWORK_PARAMS_INIT(params_, activations_) { \
  .params = params_, \
  .activations = activations_ }

/*! ai_intq_info struct handlers **********************************************/
#define INTQ_CONST    const

#define AI_INTQ_INFO_LIST(list_) \
  ((list_)->info)

#define AI_INTQ_INFO_LIST_FLAGS(list_) \
  ((list_) ? (list_)->flags : 0)

#define AI_INTQ_INFO_LIST_SIZE(list_) \
  ((list_) ? (list_)->size : 0)

#define AI_HAS_INTQ_INFO_LIST(list_) \
  ((list_) ? (((list_)->info) && ((list_)->size > 0)) : false)

#define AI_INTQ_INFO_LIST_SCALE(list_, type_, pos_) \
  (((list_) && (list_)->info && ((pos_) < (list_)->size)) ? \
  ((type_*)((list_)->info->scale))[(pos_)] : 0)

#define AI_INTQ_INFO_LIST_ZEROPOINT(list_, type_, pos_) \
  (((list_) && (list_)->info && ((pos_) < (list_)->size)) ? \
  ((type_*)((list_)->info->zeropoint))[(pos_)] : 0)

/*! ai_buffer format handlers *************************************************/

/*!
 * @enum buffer format definition
 * @ingroup ai_platform
 *
 * 32 bit signed format list.
 */
typedef int32_t ai_buffer_format;

/*! ai_buffer_meta flags ******************************************************/
#define AI_BUFFER_META_HAS_INTQ_INFO        (0x1U << 0)
#define AI_BUFFER_META_FLAG_SCALE_FLOAT     (0x1U << 0)
#define AI_BUFFER_META_FLAG_ZEROPOINT_U8    (0x1U << 1)
#define AI_BUFFER_META_FLAG_ZEROPOINT_S8    (0x1U << 2)

/*! ai_buffer format variable flags *******************************************/
#define AI_BUFFER_FMT_TYPE_NONE          (0x0)
#define AI_BUFFER_FMT_TYPE_FLOAT         (0x1)
#define AI_BUFFER_FMT_TYPE_Q             (0x2)
#define AI_BUFFER_FMT_TYPE_BOOL          (0x3)

#define AI_BUFFER_FMT_FLAG_CONST         (0x1U << 30)
#define AI_BUFFER_FMT_FLAG_STATIC        (0x1U << 29)
#define AI_BUFFER_FMT_FLAG_IS_IO         (0x1U << 27)

#define AI_BUFFER_FMT_PACK(value_, mask_, bits_) \
  ( ((value_) & (mask_)) << (bits_) )

#define AI_BUFFER_FMT_UNPACK(fmt_, mask_, bits_) \
  ( (AI_BUFFER_FMT_OBJ(fmt_) >> (bits_)) & (mask_) )

#define AI_BUFFER_FMT_OBJ(fmt_) \
  ((ai_buffer_format)(fmt_))

#define AI_BUFFER_FMT_GET_FLOAT(fmt_) \
  AI_BUFFER_FMT_UNPACK(fmt_, 0x1, 24)

#define AI_BUFFER_FMT_GET_SIGN(fmt_) \
  AI_BUFFER_FMT_UNPACK(fmt_, 0x1, 23)

#define AI_BUFFER_FMT_GET_TYPE(fmt_) \
  AI_BUFFER_FMT_UNPACK(fmt_, 0xF, 17)

#define AI_BUFFER_FMT_GET_BITS(fmt_) \
  AI_BUFFER_FMT_UNPACK(fmt_, 0x7F, 7)

#define AI_BUFFER_FMT_SET_BITS(bits_) \
  AI_BUFFER_FMT_PACK((bits_), 0x7F, 7)

#define AI_BUFFER_FMT_GET_FBITS(fmt_) \
  ( (ai_i8)AI_BUFFER_FMT_UNPACK(fmt_, 0x7F, 0) - 64 )

#define AI_BUFFER_FMT_SET_FBITS(fbits_) \
  AI_BUFFER_FMT_PACK((fbits_)+64, 0x7F, 0)

#define AI_BUFFER_FMT_SET(type_id_, sign_bit_, float_bit_, bits_, fbits_) \
  AI_BUFFER_FMT_OBJ( \
    AI_BUFFER_FMT_PACK(float_bit_, 0x1, 24) | \
    AI_BUFFER_FMT_PACK(sign_bit_, 0x1, 23) | \
    AI_BUFFER_FMT_PACK(0, 0x3, 21) | \
    AI_BUFFER_FMT_PACK(type_id_, 0xF, 17) | \
    AI_BUFFER_FMT_PACK(0, 0x7, 14) | \
    AI_BUFFER_FMT_SET_BITS(bits_) | \
    AI_BUFFER_FMT_SET_FBITS(fbits_) \
  )

#define AI_BUFFER_FMT_GET(fmt_) \
  (AI_BUFFER_FMT_OBJ(fmt_) & 0x01FFFFFF)

#define AI_BUFFER_FORMAT(buf_) \
  AI_BUFFER_FMT_GET((buf_)->format)
#define AI_BUFFER_WIDTH(buf_) \
  ((buf_)->width)
#define AI_BUFFER_HEIGHT(buf_) \
  ((buf_)->height)
#define AI_BUFFER_CHANNELS(buf_) \
  ((buf_)->channels)
#define AI_BUFFER_N_BATCHES(buf_) \
  ((buf_)->n_batches)
#define AI_BUFFER_DATA(buf_, type_) \
  ((type_*)((buf_)->data))

#define AI_BUFFER_META_INFO(buf_) \
  ((buf_)->meta_info)

#define AI_BUFFER_META_INFO_INTQ(meta_) \
  ((meta_) && ((meta_)->flags & AI_BUFFER_META_HAS_INTQ_INFO)) \
    ? ((meta_)->intq_info) : NULL

#define AI_BUFFER_META_INFO_INTQ_GET_SCALE(meta_, pos_) \
  ( (AI_BUFFER_META_INFO_INTQ(meta_)) \
    ? AI_INTQ_INFO_LIST_SCALE(AI_BUFFER_META_INFO_INTQ(meta_), ai_float, pos_) \
    : 0 )

#define AI_BUFFER_META_INFO_INTQ_GET_ZEROPOINT(meta_, pos_) \
  ( (AI_BUFFER_META_INFO_INTQ(meta_)) \
    ? ((AI_INTQ_INFO_LIST_FLAGS(AI_BUFFER_META_INFO_INTQ(meta_))&AI_BUFFER_META_FLAG_ZEROPOINT_U8) \
      ? AI_INTQ_INFO_LIST_ZEROPOINT(AI_BUFFER_META_INFO_INTQ(meta_), ai_u8, pos_) \
      : AI_INTQ_INFO_LIST_ZEROPOINT(AI_BUFFER_META_INFO_INTQ(meta_), ai_i8, pos_) ) \
    : 0 )

#define AI_BUFFER_META_INFO_INIT(flags_, intq_info_) { \
  .flags = (flags_), \
  .intq_info = AI_PACK(intq_info_) \
}

#define AI_BUFFER_SIZE(buf_) \
  (((buf_)->width) * ((buf_)->height) * ((buf_)->channels))

#define AI_BUFFER_BYTE_SIZE(count_, fmt_) \
  ( (((count_) * AI_BUFFER_FMT_GET_BITS(fmt_))+4) >> 3 )


/*!
 * @enum buffer formats enum list
 * @ingroup ai_platform
 *
 * List of supported ai_buffer format types.
 */
enum {
  AI_BUFFER_FORMAT_NONE     = AI_BUFFER_FMT_SET(AI_BUFFER_FMT_TYPE_NONE, 0, 0,  0, 0),
  AI_BUFFER_FORMAT_FLOAT    = AI_BUFFER_FMT_SET(AI_BUFFER_FMT_TYPE_FLOAT, 1, 1, 32, 0),

  AI_BUFFER_FORMAT_U8       = AI_BUFFER_FMT_SET(AI_BUFFER_FMT_TYPE_Q, 0, 0,  8, 0),
  AI_BUFFER_FORMAT_U16      = AI_BUFFER_FMT_SET(AI_BUFFER_FMT_TYPE_Q, 0, 0, 16, 0),
  AI_BUFFER_FORMAT_S8       = AI_BUFFER_FMT_SET(AI_BUFFER_FMT_TYPE_Q, 1, 0,  8, 0),
  AI_BUFFER_FORMAT_S16      = AI_BUFFER_FMT_SET(AI_BUFFER_FMT_TYPE_Q, 1, 0, 16, 0),

  AI_BUFFER_FORMAT_Q        = AI_BUFFER_FMT_SET(AI_BUFFER_FMT_TYPE_Q, 1, 0,  0, 0),
  AI_BUFFER_FORMAT_Q7       = AI_BUFFER_FMT_SET(AI_BUFFER_FMT_TYPE_Q, 1, 0,  8, 7),
  AI_BUFFER_FORMAT_Q15      = AI_BUFFER_FMT_SET(AI_BUFFER_FMT_TYPE_Q, 1, 0, 16, 15),

  AI_BUFFER_FORMAT_UQ       = AI_BUFFER_FMT_SET(AI_BUFFER_FMT_TYPE_Q, 0, 0,  0, 0),
  AI_BUFFER_FORMAT_UQ7      = AI_BUFFER_FMT_SET(AI_BUFFER_FMT_TYPE_Q, 0, 0,  8, 7),
  AI_BUFFER_FORMAT_UQ15     = AI_BUFFER_FMT_SET(AI_BUFFER_FMT_TYPE_Q, 0, 0, 16, 15),

  AI_BUFFER_FORMAT_BOOL     = AI_BUFFER_FMT_SET(AI_BUFFER_FMT_TYPE_BOOL, 0, 0, 8, 0),
};

/******************************************************************************/


/* printf formats */
#define SSIZET_FMT  "%" PRIu32
#define AII32_FMT   "%" PRId32
#define AIU32_FMT   "%" PRIu32

typedef uint8_t ai_custom_type_signature;

typedef void (*ai_handle_func)(void*);

typedef float ai_float;
typedef double ai_double;

typedef bool ai_bool;

typedef uint32_t ai_size;

typedef uintptr_t ai_uptr;

typedef unsigned int ai_uint;
typedef uint8_t  ai_u8;
typedef uint16_t ai_u16;
typedef uint32_t ai_u32;
typedef uint64_t ai_u64;

typedef int     ai_int;
typedef int8_t  ai_i8;
typedef int16_t ai_i16;
typedef int32_t ai_i32;
typedef int64_t ai_i64;

typedef uint32_t ai_signature;


/******************************************************************************/
/*!
 * @struct ai_intq_info
 * @ingroup ai_platform
 * @brief an element of the ai_intq_info_list entry. It reports an array for the
 * scale and zeropoint values for each buffer. Optional flags are also present
 */
typedef struct ai_intq_info_ {
  INTQ_CONST ai_float*  scale;
  INTQ_CONST ai_handle  zeropoint;
} ai_intq_info;

/*!
 * @struct ai_intq_info_list
 * @ingroup ai_platform
 * @brief list reporting meta info for quantized networks integer support
 * when size > 1 it means a per channel out quantization
 */
typedef struct ai_intq_info_list_ {
  ai_u16          flags;  /*!< optional flags to store intq info attributes */
  ai_u16          size;   /*!< number of elements in the the intq_info list  */
  INTQ_CONST ai_intq_info* info;  /*!< pointer to an array of quant info */
} ai_intq_info_list;

/******************************************************************************/
/*!
 * @struct ai_buffer_meta_info
 * @ingroup ai_platform
 * @brief Optional meta attributes associated with the I/O buffer.
 * This datastruct is used also for network querying, where the data field may
 * may be NULL.
 */
typedef struct ai_buffer_meta_info_ {
  ai_u32                  flags;      /*!< meta info flags */
  ai_intq_info_list*      intq_info;  /*!< meta info related to integer format */
} ai_buffer_meta_info;

/*!
 * @struct ai_buffer
 * @ingroup ai_platform
 * @brief Memory buffer storing data (optional) with a shape, size and type.
 * This datastruct is used also for network querying, where the data field may
 * may be NULL.
 */
typedef struct ai_buffer_ {
  ai_buffer_format        format;     /*!< buffer format */
  ai_u16                  n_batches;  /*!< number of batches in the buffer */
  ai_u16                  height;     /*!< buffer height dimension */
  ai_u16                  width;      /*!< buffer width dimension */
  ai_u32                  channels;   /*!< buffer number of channels */
  ai_handle               data;       /*!< pointer to buffer data */
  ai_buffer_meta_info*    meta_info;  /*!< pointer to buffer metadata info */
} ai_buffer;

/*!
 * @struct ai_platform_version
 * @ingroup ai_platform
 * @brief Datastruct storing platform version info
 */
typedef struct ai_platform_version_ {
  ai_u8               major;
  ai_u8               minor;
  ai_u8               micro;
  ai_u8               reserved;
} ai_platform_version;

/*!
 * @struct ai_network_report
 * @ingroup ai_platform
 *
 * Datastructure to query a network report with some relevant network detail.
 */
typedef struct ai_network_report_ {
  const char*                     model_name;
  const char*                     model_signature;
  const char*                     model_datetime;

  const char*                     compile_datetime;

  const char*                     runtime_revision;
  ai_platform_version             runtime_version;

  const char*                     tool_revision;
  ai_platform_version             tool_version;
  ai_platform_version             tool_api_version;

  ai_platform_version             api_version;
  ai_platform_version             interface_api_version;

  ai_u32                          n_macc;

  ai_u16                          n_inputs;
  ai_u16                          n_outputs;
  ai_buffer*                      inputs;
  ai_buffer*                      outputs;

  ai_buffer                       activations;
  ai_buffer                       params;

  ai_u32                          n_nodes;

  ai_signature                    signature;
} ai_network_report;

#endif /*AI_PLATFORM_H*/
