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
 * \file aiPbMgr.c
 * \brief Helper function for AI ProtoBuffer support
 */

#include <aiPbMgr.h>
#include <aiPbIO.h>

#include <pb_encode.h>
#include <pb_decode.h>

#include <aiTestUtility.h> /* for HAL and specific device functions */


/*---------------------------------------------------------------------------*/

static struct pbContextMgr {
  pb_istream_t input;
  pb_ostream_t output;
  const aiPbCmdFunc *funcs;
  uint32_t  n_func;
} pbContextMgr;

void aiPbMgrInit(const aiPbCmdFunc *funcs)
{
  const aiPbCmdFunc *cfunc;

  pb_io_stream_init();

  pbContextMgr.input = pb_io_istream(0);
  pbContextMgr.output = pb_io_ostream(0);

  pbContextMgr.n_func = 0;
  pbContextMgr.funcs = NULL;

  if (funcs) {
    cfunc = funcs;
    while (cfunc->process) {
      pbContextMgr.n_func++;
      cfunc++;
    }
    pbContextMgr.funcs = funcs;
  }
}

int aiPbMgrWaitAndProcess(void)
{
  uint32_t idx;
  static reqMsg  req = reqMsg_init_zero;
  static respMsg resp = respMsg_init_default;
  const aiPbCmdFunc *cfunc;

  pb_io_flush_istream();
  if (pb_decode_delimited(&pbContextMgr.input, reqMsg_fields, &req)) {
    pb_io_flush_istream();
    for (idx = 0; idx < pbContextMgr.n_func; idx++) {
      cfunc = &pbContextMgr.funcs[idx];
      if (cfunc->cmd == req.cmd) {
        cfunc->process(&req, &resp, cfunc->param);
        break;
      }
    }
    if (idx == pbContextMgr.n_func) {
      aiPbMgrSendAck(&req, &resp, EnumState_S_ERROR,
          EnumError_E_INVALID_PARAM, EnumError_E_INVALID_PARAM);
    }
  }

  pb_io_flush_istream();

  return 0;
}


/*---------------------------------------------------------------------------*/

void aiPbMgrSendResp(const reqMsg *req, respMsg *resp,
    EnumState state)
{
  resp->reqid = req->reqid;
  resp->state = state;
  pb_encode(&pbContextMgr.output, respMsg_fields, resp);
  pb_io_flush_ostream();
}

void aiPbMgrSendAck(const reqMsg *req, respMsg *resp,
    EnumState state, uint32_t param, EnumError error)
{
  resp->which_payload = respMsg_ack_tag;
  resp->payload.ack.param = param;
  resp->payload.ack.error = error;
  aiPbMgrSendResp(req, resp, state);
}

bool aiPbMgrWaitAck(void)
{
  bool res;
  ackMsg ack = ackMsg_init_default;
  res = pb_decode_delimited(&pbContextMgr.input, ackMsg_fields, &ack);
  pb_io_flush_istream();
  return res;
}

bool aiPbMgrSendLog(const reqMsg *req, respMsg *resp,
    EnumState state, uint32_t lvl, const char *str)
{
  bool res;
  ackMsg ack = ackMsg_init_default;

  size_t len = strlen(str);

  resp->which_payload = respMsg_log_tag;
  resp->payload.log.level = lvl;
  if (len >= sizeof(resp->payload.log.str))
    len = sizeof(resp->payload.log.str) - 1;

  memcpy(&resp->payload.log.str[0], str, len+1);

  aiPbMgrSendResp(req, resp, state);

  res = pb_decode_delimited(&pbContextMgr.input, ackMsg_fields, &ack);
  pb_io_flush_istream();
  return res;
}

struct aiPbMgrBuffer {
  ai_buffer *buffer;
  uint32_t n_max;
  uint32_t n_ops;
  uint32_t err;
  void *msg;
};

uint32_t aiPbAiBufferSize(const ai_buffer *buffer)
{
  if (!buffer)
    return 0;
  else
    return buffer->channels * buffer->height
        * buffer->width * buffer->n_batches;
}

static ai_buffer_format aiPbMsgFmtToAiFmt(const uint32_t msgFmt)
{
  return (ai_buffer_format)msgFmt;
}

static uint32_t aiPbAiFmtToMsgFmt(const ai_buffer_format aiFmt)
{
  return (uint32_t)aiFmt;
}

static size_t aiPbBufferGetItemSize(ai_buffer_format format)
{
  return (size_t)AI_BUFFER_BYTE_SIZE(1,format);
}

static bool aiPbBuffer_read_cb3(pb_istream_t *stream, const pb_field_t *field,
    void **arg)
{
  struct aiPbMgrBuffer *bm = (struct aiPbMgrBuffer *)*arg;
  aiBufferByteMsg *msg;
  ai_buffer_format format;
  size_t itsize;

  UNUSED(field);

  int maxr = bm->n_max;
  msg = (aiBufferByteMsg *)bm->msg;
  format = aiPbMsgFmtToAiFmt(msg->shape.format);

  /* todo(jmd) - adding scale/zeropoint values */

      /* Check shape/format */
      bm->err = EnumError_E_NONE;
      if ((format == AI_BUFFER_FORMAT_NONE) || (format != bm->buffer->format)) {
        maxr = 0;
        bm->err = EnumError_E_INVALID_FORMAT;
      } else if ((msg->shape.channels != bm->buffer->channels) ||
          (msg->shape.height != bm->buffer->height) ||
          (msg->shape.width != bm->buffer->width) ||
          (msg->shape.n_batches != bm->buffer->n_batches)) {
        maxr = 0;
        bm->err = EnumError_E_INVALID_SHAPE;
      }

      itsize = aiPbBufferGetItemSize(format);

      /* Read data */
      uint8_t *pw = (uint8_t *)bm->buffer->data;
      while (stream->bytes_left) {
        uint64_t number;
        if (!pb_read(stream, (pb_byte_t *)&number, itsize))
          return false;
        if (maxr > 0) {
          if (pw) {
            memcpy(pw, &number, itsize);
            pw += itsize;
          }
          maxr--;
        }
        bm->n_ops++;
      }

      /* Check nb_op */
      if ((bm->err == EnumError_E_NONE) && (bm->n_ops != bm->n_max))
        bm->err = EnumError_E_INVALID_SIZE;

      return true;
}

static bool aiPbBuffer_write_cb3(pb_ostream_t *stream, const pb_field_t *field,
    void * const *arg)
{
  struct aiPbMgrBuffer *bm = (struct aiPbMgrBuffer *)*arg;
  size_t itsize;

  int maxw = bm->n_max;
  ai_buffer_format format;

  if ((maxw == 0) || (!bm->buffer))
    return true;

  format = bm->buffer->format;

  itsize = aiPbBufferGetItemSize(format);

  /* Write data */
  pb_byte_t *pr = (pb_byte_t *)bm->buffer->data;

  if (!pb_encode_tag_for_field(stream, field))
    return false;

  if (!pb_encode_string(stream, pr, itsize * maxw))
    return false;

  bm->n_ops = maxw;

  return true;
}

bool aiPbMgrReceiveAiBuffer3(const reqMsg *req, respMsg *resp,
    EnumState state, ai_buffer *buffer)
{
  aiBufferByteMsg msg;
  struct aiPbMgrBuffer hdlb;
  bool res = true;

  hdlb.n_ops = 0;
  hdlb.buffer = buffer;
  hdlb.err = EnumError_E_NONE;
  hdlb.n_max = aiPbAiBufferSize(buffer);
  hdlb.msg = &msg;

  msg.datas.funcs.decode = &aiPbBuffer_read_cb3;
  msg.datas.arg = &hdlb;

  /* Waiting buffer message */
  pb_decode_delimited(&pbContextMgr.input, aiBufferByteMsg_fields, &msg);
  pb_io_flush_istream();

  /* Send ACK and wait ACK (or send ACK only if error) */
  if (hdlb.err) {
    aiPbMgrSendAck(req, resp, EnumState_S_ERROR, hdlb.err,
        (EnumError)hdlb.err);
    res = false;
  } else {
    aiPbMgrSendAck(req, resp, state, hdlb.n_ops, EnumError_E_NONE);
    if ((state == EnumState_S_WAITING) ||
        (state == EnumState_S_PROCESSING))
      aiPbMgrWaitAck();
  }

  return res;
}

#if !defined(NO_X_CUBE_AI_RUNTIME)
static void aiPbMgrSetMetaInfo(const ai_buffer_meta_info *meta_info, const int idx,
    aiBufferShapeMsg *shape)
{
  shape->scale = 0.0f;
  shape->zeropoint = 0;
  if (AI_BUFFER_META_INFO_INTQ(meta_info)) {
    shape->scale = AI_BUFFER_META_INFO_INTQ_GET_SCALE(meta_info, idx);
    shape->zeropoint = AI_BUFFER_META_INFO_INTQ_GET_ZEROPOINT(meta_info, idx);
  }
}
#endif

bool aiPbMgrSendAiBuffer4(const reqMsg *req, respMsg *resp, EnumState state,
    uint32_t type, uint32_t id, ai_float dur_ms, const ai_buffer *buffer,
    ai_float scale, ai_i32 zero_point)
{
  struct aiPbMgrBuffer hdlb;
  const ai_buffer_meta_info *meta_info = AI_BUFFER_META_INFO(buffer);

#if defined(AI_PB_FULL_IO) && (AI_PB_FULL_IO == 1)
  const int is_io = AI_BUFFER_FMT_FLAG_IS_IO & buffer->format;
#endif

  hdlb.n_ops = 0;
  hdlb.buffer = (ai_buffer *)buffer;
  hdlb.err = EnumError_E_NONE;
  hdlb.n_max = aiPbAiBufferSize(buffer);
  hdlb.msg = NULL;

#if defined(AI_PB_FULL_IO) && (AI_PB_FULL_IO == 1)
  if ((type & PB_BUFFER_TYPE_SEND_WITHOUT_DATA) && (!is_io)) {
    hdlb.n_max  = 0;
  }
#else
  if (type & PB_BUFFER_TYPE_SEND_WITHOUT_DATA) {
    hdlb.n_max  = 0;
  }
#endif
  type &= (~PB_BUFFER_TYPE_SEND_WITHOUT_DATA);

  /* Fill Node sub-message */
  resp->which_payload = respMsg_node_tag;
  resp->payload.node.type = type;
  resp->payload.node.id = id;
  resp->payload.node.duration = dur_ms;
  resp->payload.node.buffer.shape.format = aiPbAiFmtToMsgFmt(buffer->format);
  resp->payload.node.buffer.shape.n_batches = buffer->n_batches;
  resp->payload.node.buffer.shape.height = buffer->height;
  resp->payload.node.buffer.shape.width = buffer->width;
  resp->payload.node.buffer.shape.channels = buffer->channels;

  if (meta_info && scale == 0.0f) {
#if defined(NO_X_CUBE_AI_RUNTIME) && NO_X_CUBE_AI_RUNTIME == 1
  struct ai_buffer_ext *ext = (struct ai_buffer_ext *)buffer;
  resp->payload.node.buffer.shape.scale = ext->extra.scale;
  resp->payload.node.buffer.shape.zeropoint = ext->extra.zero_point;
#else
    aiPbMgrSetMetaInfo(meta_info, 0, &resp->payload.node.buffer.shape);
#endif
  }
  else {
    resp->payload.node.buffer.shape.scale = scale;
    resp->payload.node.buffer.shape.zeropoint = zero_point;
  }

  resp->payload.node.buffer.datas.funcs.encode = &aiPbBuffer_write_cb3;
  resp->payload.node.buffer.datas.arg = &hdlb;

  /* Send msg */
  aiPbMgrSendResp(req, resp, state);

  /* Waiting ACK */
  if (state == EnumState_S_PROCESSING)
    return aiPbMgrWaitAck();
  else
    return true;
}

/*---------------------------------------------------------------------------*/

void aiPbCmdSync(const reqMsg *req, respMsg *resp, void *param)
{
  resp->which_payload = respMsg_sync_tag;
  resp->payload.sync.version =
      EnumVersion_P_VERSION_MAJOR << 8 |
      EnumVersion_P_VERSION_MINOR;

  resp->payload.sync.capability = EnumCapability_CAP_FIXED_POINT;

#if defined(AI_PB_TEST) && (AI_PB_TEST == 1)
  resp->payload.sync.capability |= EnumCapability_CAP_SELF_TEST;
#endif

  if (param)
    resp->payload.sync.capability |= (uint32_t)param;

  aiPbMgrSendResp(req, resp, EnumState_S_IDLE);
}

void aiPbCmdSysInfo(const reqMsg *req, respMsg *resp, void *param)
{
  UNUSED(param);
  resp->which_payload = respMsg_sinfo_tag;
  resp->payload.sinfo.devid = HAL_GetDEVID();
#ifdef STM32MP1
  resp->payload.sinfo.sclock = HAL_RCC_GetSystemCoreClockFreq();
  resp->payload.sinfo.hclock = HAL_RCC_GetHCLK3Freq();
#else
  resp->payload.sinfo.sclock = HAL_RCC_GetSysClockFreq();
  resp->payload.sinfo.hclock = HAL_RCC_GetHCLKFreq();
#endif
  resp->payload.sinfo.cache = getFlashCacheConf();

  aiPbMgrSendResp(req, resp, EnumState_S_IDLE);
}


static void init_aibuffer_msg(const ai_buffer *aibuffer, aiBufferShapeMsg *msg)
{
  if ((!aibuffer) || (!msg))
    return;

#if defined(NO_X_CUBE_AI_RUNTIME) && NO_X_CUBE_AI_RUNTIME == 1
  struct ai_buffer_ext *from_ = (struct ai_buffer_ext *)aibuffer;
  msg->format = aiPbAiFmtToMsgFmt(from_->buffer.format);
  msg->channels = from_->buffer.channels;
  msg->height = from_->buffer.height;
  msg->width = from_->buffer.width;
  msg->n_batches = from_->buffer.n_batches;
  if (from_->buffer.meta_info) {
    msg->scale = from_->extra.scale;
    msg->zeropoint = from_->extra.zero_point;
  } else {
    msg->scale = 0.0f;
    msg->zeropoint = 0;
  }
#else
  const ai_buffer_meta_info *meta_info = AI_BUFFER_META_INFO(aibuffer);

  msg->format = aiPbAiFmtToMsgFmt(aibuffer->format);
  msg->channels = aibuffer->channels;
  msg->height = aibuffer->height;
  msg->width = aibuffer->width;
  msg->n_batches = aibuffer->n_batches;
  aiPbMgrSetMetaInfo(meta_info, 0, msg);
#endif
}

static bool nn_shape_w_cb(pb_ostream_t *stream, const pb_field_t *field,
    const ai_buffer *aibuffer, int maxw)
{
  aiBufferShapeMsg msg;

  for (int i = 0; i < maxw; i++) {
    if (!pb_encode_tag_for_field(stream, field))
      return false;

    init_aibuffer_msg(&aibuffer[i], &msg);

    if (!pb_encode_submessage(stream, aiBufferShapeMsg_fields, &msg))
      return false;
  }
  return true;
}

static bool nn_inputs_w_cb(pb_ostream_t *stream, const pb_field_t *field,
    void * const *arg)
{
  ai_network_report *report = (ai_network_report *)*arg;

  if (!report)
    return true;

  return nn_shape_w_cb(stream, field, &report->inputs[0], report->n_inputs);
}

static bool nn_outputs_w_cb(pb_ostream_t *stream, const pb_field_t *field,
    void * const *arg)
{
  ai_network_report *report = (ai_network_report *)*arg;

  if (!report)
    return true;

  return nn_shape_w_cb(stream, field, &report->outputs[0], report->n_outputs);
}

void aiPbStrCopy(const char *src, char *dst, uint32_t max)
{
  const char undef[] = "UNDEFINED";
  size_t l = strlen(src);

  if (l > max)
    l = max-1;

  if (!dst)
    return;

  if (src && l)
    memcpy(dst, src, l+1);
  else
    memcpy(dst, undef, strlen(undef)+1);
}

uint32_t aiPbVersionToUint32(const ai_platform_version *ver)
{
  if (!ver)
    return 0;

  return ver->major << 24 | ver->minor << 16
      | ver->micro << 8 | ver->reserved;
}

void aiPbMgrSendNNInfo(const reqMsg *req, respMsg *resp,
    EnumState state, const ai_network_report *nn)
{
  resp->which_payload = respMsg_ninfo_tag;

  aiPbStrCopy(nn->model_name,
      &resp->payload.ninfo.model_name[0],
      sizeof(resp->payload.ninfo.model_name));
  aiPbStrCopy(nn->model_signature,
      &resp->payload.ninfo.model_signature[0],
      sizeof(resp->payload.ninfo.model_signature));
  aiPbStrCopy(nn->model_datetime,
      &resp->payload.ninfo.model_datetime[0],
      sizeof(resp->payload.ninfo.model_datetime));
  aiPbStrCopy(nn->compile_datetime,
      &resp->payload.ninfo.compile_datetime[0],
      sizeof(resp->payload.ninfo.compile_datetime));
  aiPbStrCopy(nn->runtime_revision,
      &resp->payload.ninfo.runtime_revision[0],
      sizeof(resp->payload.ninfo.runtime_revision));
  aiPbStrCopy(nn->tool_revision,
      &resp->payload.ninfo.tool_revision[0],
      sizeof(resp->payload.ninfo.tool_revision));

  resp->payload.ninfo.n_inputs = nn->n_inputs;
  resp->payload.ninfo.n_outputs = nn->n_outputs;
  resp->payload.ninfo.n_nodes = nn->n_nodes;
  resp->payload.ninfo.n_macc = nn->n_macc;

  resp->payload.ninfo.signature = nn->signature;
  resp->payload.ninfo.api_version =
      aiPbVersionToUint32(&nn->api_version);
  resp->payload.ninfo.interface_api_version =
      aiPbVersionToUint32(&nn->interface_api_version);
  resp->payload.ninfo.runtime_version =
      aiPbVersionToUint32(&nn->runtime_version);
  resp->payload.ninfo.tool_version =
      aiPbVersionToUint32(&nn->tool_version);
  resp->payload.ninfo.tool_api_version =
      aiPbVersionToUint32(&nn->tool_api_version);

  init_aibuffer_msg(&nn->activations, &resp->payload.ninfo.activations);
  init_aibuffer_msg(&nn->params, &resp->payload.ninfo.weights);

  resp->payload.ninfo.inputs.funcs.encode = nn_inputs_w_cb;
  resp->payload.ninfo.inputs.arg = (void *)nn;

  resp->payload.ninfo.outputs.funcs.encode = nn_outputs_w_cb;
  resp->payload.ninfo.outputs.arg = (void *)nn;

  aiPbMgrSendResp(req, resp, state);
}


/*---------------------------------------------------------------------------*/

#if defined(AI_PB_TEST) && (AI_PB_TEST == 1)

#include <stdlib.h>
#include <stdio.h>

#define _MAX_BUFF_SIZE_ (1024) /* min 256 see test function */
static ai_float buffer_test[_MAX_BUFF_SIZE_];
static ai_buffer ai_buffer_test;

void aiPbTestRstAiBuffer(ai_buffer *buffer)
{
  buffer->channels = _MAX_BUFF_SIZE_;
  buffer->height = buffer->width = buffer->n_batches = 1;
  buffer->data = (ai_handle)buffer_test;
  buffer->format = AI_BUFFER_FORMAT_FLOAT;

  for (int i=0; i<_MAX_BUFF_SIZE_; i++) {
    ai_float value = 2.0f * (ai_float) rand() / (ai_float) RAND_MAX - 1.0f;
    buffer_test[i] = value;
  }
}

/* https://rosettacode.org/wiki/CRC-32#Python */
__STATIC_INLINE uint32_t rc_crc32(uint32_t crc, const char *buf, size_t len)
{
  static uint32_t table[256];
  static int have_table = 0;
  uint32_t rem;
  uint8_t octet;
  int i, j;
  const char *p, *q;

  /* This check is not thread safe; there is no mutex. */
  if (have_table == 0) {
    /* Calculate CRC table. */
    for (i = 0; i < 256; i++) {
      rem = i;  /* remainder from polynomial division */
      for (j = 0; j < 8; j++) {
        if (rem & 1) {
          rem >>= 1;
          rem ^= 0xedb88320;
        } else
          rem >>= 1;
      }
      table[i] = rem;
    }
    have_table = 1;
  }

  crc = ~crc;
  q = buf + len;
  for (p = buf; p < q; p++) {
    octet = *p;  /* Cast to unsigned octet. */
    crc = (crc >> 8) ^ table[(crc & 0xff) ^ octet];
  }
  return ~crc;
}

static uint32_t aiPbTestCalculateHash(const ai_buffer *aibuf)
{
  uint32_t crc = 0;
  if (!aibuf || !aibuf->data)
    return 0;

  const char *buf = (const char *)aibuf->data;
  size_t len = aiPbAiBufferSize(aibuf);

  if ((aibuf->format == AI_BUFFER_FORMAT_FLOAT) ||
      (aibuf->format == AI_BUFFER_FORMAT_NONE))
    len <<= 2;
  else if ((aibuf->format == AI_BUFFER_FORMAT_Q15) ||
      (aibuf->format == AI_BUFFER_FORMAT_S16) ||
      (aibuf->format == AI_BUFFER_FORMAT_U16) )
    len <<= 1;

  crc = rc_crc32(0, buf, len);

  return crc;
}

/* TEST - Receive and/or Send a simple buffer */
static void aiPbTestCmdSimpleBuffer(const reqMsg *req, respMsg *resp,
    void *param)
{
  uint32_t sb;
  uint32_t hash;
  uint32_t stest = req->param & 0xFFFF;
  bool chash = true;

  UNUSED(param);

  aiPbTestRstAiBuffer(&ai_buffer_test);
  sb = ai_buffer_test.channels;

  if (req->param >> 16) {
    if ((req->param >> 16) < ai_buffer_test.channels)
      sb = req->param >> 16;
    chash = false;
  } else {
    sb = 256;
  }

  if (stest == 100) {
    /* Test download - normal/nominal processing operation */
    /*   ACK with expected buffer size                     */
    /*   Download a FLOAT buffer                           */
    /*   emulate a processing                              */
    /*   ACK operation DONE                                */
    ai_buffer_test.channels = sb;

    /* Send a ACK with the expected number of items */
    aiPbMgrSendAck(req, resp, EnumState_S_WAITING, sb, EnumError_E_NONE);

    /* Wait & Receive the buffer */
    aiPbMgrReceiveAiBuffer3(req, resp, EnumState_S_PROCESSING,
        &ai_buffer_test);

    /* Emulate processing */
    HAL_Delay(300);  /* 300ms */

    /* Send ACK/DONE (operation DONE) */
    aiPbMgrSendAck(req, resp, EnumState_S_DONE, 0, EnumError_E_NONE);

  } else if (stest == 101) {
    /* Test download - quick processing operation          */
    /*   ACK with expected buffer size                     */
    /*   Download a FLOAT buffer                           */
    /*   ACK operation DONE                                */
    sb = 512;
    ai_buffer_test.channels = sb;

    /* Send a ACK with the expected number of items */
    aiPbMgrSendAck(req, resp, EnumState_S_WAITING, sb,
        EnumError_E_NONE);

    /* Wait input buffer and send operation DONE) */
    aiPbMgrReceiveAiBuffer3(req, resp, EnumState_S_DONE,
        &ai_buffer_test);

  } else if (stest == 102) {
    /* Test upload - without data (only shape) */
    sb = 256;
    ai_buffer_test.channels = sb;

    /* Send a ACK with the number of items */
    aiPbMgrSendAck(req, resp, EnumState_S_PROCESSING, sb,
        EnumError_E_NONE);

    /* Send buffer and operation DONE */
    aiPbMgrSendAiBuffer4(req, resp, EnumState_S_DONE,
        PB_BUFFER_TYPE_SEND_WITHOUT_DATA,
        req->param, ~req->param, &ai_buffer_test, 0.0f, 0);

  } else if (stest == 103) {
    /* Test upload - without data (shape == 0) */
    sb = 0;
    ai_buffer_test.channels = sb;

    /* Send a ACK with the number of items */
    aiPbMgrSendAck(req, resp, EnumState_S_PROCESSING, sb,
        EnumError_E_NONE);

    /* Send buffer and operation DONE */
    aiPbMgrSendAiBuffer4(req, resp, EnumState_S_DONE,
        0, 0, 0, &ai_buffer_test, 0.0f, 0);

  } else if (stest == 130) {
    /* Test download - quick processing operation */
    ai_buffer_test.channels = sb;

    if (req->opt)
      ai_buffer_test.format = aiPbMsgFmtToAiFmt(req->opt);


    if (chash) {
      /* Calculate the hash */
      hash = aiPbTestCalculateHash(&ai_buffer_test);

      /* Send a ACK with the expected number of items */
      aiPbMgrSendAck(req, resp, EnumState_S_PROCESSING, hash,
          EnumError_E_NONE);
    } else {
      aiPbMgrSendAck(req, resp, EnumState_S_PROCESSING, sb,
          EnumError_E_NONE);
    }

    /* Send the buffer */
    aiPbMgrSendAiBuffer4(req, resp, EnumState_S_DONE,
        0, 0, 0, &ai_buffer_test, 0.0f, 0);

  } else if (stest == 140) {
    /* Test download - quick processing operation */
    ai_buffer_test.channels = sb;

    if (req->opt)
      ai_buffer_test.format = aiPbMsgFmtToAiFmt(req->opt);

    /* Send a ACK with the expected number of items */
    aiPbMgrSendAck(req, resp, EnumState_S_WAITING, sb, EnumError_E_NONE);

    if (chash) {
      /* Wait input buffer  */
      aiPbMgrReceiveAiBuffer3(req, resp, EnumState_S_PROCESSING,
          &ai_buffer_test);

      /* Calculate the hash */
      hash = aiPbTestCalculateHash(&ai_buffer_test);

      /* Send the result (param part of the ACK msg) */
      aiPbMgrSendAck(req, resp, EnumState_S_DONE, hash,
          EnumError_E_NONE);
    } else {
      aiPbMgrReceiveAiBuffer3(req, resp,
          EnumState_S_DONE,
          &ai_buffer_test);
    }

  } else if (stest == 200) {
    /* Test download/upload buffer */
    ai_buffer_test.channels = sb;

    /* Format of the uploaded buffer is param dependent */
    if (req->opt)
      ai_buffer_test.format = aiPbMsgFmtToAiFmt(req->opt);

    /* Send a ACK with the expected number of item */
    aiPbMgrSendAck(req, resp, EnumState_S_WAITING, sb, EnumError_E_NONE);

    /* Wait input buffer */
    aiPbMgrReceiveAiBuffer3(req, resp, EnumState_S_PROCESSING,
        &ai_buffer_test);

    /* Process data */
    if (ai_buffer_test.format == AI_BUFFER_FORMAT_FLOAT) {
      for (uint32_t i=0; i<sb; i++) {
        ai_float v = ((ai_float *)ai_buffer_test.data)[i];
        ((ai_float *)ai_buffer_test.data)[i] = v * 2.0;
      }
    }

    /* Send modified buffer and DONE operation */
    aiPbMgrSendAiBuffer4(req, resp, EnumState_S_DONE,
        0, 0, 0, &ai_buffer_test, 0.0f, 0);

  } else {
    aiPbMgrSendAck(req, resp, EnumState_S_ERROR,
        EnumError_E_INVALID_PARAM, EnumError_E_INVALID_PARAM);
  }
}

/* TEST CMD */
void aiPbTestCmd(const reqMsg *req, respMsg *resp, void *param)
{
  uint32_t stest = req->param & 0xFFFF;
  if (stest == 0) {
    aiPbMgrSendAck(req, resp, EnumState_S_DONE, 0, EnumError_E_NONE);
  } else if (stest == 1) {
    if (!req->name[0]) {
      aiPbMgrSendAck(req, resp, EnumState_S_ERROR,
          EnumError_E_INVALID_PARAM, EnumError_E_INVALID_PARAM);
    } else {
      aiPbMgrSendAck(req, resp, EnumState_S_DONE,
          strlen(req->name), EnumError_E_NONE);
    }
  } else if (stest == 2) {
    const char *str = "Hello..";
    aiPbMgrSendLog(req, resp, EnumState_S_DONE, 1, str);
  } else if (stest == 3) {
    /* Time out test */
    HAL_Delay(500);
  } else if (stest == 4) {
    char str[20] = "Bye Bye..";
    for (int i=0; i<5;i++) {
      str[0] = 'A' + i;
      aiPbMgrSendLog(req, resp, EnumState_S_PROCESSING, i, str);
    }
    aiPbMgrSendAck(req, resp, EnumState_S_DONE, 0, EnumError_E_NONE);
  } else if (stest == 10) {
    aiPbMgrSendAck(req, resp, EnumState_S_PROCESSING, req->param,
        EnumError_E_NONE);
    aiPbMgrSendAck(req, resp, EnumState_S_DONE, req->param,
        EnumError_E_NONE);
  } else if (stest == 11) {
    int i;
    for (i=0; i<200; i++)
      aiPbMgrSendAck(req, resp, EnumState_S_PROCESSING, i,
          EnumError_E_NONE);
    aiPbMgrSendAck(req, resp, EnumState_S_DONE, req->param,
        EnumError_E_NONE);
  } else if ((stest >= 100) && (stest < 400)) {
    aiPbTestCmdSimpleBuffer(req, resp, param);
  } else {
    aiPbMgrSendAck(req, resp, EnumState_S_DONE, req->param + 1,
        EnumError_E_NONE);
  }
}

#endif
