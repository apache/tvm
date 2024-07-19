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

#include "audio_rx_slot.h"

AudioRxSlot::AudioRxSlot(int bck_pin, int ws_pin, int data_pin, int channel_pin) {
  bck_pin_ = (gpio_num_t)bck_pin;
  ws_pin_ = (gpio_num_t)ws_pin;
  din_pin_ = (gpio_num_t)data_pin;
  channel_pin_ = (channel_pin < 0) ? GPIO_NUM_NC : (gpio_num_t)channel_pin;
  mclk_pin_ = GPIO_NUM_NC;
  dout_pin_ = GPIO_NUM_NC;
  rx_handle_ = NULL;
  bytes_width_ = A_BYTES_PER_SAMPLE_DATA_32BIT;
  fragment_size_ = 1 * bytes_width_;
  buffer_size_ = 1 * bytes_width_;
}

AudioRxSlot::~AudioRxSlot() { end(); }

// Disable interface
void AudioRxSlot::end() {
  if (rx_handle_ != NULL) {
    i2s_channel_disable(rx_handle_);
    i2s_del_channel(rx_handle_);
    rx_handle_ = NULL;
  }
}

// Interface setup and start
int AudioRxSlot::begin(uint32_t sample_rate, IOPlan& io_plan, int port_number, eBitWidth bit_width,
                       eSlotType slot_type, eSlotMode slot_mode, eBitShift bit_shift) {
  int errors = 0;

  if (channel_pin_ != GPIO_NUM_NC) {
    gpio_config_t io_conf;
    io_conf.intr_type = GPIO_INTR_DISABLE;
    io_conf.mode = GPIO_MODE_OUTPUT;
    io_conf.pin_bit_mask = (1ULL << channel_pin_);
    io_conf.pull_down_en = GPIO_PULLDOWN_DISABLE;
    io_conf.pull_up_en = GPIO_PULLUP_DISABLE;
    gpio_config(&io_conf);
    gpio_set_level(channel_pin_, (slot_type == stStdRight) ? 1 : 0);
  }

  i2s_port_t i2s_port_number = I2S_NUM_0;
  switch (port_number) {
    case 0:
      break;
    default:
      errors++;
      break;
  }

  i2s_data_bit_width_t i2s_bit_width = I2S_DATA_BIT_WIDTH_8BIT;
  switch (bit_width) {
    case bwBitWidth8:
      i2s_bit_width = I2S_DATA_BIT_WIDTH_8BIT;
      bytes_width_ = A_BYTES_PER_SAMPLE_DATA_8BIT;
      break;
    case bwBitWidth16:
      i2s_bit_width = I2S_DATA_BIT_WIDTH_16BIT;
      bytes_width_ = A_BYTES_PER_SAMPLE_DATA_16BIT;
      break;
    case bwBitWidth24:
      i2s_bit_width = I2S_DATA_BIT_WIDTH_24BIT;
      bytes_width_ = A_BYTES_PER_SAMPLE_DATA_24BIT;
      break;
    case bwBitWidth32:
      i2s_bit_width = I2S_DATA_BIT_WIDTH_32BIT;
      bytes_width_ = A_BYTES_PER_SAMPLE_DATA_32BIT;
      break;
    default:
      errors++;
      break;
  }
  fragment_size_ = io_plan.read_fragment_size;
  buffer_size_ = io_plan.polling_buffer_size;

  i2s_std_slot_mask_t i2s_slot_mask = I2S_STD_SLOT_RIGHT;
  switch (slot_type) {
    case stStdLeft:
      i2s_slot_mask = I2S_STD_SLOT_LEFT;
      break;
    case stStdRight:
      i2s_slot_mask = I2S_STD_SLOT_RIGHT;
      break;
    case stStdBoth:
      i2s_slot_mask = I2S_STD_SLOT_BOTH;
      break;
    default:
      errors++;
      break;
  }

  i2s_slot_mode_t i2s_slot_mode = I2S_SLOT_MODE_MONO;
  switch (slot_mode) {
    case smStereo:
      i2s_slot_mode = I2S_SLOT_MODE_STEREO;
      break;
    case smMono:
      i2s_slot_mode = I2S_SLOT_MODE_MONO;
      break;
    default:
      errors++;
      break;
  }

  bool i2s_bit_shift = true;
  switch (bit_shift) {
    case bsDisable:
      i2s_bit_shift = false;
      break;
    case bsEnable:
      i2s_bit_shift = true;
      break;
    default:
      errors++;
      break;
  }

  i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(i2s_port_number, I2S_ROLE_MASTER);
  chan_cfg.dma_desc_num = io_plan.dma_desc_num;
  chan_cfg.dma_frame_num = io_plan.dma_frame_num;
  i2s_std_config_t rx_std_cfg = {
      .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(sample_rate),
      .slot_cfg = I2S_STD_MSB_SLOT_DEFAULT_CONFIG(i2s_bit_width, i2s_slot_mode),
      .gpio_cfg =
          {
              .mclk = mclk_pin_,
              .bclk = bck_pin_,
              .ws = ws_pin_,
              .dout = dout_pin_,
              .din = din_pin_,
              .invert_flags =
                  {
                      .mclk_inv = false,
                      .bclk_inv = false,
                      .ws_inv = false,
                  },
          },
  };
  if (i2s_bit_width == I2S_DATA_BIT_WIDTH_24BIT) {
    rx_std_cfg.clk_cfg.mclk_multiple = I2S_MCLK_MULTIPLE_384;
  }
  rx_std_cfg.slot_cfg.slot_mode = i2s_slot_mode;
  rx_std_cfg.slot_cfg.slot_mask = i2s_slot_mask;
  rx_std_cfg.slot_cfg.bit_shift = i2s_bit_shift;

  if (errors == 0) {
    TST_ESP_OK(i2s_new_channel(&chan_cfg, NULL, &rx_handle_));
    TST_ESP_OK(i2s_channel_init_std_mode(rx_handle_, &rx_std_cfg));
    TST_ESP_OK(i2s_channel_enable(rx_handle_));
  }

  return errors;
}

size_t AudioRxSlot::read(void* fragment) {
  size_t bytes_read = 0;
  i2s_channel_read(rx_handle_, fragment, fragment_size_, &bytes_read, portMAX_DELAY);
  return bytes_read;
}

size_t AudioRxSlot::read(void* fragment, size_t bytes) {
  size_t bytes_read = 0;
  i2s_channel_read(rx_handle_, fragment, (bytes / bytes_width_) * bytes_width_, &bytes_read,
                   portMAX_DELAY);
  return bytes_read;
}

/*!
 * \brief
 * The driver buffer works with a DMA channel with cyclic descriptors. For optimal operation of this
 * logic, it is necessary to optimally configure the driver's DMA channel for the conditions of the
 * task. I.e., it is necessary to determine the number of blocks (number of descriptors) and the
 * size of each block (number of frames per block). These two parameters are required to configure
 * the driver's DMA channel and are calculated in plan() function. In turn, the number of DMA blocks
 * and the size of each DMA block affect the optimal transaction size of the read operation
 * (see 'fragment time') and the size of the entire buffer (see 'polling time'), which are also
 * calculated in plan() function. So, according to the parameters of the task, plan() function
 * calculates the 4 parameters:
 *   - number of blocks of the driver's DMA channel (used to configure the driver),
 *   - number of frames in each such block (used to configure the driver),
 *   - size of the portion of the read operation (corresponds to 'fragment time', but matching is
 *     optional exact; this size is recommended to be used in read operations),
 *   - full size of the driver buffer (corresponds to 'polling time', but the correspondence is not
 *     necessarily exact; max number of data that can be accumulated between two reads).
 * It is important to take into account here:
 *   - the max block size of the DMA channel is 4092 bytes;
 *   - the DMA channel is focused on operations with 16-bit data (although we have 8–bit interface);
 *   - frames must fit completely in each block;
 *   - it is necessary to get at least 2 descriptors (2 DMA blocks), and a lot descriptors will
 *     require a lot of memory (there is an artificial upper limit of 64).
 */
void AudioRxSlot::plan(uint32_t sample_rate, uint32_t data_bit_width, uint32_t slot_num,
                       uint32_t polling_time_msec, uint32_t fragment_time_msec, IOPlan& io_plan) {
#define I2S_DMA_BUFFER_MAX_SIZE 4092
#define I2S_DMA_DESC_MAX_CNT 64
#define PLAN_LAST_STAGE_ID 5

  const uint32_t bytes_per_sample_x = PARTS(data_bit_width, 8);
  const uint32_t bytes_per_frame_x = bytes_per_sample_x * slot_num;

  const uint32_t bytes_per_sample = PARTS(data_bit_width, 16) * 2;
  const uint32_t bytes_per_frame = bytes_per_sample * slot_num;

  if (!fragment_time_msec)
    fragment_time_msec = polling_time_msec;
  else if (fragment_time_msec >= polling_time_msec)
    polling_time_msec = fragment_time_msec;
  else {
    if ((polling_time_msec / fragment_time_msec) < 2)
      polling_time_msec = fragment_time_msec * 2;
    else
      polling_time_msec = PARTS(polling_time_msec, fragment_time_msec) * fragment_time_msec;
  }

  io_plan.read_fragment_size = PARTS(fragment_time_msec * sample_rate, 1000) * bytes_per_frame_x;
  uint32_t dma_buffer_size = 0;
  uint32_t read_fragment_size_x = io_plan.read_fragment_size;
  int stage = 0;
  while (stage <= PLAN_LAST_STAGE_ID) {
    uint32_t step = (stage < PLAN_LAST_STAGE_ID) ? read_fragment_size_x : 1;
    uint32_t diff = -1;
    uint32_t dma_buffer_size_x = (I2S_DMA_BUFFER_MAX_SIZE / step) * step;
    dma_buffer_size = 0;
    io_plan.polling_buffer_size =
        PARTS(polling_time_msec, fragment_time_msec) * read_fragment_size_x;
    while (dma_buffer_size_x >= bytes_per_frame) {
      if (!(dma_buffer_size_x % bytes_per_frame) && !(dma_buffer_size_x % bytes_per_frame_x)) {
        io_plan.dma_desc_num = PARTS(io_plan.polling_buffer_size, dma_buffer_size_x);
        if (io_plan.dma_desc_num > I2S_DMA_DESC_MAX_CNT) {
          if (diff == -1) dma_buffer_size = dma_buffer_size_x;
          break;
        } else if (io_plan.dma_desc_num < 2) {
          dma_buffer_size = dma_buffer_size_x;
        } else {
          uint32_t diff_x = io_plan.dma_desc_num * dma_buffer_size_x - io_plan.polling_buffer_size;
          if (diff_x < diff) {
            dma_buffer_size = dma_buffer_size_x;
            if (!diff_x) break;
            diff = diff_x;
          }
        }
      }
      dma_buffer_size_x -= step;
    }  // while

    if (dma_buffer_size) io_plan.dma_desc_num = PARTS(io_plan.polling_buffer_size, dma_buffer_size);

    if (stage == PLAN_LAST_STAGE_ID) break;

    stage++;
    if (dma_buffer_size && (io_plan.dma_desc_num <= I2S_DMA_DESC_MAX_CNT)) {
      io_plan.read_fragment_size = read_fragment_size_x;
      break;
    }
    read_fragment_size_x = (stage < PLAN_LAST_STAGE_ID) ? read_fragment_size_x + bytes_per_frame_x
                                                        : io_plan.read_fragment_size;
    continue;
  }

  if (io_plan.dma_desc_num < 2) io_plan.dma_desc_num = 2;
  io_plan.dma_frame_num = dma_buffer_size / bytes_per_frame;
  io_plan.polling_buffer_size = io_plan.dma_desc_num * dma_buffer_size;
}

// Return read transaction size
size_t AudioRxSlot::get_fragment_size() { return fragment_size_; }

// Return polling buffer size
size_t AudioRxSlot::get_buffer_size() { return buffer_size_; }
