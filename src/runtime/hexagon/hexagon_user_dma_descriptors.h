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

#ifndef TVM_RUNTIME_HEXAGON_HEXAGON_USER_DMA_DESCRIPTORS_H_
#define TVM_RUNTIME_HEXAGON_HEXAGON_USER_DMA_DESCRIPTORS_H_

namespace tvm {
namespace runtime {
namespace hexagon {

// NOTE: Using 2D descriptor size even for 1D descriptors
#define DMA_DESC_2D_SIZE 32

// DMA State
// desc[0][3:0]
#define DESC_STATE_MASK 0x0000000F
#define DESC_STATE_SHIFT 0
#define DESC_STATE_READY 0

// desc[0][31:4]
// Descriptors addresses must be (minimum) 16 byte aligned
// -> Lower 4 bits masked to clear DMA Status
// -> But, descriptor address is not shifted
#define DESC_NEXT_MASK 0xFFFFFFF0
#define DESC_NEXT_SHIFT 0

// desc[1][23:0]
#define DESC_LENGTH_MASK 0x00FFFFFF
#define DESC_LENGTH_SHIFT 0

// desc[1][25:24]
#define DESC_DESCTYPE_MASK 0x03000000
#define DESC_DESCTYPE_SHIFT 24
#define DESC_DESCTYPE_1D 0
#define DESC_DESCTYPE_2D 1

// TODO(Straw): Definition?  Not in the spec.
// desc[1][26]
#define DESC_DSTCOMP_MASK 0x04000000
#define DESC_DSTCOMP_SHIFT 26
// desc[1][27]
#define DESC_SRCCOMP_MASK 0x08000000
#define DESC_SRCCOMP_SHIFT 27
#define DESC_COMP_NONE 0
#define DESC_COMP_DLBC 1

// desc[1][28]
#define DESC_BYPASSDST_MASK 0x10000000
#define DESC_BYPASSDST_SHIFT 28
// desc[1][29]
#define DESC_BYPASSSRC_MASK 0x20000000
#define DESC_BYPASSSRC_SHIFT 29
#define DESC_BYPASS_OFF 0
#define DESC_BYPASS_ON 1

// desc[1][30]
#define DESC_ORDER_MASK 0x40000000
#define DESC_ORDER_SHIFT 30
#define DESC_ORDER_NOORDER 0
#define DESC_ORDER_ORDER 1

// desc[1][31]
#define DESC_DONE_MASK 0x80000000
#define DESC_DONE_SHIFT 31
#define DESC_DONE_INCOMPLETE 0
#define DESC_DONE_COMPLETE 1

// desc[2]
#define DESC_SRC_MASK 0xFFFFFFFF
#define DESC_SRC_SHIFT 0

// desc[3]
#define DESC_DST_MASK 0xFFFFFFFF
#define DESC_DST_SHIFT 0

// desc[4][25:24]
#define DESC_CACHEALLOC_MASK 0x03000000
#define DESC_CACHEALLOC_SHIFT 24
#define DESC_CACHEALLOC_NONE 0
#define DESC_CACHEALLOC_WRITEONLY 1
#define DESC_CACHEALLOC_READONLY 2
#define DESC_CACHEALLOC_READWRITE 3

// TODO(Straw): Definition?  Not in the spec.
// desc[4][31:28]
#define DESC_PADDING_MASK 0xF0000000
#define DESC_PADDING_SHIFT 28

// desc[5][15:0]
#define DESC_ROIWIDTH_MASK 0x0000FFFF
#define DESC_ROIWIDTH_SHIFT 0

// desc[5][31:16]
#define DESC_ROIHEIGHT_MASK 0xFFFF0000
#define DESC_ROIHEIGHT_SHIFT 16

// desc[6][15:0]
#define DESC_SRCSTRIDE_MASK 0x0000FFFF
#define DESC_SRCSTRIDE_SHIFT 0

// desc[6][31:16]
#define DESC_DSTSTRIDE_MASK 0xFFFF0000
#define DESC_DSTSTRIDE_SHIFT 16

// desc[7][15:0]
#define DESC_SRCWIDTHOFFSET_MASK 0x0000FFFF
#define DESC_SRCWIDTHOFFSET_SHIFT 0

// desc[7][31:16]
#define DESC_DSTWIDTHOFFSET_MASK 0xFFFF0000
#define DESC_DSTWIDTHOFFSET_SHIFT 16

#define DMA_NULL_PTR 0

/**************************/
/* 1D (linear) descriptor */
/**************************/
struct dma_desc_1d_t {
  unsigned int next_state;
  unsigned int done_order_bypass_comp_desctype_length;
  unsigned int src;
  unsigned int dst;
};

/***********************/
/* 2D (box) descriptor */
/***********************/
struct dma_desc_2d_t {
  unsigned int next_state;
  unsigned int done_order_bypass_comp_desctype_length;
  unsigned int src;
  unsigned int dst;
  unsigned int allocation_padding;
  unsigned int roiheight_roiwidth;
  unsigned int dststride_srcstride;
  unsigned int dstwidthoffset_srcwidthoffset;
};

// desc[0][3:0]
inline void dma_desc_set_state(void* dma_desc_ptr, unsigned int v) {
  dma_desc_1d_t* dma_desc_1d_ptr = reinterpret_cast<dma_desc_1d_t*>(dma_desc_ptr);
  (dma_desc_1d_ptr->next_state) &= ~DESC_STATE_MASK;
  (dma_desc_1d_ptr->next_state) |= ((v << DESC_STATE_SHIFT) & DESC_STATE_MASK);
}

// desc[0][31:4]
inline void dma_desc_set_next(void* dma_desc_ptr, unsigned int v) {
  dma_desc_1d_t* dma_desc_1d_ptr = reinterpret_cast<dma_desc_1d_t*>(dma_desc_ptr);
  (dma_desc_1d_ptr->next_state) &= ~DESC_NEXT_MASK;
  (dma_desc_1d_ptr->next_state) |= ((v << DESC_NEXT_SHIFT) & DESC_NEXT_MASK);
}

// desc[1][23:0]
inline void dma_desc_set_length(void* dma_desc_ptr, unsigned int v) {
  dma_desc_1d_t* dma_desc_1d_ptr = reinterpret_cast<dma_desc_1d_t*>(dma_desc_ptr);
  (dma_desc_1d_ptr->done_order_bypass_comp_desctype_length) &= ~DESC_LENGTH_MASK;
  (dma_desc_1d_ptr->done_order_bypass_comp_desctype_length) |=
      ((v << DESC_LENGTH_SHIFT) & DESC_LENGTH_MASK);
}

// desc[1][25:24]
inline void dma_desc_set_desctype(void* dma_desc_ptr, unsigned int v) {
  dma_desc_1d_t* dma_desc_1d_ptr = reinterpret_cast<dma_desc_1d_t*>(dma_desc_ptr);
  (dma_desc_1d_ptr->done_order_bypass_comp_desctype_length) &= ~DESC_DESCTYPE_MASK;
  (dma_desc_1d_ptr->done_order_bypass_comp_desctype_length) |=
      ((v << DESC_DESCTYPE_SHIFT) & DESC_DESCTYPE_MASK);
}

// TODO(Straw): Definition?  Not in the spec.
// desc[1][26]
inline void dma_desc_set_dstcomp(void* dma_desc_ptr, unsigned int v) {
  dma_desc_1d_t* dma_desc_1d_ptr = reinterpret_cast<dma_desc_1d_t*>(dma_desc_ptr);
  (dma_desc_1d_ptr->done_order_bypass_comp_desctype_length) &= ~DESC_DSTCOMP_MASK;
  (dma_desc_1d_ptr->done_order_bypass_comp_desctype_length) |=
      ((v << DESC_DSTCOMP_SHIFT) & DESC_DSTCOMP_MASK);
}

// TODO(Straw): Definition?  Not in the spec.
// desc[1][27]
inline void dma_desc_set_srccomp(void* dma_desc_ptr, unsigned int v) {
  dma_desc_1d_t* dma_desc_1d_ptr = reinterpret_cast<dma_desc_1d_t*>(dma_desc_ptr);
  (dma_desc_1d_ptr->done_order_bypass_comp_desctype_length) &= ~DESC_SRCCOMP_MASK;
  (dma_desc_1d_ptr->done_order_bypass_comp_desctype_length) |=
      ((v << DESC_SRCCOMP_SHIFT) & DESC_SRCCOMP_MASK);
}

// desc[1][28]
inline void dma_desc_set_bypassdst(void* dma_desc_ptr, unsigned int v) {
  dma_desc_1d_t* dma_desc_1d_ptr = reinterpret_cast<dma_desc_1d_t*>(dma_desc_ptr);
  (dma_desc_1d_ptr->done_order_bypass_comp_desctype_length) &= ~DESC_BYPASSDST_MASK;
  (dma_desc_1d_ptr->done_order_bypass_comp_desctype_length) |=
      ((v << DESC_BYPASSDST_SHIFT) & DESC_BYPASSDST_MASK);
}

// desc[1][29]
inline void dma_desc_set_bypasssrc(void* dma_desc_ptr, unsigned int v) {
  dma_desc_1d_t* dma_desc_1d_ptr = reinterpret_cast<dma_desc_1d_t*>(dma_desc_ptr);
  (dma_desc_1d_ptr->done_order_bypass_comp_desctype_length) &= ~DESC_BYPASSSRC_MASK;
  (dma_desc_1d_ptr->done_order_bypass_comp_desctype_length) |=
      ((v << DESC_BYPASSSRC_SHIFT) & DESC_BYPASSSRC_MASK);
}

// desc[1][30]
inline void dma_desc_set_order(void* dma_desc_ptr, unsigned int v) {
  dma_desc_1d_t* dma_desc_1d_ptr = reinterpret_cast<dma_desc_1d_t*>(dma_desc_ptr);
  (dma_desc_1d_ptr->done_order_bypass_comp_desctype_length) &= ~DESC_ORDER_MASK;
  (dma_desc_1d_ptr->done_order_bypass_comp_desctype_length) |=
      ((v << DESC_ORDER_SHIFT) & DESC_ORDER_MASK);
}

// desc[1][31]
inline void dma_desc_set_done(void* dma_desc_ptr, unsigned int v) {
  dma_desc_1d_t* dma_desc_1d_ptr = reinterpret_cast<dma_desc_1d_t*>(dma_desc_ptr);
  (dma_desc_1d_ptr->done_order_bypass_comp_desctype_length) &= ~DESC_DONE_MASK;
  (dma_desc_1d_ptr->done_order_bypass_comp_desctype_length) |=
      ((v << DESC_DONE_SHIFT) & DESC_DONE_MASK);
}

// desc[1][31]
inline unsigned int dma_desc_get_done(void* dma_desc_ptr) {
  dma_desc_1d_t* dma_desc_1d_ptr = reinterpret_cast<dma_desc_1d_t*>(dma_desc_ptr);
  return (((dma_desc_1d_ptr->done_order_bypass_comp_desctype_length) & DESC_DONE_MASK) >>
          DESC_DONE_SHIFT);
}

// desc[2]
inline void dma_desc_set_src(void* dma_desc_ptr, unsigned int v) {
  dma_desc_1d_t* dma_desc_1d_ptr = reinterpret_cast<dma_desc_1d_t*>(dma_desc_ptr);
  (dma_desc_1d_ptr->src) &= ~DESC_SRC_MASK;
  (dma_desc_1d_ptr->src) |= ((v << DESC_SRC_SHIFT) & DESC_SRC_MASK);
}

// desc[3]
inline void dma_desc_set_dst(void* dma_desc_ptr, unsigned int v) {
  dma_desc_1d_t* dma_desc_1d_ptr = reinterpret_cast<dma_desc_1d_t*>(dma_desc_ptr);
  (dma_desc_1d_ptr->dst) &= ~DESC_DST_MASK;
  (dma_desc_1d_ptr->dst) |= ((v << DESC_DST_SHIFT) & DESC_DST_MASK);
}

// desc[4][25:24]
inline void dma_desc_set_cachealloc(void* dma_desc_ptr, unsigned int v) {
  dma_desc_2d_t* dma_desc_2d_ptr = reinterpret_cast<dma_desc_2d_t*>(dma_desc_ptr);
  (dma_desc_2d_ptr->allocation_padding) &= ~DESC_CACHEALLOC_MASK;
  (dma_desc_2d_ptr->allocation_padding) |= ((v << DESC_CACHEALLOC_SHIFT) & DESC_CACHEALLOC_MASK);
}

// TODO(Straw): Definition?  Not in the spec.
// desc[4][31:28]
inline void dma_desc_set_padding(void* dma_desc_ptr, unsigned int v) {
  dma_desc_2d_t* dma_desc_2d_ptr = reinterpret_cast<dma_desc_2d_t*>(dma_desc_ptr);
  (dma_desc_2d_ptr->allocation_padding) &= ~DESC_PADDING_MASK;
  (dma_desc_2d_ptr->allocation_padding) |= ((v << DESC_PADDING_SHIFT) & DESC_PADDING_MASK);
}

// desc[5][15:0]
inline void dma_desc_set_roiwidth(void* dma_desc_ptr, unsigned int v) {
  dma_desc_2d_t* dma_desc_2d_ptr = reinterpret_cast<dma_desc_2d_t*>(dma_desc_ptr);
  (dma_desc_2d_ptr->roiheight_roiwidth) &= ~DESC_ROIWIDTH_MASK;
  (dma_desc_2d_ptr->roiheight_roiwidth) |= ((v << DESC_ROIWIDTH_SHIFT) & DESC_ROIWIDTH_MASK);
}

// desc[5][31:16]
inline void dma_desc_set_roiheight(void* dma_desc_ptr, unsigned int v) {
  dma_desc_2d_t* dma_desc_2d_ptr = reinterpret_cast<dma_desc_2d_t*>(dma_desc_ptr);
  (dma_desc_2d_ptr->roiheight_roiwidth) &= ~DESC_ROIHEIGHT_MASK;
  (dma_desc_2d_ptr->roiheight_roiwidth) |= ((v << DESC_ROIHEIGHT_SHIFT) & DESC_ROIHEIGHT_MASK);
}

// desc[6][15:0]
inline void dma_desc_set_srcstride(void* dma_desc_ptr, unsigned int v) {
  dma_desc_2d_t* dma_desc_2d_ptr = reinterpret_cast<dma_desc_2d_t*>(dma_desc_ptr);
  (dma_desc_2d_ptr->dststride_srcstride) &= ~DESC_SRCSTRIDE_MASK;
  (dma_desc_2d_ptr->dststride_srcstride) |= ((v << DESC_SRCSTRIDE_SHIFT) & DESC_SRCSTRIDE_MASK);
}

// desc[6][31:16]
inline void dma_desc_set_dststride(void* dma_desc_ptr, unsigned int v) {
  dma_desc_2d_t* dma_desc_2d_ptr = reinterpret_cast<dma_desc_2d_t*>(dma_desc_ptr);
  (dma_desc_2d_ptr->dststride_srcstride) &= ~DESC_DSTSTRIDE_MASK;
  (dma_desc_2d_ptr->dststride_srcstride) |= ((v << DESC_DSTSTRIDE_SHIFT) & DESC_DSTSTRIDE_MASK);
}

// desc[7][15:0]
inline void dma_desc_set_srcwidthoffset(void* dma_desc_ptr, unsigned int v) {
  dma_desc_2d_t* dma_desc_2d_ptr = reinterpret_cast<dma_desc_2d_t*>(dma_desc_ptr);
  (dma_desc_2d_ptr->dstwidthoffset_srcwidthoffset) &= ~DESC_SRCWIDTHOFFSET_MASK;
  (dma_desc_2d_ptr->dstwidthoffset_srcwidthoffset) |=
      ((v << DESC_SRCWIDTHOFFSET_SHIFT) & DESC_SRCWIDTHOFFSET_MASK);
}

// desc[7][31:16]
inline void dma_desc_set_dstwidthoffset(void* dma_desc_ptr, unsigned int v) {
  dma_desc_2d_t* dma_desc_2d_ptr = reinterpret_cast<dma_desc_2d_t*>(dma_desc_ptr);
  (dma_desc_2d_ptr->dstwidthoffset_srcwidthoffset) &= ~DESC_DSTWIDTHOFFSET_MASK;
  (dma_desc_2d_ptr->dstwidthoffset_srcwidthoffset) |=
      ((v << DESC_DSTWIDTHOFFSET_SHIFT) & DESC_DSTWIDTHOFFSET_MASK);
}

}  // namespace hexagon
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_HEXAGON_HEXAGON_USER_DMA_DESCRIPTORS_H_
