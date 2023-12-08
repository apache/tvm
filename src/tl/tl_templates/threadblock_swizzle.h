#pragma once

#include "common.h"

namespace tl {

template <int panel_width>
__device__ dim3 rasterization2DRow() {
  const int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
  const int grid_size = gridDim.x * gridDim.y;
  const int panel_size = panel_width * gridDim.x;
  const int panel_offset = block_idx % panel_size;
  const int panel_idx = block_idx / panel_size;
  const int total_panel = cutlass::ceil_div(grid_size, panel_size);
  const int stride =
      panel_idx + 1 < total_panel ? panel_width : (grid_size - panel_idx * panel_size) / gridDim.x;
  const int col_idx =
      (panel_idx & 1) ? gridDim.x - 1 - panel_offset / stride : panel_offset / stride;
  const int row_idx = panel_offset % stride + panel_idx * panel_width;
  return {col_idx, row_idx, blockIdx.z};
}

template <int panel_width>
__device__ dim3 rasterization2DColumn() {
  const int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
  const int grid_size = gridDim.x * gridDim.y;
  const int panel_size = panel_width * gridDim.y;
  const int panel_offset = block_idx % panel_size;
  const int panel_idx = block_idx / panel_size;
  const int total_panel = cutlass::ceil_div(grid_size, panel_size);
  const int stride =
      panel_idx + 1 < total_panel ? panel_width : (grid_size - panel_idx * panel_size) / gridDim.y;
  const int row_idx =
      (panel_idx & 1) ? gridDim.y - 1 - panel_offset / stride : panel_offset / stride;
  const int col_idx = panel_offset % stride + panel_idx * panel_width;
  return {col_idx, row_idx, blockIdx.z};
}

}  // namespace tl
