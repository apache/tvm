// See LICENSE for license details.

package vta

import chisel3.Module
import freechips.rocketchip.config.{Parameters, Config}
import junctions._

class VTAConfig extends Config((site, here, up) => {
  // Core
  // case XLEN => 8
  // case Trace => true

  case LOG_INP_WIDTH => 3
  case LOG_WGT_WIDTH => 3
  case LOG_ACC_WIDTH => 5
  case LOG_OUT_WIDTH => 3
  case LOG_BATCH => 0
  case LOG_BLOCK_IN => 4
  case LOG_BLOCK_OUT => 4
  case LOG_UOP_BUFF_SIZE => 15
  case LOG_INP_BUFF_SIZE => 15
  case LOG_WGT_BUFF_SIZE => 18
  case LOG_ACC_BUFF_SIZE => 17
}
)
