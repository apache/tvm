# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
Useful enumerations and others
=====================
**Author**: `Federico Peccia <https://fPecc.github.io/>`_
"""

from enum import Enum

COUNTERS = {
    1: "MAIN_LD_CYCLES",
    2: "MAIN_ST_CYCLES",
    3: "MAIN_EX_CYCLES",
    4: "MAIN_LD_ST_CYCLES",
    5: "MAIN_LD_EX_CYCLES",
    6: "MAIN_ST_EX_CYCLES",
    7: "MAIN_LD_ST_EX_CYCLES",
    8: "LOAD_DMA_WAIT_CYCLE",
    9: "LOAD_ACTIVE_CYCLE",
    10: "LOAD_SCRATCHPAD_WAIT_CYCLE",
    11: "STORE_DMA_WAIT_CYCLE",
    12: "STORE_ACTIVE_CYCLE",
    13: "STORE_POOLING_CYCLE",
    14: "STORE_SCRATCHPAD_WAIT_CYCLE",
    15: "DMA_TLB_MISS_CYCLE",
    16: "DMA_TLB_HIT_REQ",
    17: "DMA_TLB_TOTAL_REQ",
    18: "RDMA_ACTIVE_CYCLE",
    19: "RDMA_TLB_WAIT_CYCLES",
    20: "RDMA_TL_WAIT_CYCLES",
    21: "WDMA_ACTIVE_CYCLE",
    22: "WDMA_TLB_WAIT_CYCLES",
    23: "WDMA_TL_WAIT_CYCLES",
    24: "EXE_ACTIVE_CYCLE",
    25: "EXE_FLUSH_CYCLE",
    26: "EXE_CONTROL_Q_BLOCK_CYCLE",
    27: "EXE_PRELOAD_HAZ_CYCLE",
    28: "EXE_OVERLAP_HAZ_CYCLE",
    29: "SCRATCHPAD_A_WAIT_CYCLE",
    30: "SCRATCHPAD_B_WAIT_CYCLE",
    31: "SCRATCHPAD_D_WAIT_CYCLE",
    32: "ACC_A_WAIT_CYCLE",
    33: "ACC_B_WAIT_CYCLE",
    34: "ACC_D_WAIT_CYCLE",
    35: "A_GARBAGE_CYCLES",
    36: "B_GARBAGE_CYCLES",
    37: "D_GARBAGE_CYCLES",
    38: "IM2COL_MEM_CYCLES",
    39: "IM2COL_ACTIVE_CYCLES",
    40: "IM2COL_TRANSPOSER_WAIT_CYCLE",
    41: "RESERVATION_STATION_FULL_CYCLES",
    42: "RESERVATION_STATION_ACTIVE_CYCLES",
    43: "LOOP_MATMUL_ACTIVE_CYCLES",
    44: "TRANSPOSE_PRELOAD_UNROLLER_ACTIVE_CYCLES",
    45: "RESERVATION_STATION_LD_COUNT",
    46: "RESERVATION_STATION_ST_COUNT",
    47: "RESERVATION_STATION_EX_COUNT",
    48: "RDMA_BYTES_REC",
    49: "WDMA_BYTES_SENT",
    50: "RDMA_TOTAL_LATENCY",
    51: "WDMA_TOTAL_LATENCY",
}


class ClipArgs(Enum):
    """
    This is a helper enums to obtain the correct index
    of clip arguments.
    """

    A_MIN = 1
    A_MAX = 2


class BinaryElementwiseArgs(Enum):
    """This is a helper enums to access the correct index
    of binary elementwise arguments
    """

    IFM1 = 0
    IFM2 = 1
    IFM1_SCALE = 2
    IFM1_ZERO_POINT = 3
    IFM2_SCALE = 4
    IFM2_ZERO_POINT = 5
    OFM_SCALE = 6
    OFM_ZERO_POINT = 7


class QDenseArgs(Enum):
    """
    This is a helper enum to access the correct index of
    qnn.dense arguments
    """

    IFM = 0
    WEIGHTS = 1
    IFM_ZERO_POINT = 2
    WEIGHTS_ZERO_POINT = 3
    IFM_SCALE = 4
    WEIGHTS_SCALE = 5


class QConv2DArgs(Enum):
    """
    This is a helper enum to obtain the correct index
    of qnn.conv2d arguments.
    """

    IFM = 0
    WEIGHTS = 1
    IFM_ZERO_POINT = 2
    WEIGHTS_ZERO_POINT = 3
    IFM_SCALE = 4
    WEIGHTS_SCALE = 5


class RequantArgs(Enum):
    """
    This is a helper enum to obtain the correct index
    of qnn.requantize arguments.
    """

    IFM_SCALE = 1
    IFM_ZERO_POINT = 2
    OFM_SCALE = 3
    OFM_ZERO_POINT = 4
