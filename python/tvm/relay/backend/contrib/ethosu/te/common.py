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
"""Common methods for the NPU tensor expressions"""

from typing import Tuple, List


def get_layout_transform_matrices(ofm_channels: int) -> Tuple[List[List[float]], List[List[float]]]:
    """Get the NHWC->NHCWB16 and NHCWB16->NHWC layout transform matrices.
    For information about the supported layouts see https://developer.arm.com/documentation/102420/
    0200/Functional-description/Control-and-data-flow/Supported-memory-formats-for-feature-maps

    Parameters
    ----------
    ofm_channels : int
        The number of output channels in a NHWC layout

    Returns
    -------
    nhwc_to_nhcwb16, nhcwb16_to_nhwc : Tuple[List[List[float]], List[List[float]]]
        The layout transformation matrices
    """

    # The value of the last dimension (B16) is always 16.
    nhwc_to_nhcwb16 = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 1 / 16, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 16],
        [0, 0, 0, 0, 1],
    ]

    # When we convert from NHWC to NHCWB16, the new C value is given by
    # (ofm_channels - 1) // 16 + 1, which is a lossy operation, so we need to use
    # the actual value of channels in the transform matrix to accurately recover
    # the C in NHWC when we convert from NHCWB16 to NHWC.
    nhcwb16_to_nhwc = [
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        # We need to offset only if number of ofm_channels is not divisible by 16
        # Moreover, we can't use just the "ofm_channels" as last element because
        # the propogation matrices are used to propogate block configs as well.
        [0, 0, 16, 0, 0, -(int(ofm_channels % 16 != 0)) * (16 - ofm_channels % 16)],
        [0, 0, 0, 0, 0, 1],
    ]

    return nhwc_to_nhcwb16, nhcwb16_to_nhwc
