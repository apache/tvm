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
"""ROI Align operator"""

from ..base import Expr
from . import _ffi_api


def roi_align(
    data: Expr,
    rois: Expr,
    pooled_size: int | tuple[int, int] | list[int],
    spatial_scale: float,
    sample_ratio: int = -1,
    aligned: bool = False,
    layout: str = "NCHW",
    mode: str = "avg",
):
    """ROI Align operator.

    Parameters
    ----------
    data : relax.Expr
        4-D input tensor.

    rois : relax.Expr
        2-D input tensor with shape `(num_roi, 5)` in
        `[batch_idx, x1, y1, x2, y2]` format.

    pooled_size : Union[int, Tuple[int, int], List[int]]
        Output pooled size.

    spatial_scale : float
        Ratio of input feature map height (or width) to raw image height (or width).

    sample_ratio : int, optional
        Sampling ratio for ROI align. Non-positive values use adaptive sampling.

    aligned : bool, optional
        Whether to use aligned ROIAlign semantics without the legacy 1-pixel clamp.

    layout : str, optional
        Layout of the input data. Supported values are `NCHW` and `NHWC`.

    mode : str, optional
        Mode for ROI align. Supported values are `avg` and `max`.

    Returns
    -------
    result : relax.Expr
        The computed result.
    """
    if isinstance(pooled_size, int):
        pooled_size = (pooled_size, pooled_size)
    return _ffi_api.roi_align(
        data,
        rois,
        pooled_size,
        spatial_scale,
        sample_ratio,
        aligned,
        layout,
        mode,
    )
