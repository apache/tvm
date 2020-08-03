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
"""Generic image operators"""
from .default import default_schedule as _default_schedule


def schedule_dilation2d_nchw(outs):
    """Schedule for dilation2d
    Parameters
    ----------
    outs : Array of Tensor
        The computation graph description of dilation2d
        in the format of an array of tensors.
    Returns
    -------
    sch : Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)


def schedule_dilation2d_nhwc(outs):
    """Schedule for dilation2d
    Parameters
    ----------
    outs : Array of Tensor
        The computation graph description of dilation2d
        in the format of an array of tensors.
    Returns
    -------
    sch : Schedule
        The computation schedule for the op.
    """
    return _default_schedule(outs, False)
