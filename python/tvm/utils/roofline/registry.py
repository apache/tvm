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
"""Definition of generic functions for estimating peak flops and bandwidth"""
from typing import Optional
from ...target import Target, generic_func
from ...runtime import Device
from ...rpc.client import RPCSession


@generic_func
def estimate_peak_bandwidth(
    target: Target,
    dev: Device,
    remote: Optional[RPCSession] = None,
) -> float:
    """Estimate peak memory bandwidth of a target/device combo.

    Peak bandwidth is estimated by running a small experiment on the underlying
    hardware. The peak bandwidth measurement assumes that vector instructions
    are being used to load the data.

    Parameters
    ----------
    target : Target
        Target to use for measurement. This target should be as specific to the
        underlying hardware as possible.
    dev : Device
        Device to measure peak bandwidth on.
    remote : Optional[RPCSession]
      Remote session used to upload artifacts for runtime evaluation. Must be
      the same session used to create `dev`.

    Returns
    -------
    float
        Peak memory bandwidth in bytes/seconds.
    """
    raise NotImplementedError()


@generic_func
def estimate_peak_flops(
    target: Target,
    dev: Device,
    remote: Optional[RPCSession],
) -> float:
    """
    Estimate the maximum number of FLOP/s this target/device combo is capable
    of reaching by running a test program. This is a generic function that
    should be overridden for each target.

    Parameters
    ----------
    target : Target
        Target to run on. This should be as specific to the actual hardware as
        possible to make sure that LLVM generates the best vector code.
    dev : Device
        Device to run on.
    remote : Optional[RPCSession]
      Remote session used to upload artifacts for runtime evaluation. Must be
      the same session used to create `dev`.

    Returns
    -------
    float
        Approximate sustained FLOP/s of this target/device combo. Each FMA
        operation counts as two FLOPs.
    """
    raise NotImplementedError()
