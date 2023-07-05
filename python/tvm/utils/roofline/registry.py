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
from typing import Dict, Optional, Tuple

import numpy as np

from ...rpc.client import RPCSession
from ...runtime import Device
from ...target import Target, generic_func
from ...tir import PrimFunc


@generic_func
def estimate_peak_bandwidth(
    func: PrimFunc,
    features: Dict[str, np.ndarray],
    target: Target,
    dev: Device,
    remote: Optional[RPCSession] = None,
) -> Tuple[float, float, str]:
    """Estimate peak memory bandwidth of a target/device combo.

    Peak bandwidth is estimated by running a small experiment on the underlying
    hardware. The peak bandwidth measurement assumes that vector instructions
    are being used to load the data.

    Parameters
    ----------
    func : PrimFunc
        Function to estimate peak bandwidth for. Used to check if a specific
        kind of memory could be used with this function.
    features : Dict[str, np.ndarry]
        Features extracted from `func`. Used to check if a specific kind of
        memory could be used with this function.
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
    loaded_bytes : float
        Estimated bytes loaded by `func`.
    peak_bandwidth : float
        Peak memory bandwidth in bytes/seconds.
    name : str
        Name of the memory being used.
    """
    raise NotImplementedError()


@generic_func
def estimate_peak_flops(
    func: PrimFunc,
    features: Dict[str, np.ndarray],
    target: Target,
    dev: Device,
    remote: Optional[RPCSession],
) -> Tuple[float, float, str]:
    """
    Estimate the maximum number of FLOP/s this target/device combo is capable
    of reaching by running a test program. This is a generic function that
    should be overridden for each target.

    Parameters
    ----------
    func : PrimFunc
        Function to estimate peak flops for. Used to check if a specific kind
        intrinsic or dtype could be used with this function.
    features : Dict[str, np.ndarry]
        Features extracted from `func`. Used to check if a specific kind
        intrinsic or dtype could be used with this function.
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
    flops : float
        Estimated number of flops used by `func`.
    peak_flops : float
        Approximate sustained FLOP/s of this target/device combo assuming
        vectorized FMA instructions. Each FMA operation counts as two FLOPs.
    name : str
        Dtype/intrinsic used by `func` to achieve peak flops.
    """
    raise NotImplementedError()
