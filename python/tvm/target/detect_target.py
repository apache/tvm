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
"""Detect target."""

from collections.abc import Callable

from tvm_ffi import get_global_func

from ..runtime import Device, device
from . import Target


def _detect_cpu(dev: Device) -> Target:  # pylint: disable=unused-argument
    """Detect the host CPU architecture."""
    return Target(
        {
            "kind": "llvm",
            "mtriple": get_global_func(
                "tvm.codegen.llvm.GetDefaultTargetTriple",
                allow_missing=False,
            )(),
            "mcpu": get_global_func(
                "tvm.codegen.llvm.GetHostCPUName",
                allow_missing=False,
            )(),
        }
    )


SUPPORTED_DEVICE: dict[str, Callable[[Device], Target]] = {
    "cpu": _detect_cpu,
}

# Backward-compatible alias for the previous private module-level map.
SUPPORT_DEVICE = SUPPORTED_DEVICE


def register_device_target_detector(device_type: str, detector: Callable[[Device], Target]) -> None:
    """Register target detection for a runtime device type."""
    SUPPORTED_DEVICE[device_type] = detector


def detect_target_from_device(dev: str | Device) -> Target:
    """Detects Target associated with the given device. If the device does not exist,
    there will be an Error.

    Parameters
    ----------
    dev : Union[str, Device]
        The device to detect the target for.
        Supported device types are registered by backend hooks.

    Returns
    -------
    target : Target
        The detected target.
    """
    if isinstance(dev, str):
        dev = device(dev)
    device_type = Device._DEVICE_TYPE_TO_NAME[dev.dlpack_device_type()]
    if device_type not in SUPPORTED_DEVICE:
        raise ValueError(
            f"Auto detection for device `{device_type}` is not supported. "
            f"Currently only supports: {SUPPORTED_DEVICE.keys()}"
        )
    if not dev.exist:
        raise ValueError(
            f"Cannot detect device `{dev}`. Please make sure the device and its driver "
            "is installed properly, and TVM is compiled with the driver"
        )
    return SUPPORTED_DEVICE[device_type](dev)
