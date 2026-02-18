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
"""ARM CPU target tags."""
from .registry import register_tag


# ---------- Raspberry Pi (self-hosted, with host config) ----------
register_tag(
    "raspberry-pi/4b-aarch64",
    {
        "kind": "llvm",
        "mtriple": "aarch64-linux-gnu",
        "mcpu": "cortex-a72",
        "mattr": ["+neon"],
        "num-cores": 4,
        "host": {
            "kind": "llvm",
            "mtriple": "aarch64-linux-gnu",
            "mcpu": "cortex-a72",
            "mattr": ["+neon"],
            "num-cores": 4,
        },
    },
)


# ---------- ARM boards ----------
def _register_arm_tag(name, config):
    base = {"kind": "llvm", "keys": ["arm_cpu", "cpu"], "device": "arm_cpu"}
    base.update(config)
    register_tag(name, base)


_register_arm_tag(
    "arm/pixel2",
    {"model": "snapdragon835", "mtriple": "arm64-linux-android", "mattr": ["+neon"]},
)
_register_arm_tag(
    "arm/mate10",
    {"model": "kirin970", "mtriple": "arm64-linux-android", "mattr": ["+neon"]},
)
_register_arm_tag(
    "arm/rasp3b",
    {"model": "bcm2837", "mtriple": "armv7l-linux-gnueabihf", "mattr": ["+neon"]},
)
_register_arm_tag(
    "arm/rasp4b",
    {
        "model": "bcm2711",
        "mtriple": "armv8l-linux-gnueabihf",
        "mattr": ["+neon"],
        "mcpu": "cortex-a72",
    },
)
_register_arm_tag(
    "arm/rasp4b64",
    {
        "model": "bcm2711",
        "mtriple": "aarch64-linux-gnu",
        "mattr": ["+neon"],
        "mcpu": "cortex-a72",
    },
)
_register_arm_tag(
    "arm/rk3399",
    {"model": "rk3399", "mtriple": "aarch64-linux-gnu", "mattr": ["+neon"]},
)
_register_arm_tag(
    "arm/pynq",
    {"model": "pynq", "mtriple": "armv7a-linux-eabi", "mattr": ["+neon"]},
)
_register_arm_tag(
    "arm/ultra96",
    {"model": "ultra96", "mtriple": "aarch64-linux-gnu", "mattr": ["+neon"]},
)
_register_arm_tag(
    "arm/beagleai",
    {
        "model": "beagleai",
        "mtriple": "armv7a-linux-gnueabihf",
        "mattr": ["+neon", "+vfp4", "+thumb2"],
        "mcpu": "cortex-a15",
    },
)
_register_arm_tag(
    "arm/stm32mp1",
    {
        "model": "stm32mp1",
        "mtriple": "armv7a-linux-gnueabihf",
        "mattr": ["+neon", "+vfp4", "+thumb2"],
        "mcpu": "cortex-a7",
    },
)
_register_arm_tag(
    "arm/thunderx",
    {
        "model": "thunderx",
        "mtriple": "aarch64-linux-gnu",
        "mattr": ["+neon", "+crc", "+lse"],
        "mcpu": "thunderxt88",
    },
)
