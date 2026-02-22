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
"""RISC-V CPU target tags (from old riscv_cpu() translation table)."""

from .registry import register_tag


def _register_riscv_tag(name, config):
    base = {"kind": "llvm", "keys": ["arm_cpu", "cpu"], "device": "arm_cpu"}
    base.update(config)
    register_tag(name, base)


_register_riscv_tag(
    "riscv/sifive-e31",
    {
        "model": "sifive-e31",
        "mtriple": "riscv32-unknown-linux-gnu",
        "mcpu": "sifive-e31",
        "mabi": "ilp32",
    },
)
_register_riscv_tag(
    "riscv/sifive-e76",
    {
        "model": "sifive-e76",
        "mtriple": "riscv32-unknown-linux-gnu",
        "mcpu": "sifive-e76",
        "mabi": "ilp32",
    },
)
_register_riscv_tag(
    "riscv/sifive-u54",
    {
        "model": "sifive-u54",
        "mtriple": "riscv64-unknown-linux-gnu",
        "mcpu": "sifive-u54",
        "mabi": "lp64d",
    },
)
_register_riscv_tag(
    "riscv/sifive-u74",
    {
        "model": "sifive-u74",
        "mtriple": "riscv64-unknown-linux-gnu",
        "mcpu": "sifive-u74",
        "mabi": "lp64d",
    },
)
_register_riscv_tag(
    "riscv/licheepi3a",
    {
        "num-cores": 8,
        "mtriple": "riscv64-unknown-linux-gnu",
        "mcpu": "spacemit-x60",
        "mfloat-abi": "hard",
        "mabi": "lp64d",
    },
)
