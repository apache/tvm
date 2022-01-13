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
"""Common x86 related utilities"""
import tvm


def target_has_sse42(target):
    return (
        target_has_avx(target)
        or target_has_avx2(target)
        or target_has_avx512(target)
        or target_has_vnni(target)
        or target
        in {
            "silvermont",
            "slm",
            "goldmont",
            "goldmont-plus",
            "tremont",
            "nehalem",
            "corei7",
            "westmere",
            "bdver1",
            "bdver2",
            "bdver3",
            "x86-64-v2",
        }
    )


def target_has_avx(target):
    return (
        target_has_avx2(target)
        or target_has_avx512(target)
        or target_has_vnni(target)
        or target in {"sandybridge", "corei7-avx", "ivybridge", "core-avx-i"}
    )


def target_has_avx2(target):
    return (
        target_has_avx512(target)
        or target_has_vnni(target)
        or target
        in {
            "haswell",
            "core-avx2",
            "broadwell",
            "skylake",
            "bdver4",
            "znver1",
            "znver2",
            "znver3",
            "x86-64-v3",
        }
    )


def target_has_avx512(target):
    return target in {
        "skylake-avx512",
        "skx",
        "knl",
        "knm",
        "x86-64-v4",
        "cannonlake",
        # explicit enumeration of VNNI capable due to collision with alderlake
        "cascadelake",
        "icelake-client",
        "rocketlake",
        "icelake",
        "tigerlake",
        "cooperlake",
        "sapphirerapids",
    }


def target_has_vnni(target):
    return target in {
        "cascadelake",
        "icelake-client",
        "rocketlake",
        "icelake",
        "tigerlake",
        "cooperlake",
        "sapphirerapids",
        "alderlake",
    }


def get_simd_32bit_lanes():
    mcpu = tvm.target.Target.current().mcpu
    fp32_vec_len = 4
    if target_has_avx512(mcpu):
        fp32_vec_len = 16
    elif target_has_avx2(mcpu):
        fp32_vec_len = 8
    return fp32_vec_len
