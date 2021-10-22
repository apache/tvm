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
STM32 helper functions
"""

# todo - push the STM32 flags/definition in an interface file (TBD) to share them
#        with the C-code.

#  format(32b)    31..24: STM32 series
#                 23..17: spare
#                 16:     FPU is enabled or not
#                 15..12: spare
#                 11:     H7/F7:D$ is enabled
#                 10:     H7/F7:I$ is enabled             F4/L4: ART DCen
#                 9:      F7: ART enabled                 F4/L4: ART ICen   L5: ICACHE
#                 8:      F4/L4/F7:ART Prefetch enabled
#                 7..0:   ART number of cycles (latency)
#

_STM32_SERIES_F4 = 1
_STM32_SERIES_F7 = 2
_STM32_SERIES_H7 = 3
_STM32_SERIES_L5 = 4
_STM32_SERIES_L4 = 1


def _is_series(fmt, series):
    return (fmt >> 24) == series


def stm32_id_to_str(dev_id):
    """Helper function to return a Human readable device ID description"""

    switcher = {
        0x422: "STM32F303xB/C",
        0x438: "STM32F303x6/8",
        0x446: "STM32F303xD/E",
        0x431: "STM32F411xC/E",
        0x423: "STM32F401xB/C",
        0x433: "STM32F401xD/E",
        0x435: "STM32L43xxx",
        0x462: "STM32L45xxx",
        0x415: "STM32L4x6xx",
        0x470: "STM32L4Rxxx",
        0x472: "STM32L5[5,6]2xx",
        0x449: "STM32F74xxx",
        0x450: "STM32H743/53/50xx and STM32H745/55/47/57xx",
        0x451: "STM32F7[6,7]xxx",
    }
    desc_ = "0x{:X} - ".format(dev_id)
    desc_ += switcher.get(dev_id, "UNKNOW")
    return desc_


def stm32_config_to_str(cache):
    """Return human desc fpu/cache config
    see FW c-file aiTestUtility.c
    """

    def _fpu(cache_):
        if cache_ & (1 << 16):
            return "FPU enabled"
        return "no FPU"

    def _lat(cache_):
        if (cache_ >> 24) > 0 and not _is_series(cache, _STM32_SERIES_L5):
            return "lat={}".format(cache_ & 0xFF)
        return ""

    def _art_f4(cache_):
        desc = "ART:"
        if cache_ & 1 << 8:
            desc += " PRFTen"
        if cache_ & 1 << 9:
            desc += " ICen"
        if cache_ & 1 << 10:
            desc += " DCen"
        return desc

    def _icache_l5(cache_):
        desc = "ICACHE: "
        desc += "enabled" if cache_ & 1 << 9 else "disabled"
        return desc

    def _core_f7h7(cache_):
        desc = "Core:"
        if cache_ & 1 << 10:
            desc += "I$"
        else:
            desc += "--"
        if cache_ & 1 << 11:
            desc += "/D$"
        else:
            desc += "/--"
        return desc

    def _art_f7(cache_):
        desc = "ART:"
        if cache_ & 1 << 8:
            desc += " PRFTen"
        if cache_ & 1 << 9:
            desc += " ARTen"
        return desc

    desc = _fpu(cache)
    attr = _lat(cache)
    desc += ", " + attr if attr else ""

    if _is_series(cache, _STM32_SERIES_F4) or _is_series(cache, _STM32_SERIES_L4):
        attr = _art_f4(cache)
        desc += ", " + attr if attr else ""

    if _is_series(cache, _STM32_SERIES_F7) or _is_series(cache, _STM32_SERIES_H7):
        attr = _core_f7h7(cache)
        desc += ", " + attr if attr else ""

    if _is_series(cache, _STM32_SERIES_F7):
        attr = _art_f7(cache)
        desc += ", " + attr if attr else ""

    if _is_series(cache, _STM32_SERIES_L5):
        attr = _icache_l5(cache)
        desc += ", " + attr if attr else ""

    return desc
