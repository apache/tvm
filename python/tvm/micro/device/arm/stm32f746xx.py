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
"""Config definitions for ARM STM32F746XX devices"""
from .. import create_micro_lib_base, register_device

DEVICE_ID = 'arm.stm32f746xx'
TOOLCHAIN_PREFIX = 'arm-none-eabi-'

def create_micro_lib(obj_path, src_path, lib_type, options=None):
    if options is None:
        options = []
    options += [
        '-mcpu=cortex-m7',
        '-mlittle-endian',
        '-mfloat-abi=hard',
        '-mfpu=fpv5-sp-d16',
        '-mthumb',
        '-gdwarf-5',
        ]
    create_micro_lib_base(obj_path, src_path, TOOLCHAIN_PREFIX, DEVICE_ID, lib_type, options=options)


def default_config(server_addr, server_port):
    return {
        'device_id': DEVICE_ID,
        'toolchain_prefix': TOOLCHAIN_PREFIX,
        #
        # [Device Memory Layout]
        #   RAM   (rwx) : START = 0x20000000, LENGTH = 320K
        #   FLASH (rx)  : START = 0x8000000,  LENGTH = 1024K
        #
        'mem_layout': {
            'text': {
                'start': 0x20000180,
                'size': 20480,
            },
            'rodata': {
                'start': 0x20005180,
                'size': 20480,
            },
            'data': {
                'start': 0x2000a180,
                'size': 768,
            },
            'bss': {
                'start': 0x2000a480,
                'size': 768,
            },
            'args': {
                'start': 0x2000a780,
                'size': 1280,
            },
            'heap': {
                'start': 0x2000ac80,
                'size': 262144,
            },
            'workspace': {
                'start': 0x2004ac80,
                'size': 20480,
            },
            'stack': {
                'start': 0x2004fc80,
                'size': 80,
            },
        },
        'word_size': 4,
        'thumb_mode': True,
        'comms_method': 'openocd',
        'server_addr': server_addr,
        'server_port': server_port,
    }


register_device(DEVICE_ID, {
    'create_micro_lib': create_micro_lib,
    'default_config': default_config,
})
