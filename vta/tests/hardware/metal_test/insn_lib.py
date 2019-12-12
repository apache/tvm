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

from ctypes import *
from _macros_h import *
import numpy as np

# RANDOM RANGE MAX
RAND_MAX = 32767

# test library
testlib = CDLL('./lib.so')

def wrap_function(lib, funcname, restype, argtypes):
    """Simplify wrapping ctypes functions"""
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func

"""Structures"""
class VTAGenericInsn(Structure):
    _fields_ = [("opcode", c_uint64, VTA_OPCODE_BIT_WIDTH),
            ("pop_prev_dep", c_uint64, 1),
            ("pop_next_dep", c_uint64, 1),
            ("push_prev_dep", c_uint64, 1),
            ("push_next_dep", c_uint64, 1),
            ("pad_0", c_uint64, 64 - VTA_OPCODE_BIT_WIDTH - 4),
            ("pad_1", c_uint64, 64)]

class VTAUop(Structure):
    _fields_ = [("dst_inx", c_uint32, VTA_LOG_ACC_BUFF_DEPTH),
                ("src_idx", c_uint32, VTA_LOG_INP_BUFF_DEPTH),
                ("wgt_idx", c_uint32, VTA_LOG_WGT_BUFF_DEPTH)]

"""Types"""
VTADeviceHandle = c_void_p
# may be 32 bit 
vta_phy_addr_t = c_uint64
uop_T = c_uint32
inp_T = c_int8
wgt_T = c_int8
out_T = c_int8
acc_T = c_int32

# double pointer helper
def get_2dpointer(arr):
    pt = (c_void_p * arr.shape[0])()
    for i in range(arr.shape[0]):
        pt[i] = arr[i].ctypes.data_as(c_void_p)
    return cast(pt, POINTER(c_void_p))

def get_2darray(pt, row, clmn):
    output_list = pt[:row]
    res = np.zeros((row, clmn))
    for i in range(row):
        res[i] = (output_list[i][:clmn])
    return res
        
"""Wrap C functions into python"""

"""Print Functions"""
print_instruction = wrap_function(testlib, 'printInstruction', None, [c_int, POINTER(VTAGenericInsn)])
print_param = wrap_function(testlib, 'printParameters', None, None)
print_uop = wrap_function(testlib, 'printMicroOp', None, [c_int, POINTER(VTAUop)])

"""Memory Function"""
alloc_buffer = wrap_function(testlib, 'allocBuffer', c_void_p, [c_size_t])

transfer = wrap_function(testlib, 'transfer', None, [c_void_p, c_void_p, c_size_t])

pack_buffer = wrap_function(testlib, 'packBufferWrap', None, 
        [c_int, c_void_p, POINTER(c_void_p), c_int, c_int, c_int, c_int])

unpack_buffer = wrap_function(testlib, 'unpackBufferWrap', None, 
        [POINTER(POINTER(out_T)), POINTER(c_uint32), c_int, c_int, c_int, c_int])

alloc_2darray = wrap_function(testlib, 'alloc2dArrayWrap', 
        POINTER(POINTER(out_T)), [c_int, c_int])

concat_uop = wrap_function(testlib, 'concatUop', None, 
        [POINTER(VTAUop), POINTER(VTAUop), c_size_t, c_size_t])

"""Get Instructions, Uops and Opcodes"""
get_alu_insn = wrap_function(testlib, 'getALUInsn', VTAGenericInsn, 
        [c_int, c_int, c_bool, c_int, c_bool, c_int, c_int, c_int, c_int])

get_gemm_insn = wrap_function(testlib, 'getGEMMInsn', VTAGenericInsn, 
        [c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int])

get_finish_insn = wrap_function(testlib, 'getFinishInsn', VTAGenericInsn, [c_bool, c_bool])

get_opcode_string = wrap_function(testlib, 'getOpcodeString', c_char_p, [c_int, c_bool])

get_gemm_uops = wrap_function(testlib, 'getGEMMUops', POINTER(VTAUop), 
        [c_int, c_int, c_int, c_bool, c_bool])

get_copy_uops = wrap_function(testlib, 'getCopyUops', POINTER(VTAUop), [c_int, c_int, c_int])

get_map_alu_uops = wrap_function(testlib, 'getMapALUUops', POINTER(VTAUop), [c_int, c_bool])

get_1Dloadstore_insn = wrap_function(testlib, 'get1DLoadStoreInsn', VTAGenericInsn, 
        [c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int])

get_2Dloadstore_insn = wrap_function(testlib, 'get2DLoadStoreInsn', VTAGenericInsn, 
        [c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int])

"""VTA Device Functions"""
vta_mem_alloc = wrap_function(testlib, 'VTAMemAlloc', c_void_p, [c_size_t, c_int])

vta_mem_free = wrap_function(testlib, 'VTAMemFree', None, [c_void_p])

vta_mem_get_phy_addr = wrap_function(testlib, 'VTAMemGetPhyAddr', vta_phy_addr_t, [c_void_p])

vta_write_mapped_reg = wrap_function(testlib, 'VTAWriteMappedReg', None, [c_void_p, c_uint32, c_uint32])

vta_read_mapped_reg = wrap_function(testlib, 'VTAReadMappedReg', c_uint32, [c_void_p, c_uint32])

vta_map_reg = wrap_function(testlib, 'VTAMapRegister', c_void_p, [c_uint32])

vta_unmap_reg = wrap_function(testlib, 'VTAUnmapRegister', None, [c_void_p])

vta_device_free = wrap_function(testlib, 'VTADeviceFree', None, [VTADeviceHandle])

vta_device_run = wrap_function(testlib, 'VTADeviceRun', c_int, [VTADeviceHandle, vta_phy_addr_t, c_uint32, c_uint32])

vta_device_alloc = wrap_function(testlib, 'VTADeviceAlloc', VTADeviceHandle, None)

vta_run = wrap_function(testlib, 'vta', c_uint64, 
        [c_uint32, POINTER(VTAGenericInsn), POINTER(VTAUop), POINTER(c_uint32), POINTER(c_uint32), POINTER(c_uint32), POINTER(c_uint32)])
