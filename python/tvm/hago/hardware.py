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
from __future__ import absolute_import
import tvm
from tvm import relay
from tvm._ffi.runtime_ctypes import DataType

import numpy as np
from collections import defaultdict

######################################################

# rename to OpSpec

class OpDesc(object):
    def __init__(self,
                 idtypes=None,
                 odtypes=None):

        def prepare_dtypes(dtypes):
            if dtypes is None:
                return dtypes
            assert isinstance(dtypes, list)
            ret = []
            for dtype in dtypes:
                if isinstance(dtype, (str, np.dtype)):
                    dtype = DataType(dtype)
                assert isinstance(dtype, DataType)
                ret.append(dtype)
            return ret

        self.idtypes = prepare_dtypes(idtypes)
        self.odtypes = prepare_dtypes(odtypes)

    def idtype(self, idx):
        return self.idtypes[idx]

    def odtype(self, idx):
        return self.odtypes[idx]

    def ishape(self, idx):
        pass

    def oshape(self, idx):
        pass

    def ilayout(self, idx):
        pass

    def olayout(self, idx):
        pass

    def __str__(self):
        return 'OpDesc[idtypes={}, odtypes={}]'.format(self.idtypes, self.odtypes)

    def __repr__(self):
        return 'OpDesc[idtypes={}, odtypes={}]'.format(self.idtypes, self.odtypes)


class Hardware(object):
    def __init__(self):
        self._op_descs = defaultdict(list)

    def __getitem__(self, op_name):
        if isinstance(op_name, tvm.ir.Op):
            op_name = op_name.name
        return self._op_descs[op_name]

    @property
    def ops(self):
        return self._op_descs.keys()



def is_integer_instruction(cstr):
    for dtype in (cstr.idtypes + cstr.odtypes):
        if 'float' in str(dtype):
            return False
    return True

def is_float_instruction(cstr):
    for dtype in (cstr.idtypes + cstr.odtypes):
        if 'int' in str(dtype):
            return False
    return True

def integer_constraints(constraints):
    cstrs = []
    for cstr in constraints:
        if is_integer_instruction(cstr):
            cstrs.append(cstr)
    return cstrs

def float_constraints(constraints):
    cstrs = []
    for cstr in constraints:
        if is_float_instruction(cstr):
            cstrs.append(cstr)
    return cstrs

def support_integer_computation(constraints):
    for cstr in constraints:
        if is_integer_instruction(cstr):
            return True
    return False

def support_float_computation(constraints):
    for cstr in constraints:
        if is_float_instruction(cstr):
            return True
    return False



def create_accelerator_description():
    desc = Hardware()
    # desc['add'].append(OpDesc(idtypes=['int8', 'int8'], odtypes=['int16']))
    # desc['add'].append(OpDesc(idtypes=['int8', 'int8'], odtypes=['int32']))
    # desc['add'].append(OpDesc(idtypes=['int16', 'int16'], odtypes=['int32']))
    desc['add'].append(OpDesc(idtypes=['int32', 'int32'], odtypes=['int32']))
    desc['add'].append(OpDesc(idtypes=['float32', 'float32'], odtypes=['float32']))
    # TODO(ziheng) enable int32 addition will lead to overflow easily
    #  - add output_bit constraint to restrict the using for output bit-width
    # TODO(ziheng) enable int16 conv2d will lead to overflow easily
    #  - add input_bit constraint to restrict the using for output bit-width
    # desc['nn.conv2d'].append(OpDesc(idtypes=['int8', 'int8'], odtypes=['int16']))
    desc['nn.conv2d'].append(OpDesc(idtypes=['int8', 'int8'], odtypes=['int32']))
    # desc['nn.conv2d'].append(OpDesc(idtypes=['int16', 'int16'], odtypes=['int32']))

    desc['nn.relu'].append(OpDesc(idtypes=['int32', 'int32'], odtypes=['int32']))
    desc['nn.max_pool2d'].append(OpDesc(idtypes=['int32', 'int32'], odtypes=['int32']))
    desc['nn.batch_flatten'].append(OpDesc(idtypes=['float32'], odtypes=['float32']))
    desc['nn.dense'].append(OpDesc(idtypes=['float32', 'float32'], odtypes=['float32']))
    desc['nn.global_avg_pool2d'].append(OpDesc(idtypes=['float32'], odtypes=['float32']))
    return desc
