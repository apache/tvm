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
    def __init__(self, in_dtypes=None, out_dtypes=None):
        self.in_dtypes = self._prepare_dtypes(in_dtypes)
        self.out_dtypes = self._prepare_dtypes(out_dtypes)

    def in_dtype(self, idx):
        if isinstance(self.in_dtypes, list):
            return self.in_dtypes[idx]
        return self.in_dtypes

    def out_dtype(self, idx):
        if isinstance(self.out_dtypes, list):
            return self.out_dtypes[idx]
        return self.out_dtypes

    def ishape(self, idx):
        pass

    def oshape(self, idx):
        pass

    def ilayout(self, idx):
        pass

    def olayout(self, idx):
        pass

    def __str__(self):
        return 'OpDesc[in_dtypes={}, out_dtypes={}]'.format(self.in_dtypes, self.out_dtypes)

    def __repr__(self):
        return 'OpDesc[in_dtypes={}, out_dtypes={}]'.format(self.in_dtypes, self.out_dtypes)

    def _prepare_dtypes(self, dtypes):
        def convert_dtype(dtype):
            if isinstance(dtype, (str, np.dtype)):
                dtype = DataType(dtype)
            assert isinstance(dtype, DataType)
            return dtype

        if dtypes is None:
            return dtypes
        if isinstance(dtypes, list):
            ret = []
            for dtype in dtypes:
                dtype = convert_dtype(dtype)
                ret.append(dtype)
            return ret
        else:
            ret = convert_dtype(dtypes)
            return ret

    def _list_dtypes(self, dtypes):
        ret = []
        if isinstance(dtypes, list):
            ret += dtypes
        else:
            ret.append(dtypes)
        return ret

    def is_integer(self):
        dtypes = self._list_dtypes(self.in_dtypes) + self._list_dtypes(self.out_dtypes)
        for dtype in dtypes:
            if 'float' in str(dtype):
                return False
        return True
    
    def is_float(self):
        dtypes = self._list_dtypes(self.in_dtypes) + self._list_dtypes(self.out_dtypes)
        for dtype in dtypes:
            if 'int' in str(dtype):
                return False
        return True

class Hardware(object):
    def __init__(self):
        self._op_descs = defaultdict(list)

    def add_op_desc(self, op_name, desc):
        if isinstance(op_name, tvm.ir.Op):
            op_name = op_name.name
        self._op_descs[op_name].append(desc)

    def op_descs(self, op_name):
        if isinstance(op_name, tvm.ir.Op):
            op_name = op_name.name
        return self._op_descs[op_name]

    @property
    def ops(self):
        return self._op_descs.keys()

    def list_integer_descs(self, op_name):
        descs = self.op_descs(op_name)
        ret = []
        for desc in descs:
            if desc.is_integer():
                ret.append(desc)
        return ret
    
    def list_float_descs(self, op_name):
        descs = self.op_descs(op_name)
        ret = []
        for desc in descs:
            if desc.is_float():
                ret.append(desc)
        return ret


def create_accelerator_description():
    hardware = Hardware()
    hardware.add_op_desc('add', OpDesc(in_dtypes='float32', out_dtypes='float32'))
    # hardware.add_op_desc('add', OpDesc(in_dtypes='int8', out_dtypes='int16'))
    # hardware.add_op_desc('add', OpDesc(in_dtypes='int16', out_dtypes='int32'))
    hardware.add_op_desc('add', OpDesc(in_dtypes='int32', out_dtypes='int32'))
    # hardware.add_op_desc('nn.conv2d', OpDesc(in_dtypes='int8', out_dtypes='int16'))
    hardware.add_op_desc('nn.conv2d', OpDesc(in_dtypes='int8', out_dtypes='int32'))
    # hardware.add_op_desc('nn.conv2d', OpDesc(in_dtypes='int16', out_dtypes='int32'))

    hardware.add_op_desc('concatenate', OpDesc(in_dtypes='float32', out_dtypes='float32'))

    hardware.add_op_desc('nn.relu', OpDesc(in_dtypes='int32', out_dtypes='int32'))
    hardware.add_op_desc('nn.avg_pool2d', OpDesc(in_dtypes='float32', out_dtypes='float32'))
    # hardware.add_op_desc('nn.avg_pool2d', OpDesc(in_dtypes='int32', out_dtypes='int32'))
    hardware.add_op_desc('nn.max_pool2d', OpDesc(in_dtypes='int32', out_dtypes='int32'))
    hardware.add_op_desc('nn.batch_flatten', OpDesc(in_dtypes='float32', out_dtypes='float32'))
    hardware.add_op_desc('nn.dense', OpDesc(in_dtypes='float32', out_dtypes='float32'))
    hardware.add_op_desc('nn.global_avg_pool2d', OpDesc(in_dtypes='float32', out_dtypes='float32'))


    hardware.add_op_desc('nn.pad', OpDesc(in_dtypes='float32', out_dtypes='float32'))
    hardware.add_op_desc('nn.pad', OpDesc(in_dtypes='int8', out_dtypes='int8'))
    hardware.add_op_desc('layout_transform', OpDesc(in_dtypes='float32', out_dtypes='float32'))
    hardware.add_op_desc('layout_transform', OpDesc(in_dtypes='int8', out_dtypes='int8'))
    hardware.add_op_desc('multiply', OpDesc(in_dtypes='float32', out_dtypes='float32'))
    hardware.add_op_desc('subtract', OpDesc(in_dtypes='float32', out_dtypes='float32'))
    hardware.add_op_desc('nn.adaptive_avg_pool2d', OpDesc(in_dtypes='float32', out_dtypes='float32'))
    hardware.add_op_desc('mean', OpDesc(in_dtypes='float32', out_dtypes='float32'))
    hardware.add_op_desc('nn.softmax', OpDesc(in_dtypes='float32', out_dtypes='float32'))
    return hardware
