from __future__ import absolute_import
from tvm._ffi.runtime_ctypes import TVMType

import numpy as np
from collections import defaultdict

######################################################

# for different hardwares, we need to consider instructions that it support. Reflect on graph level:
# - dtype constraint
# - shape constraint
# - layout constraint
# - op/subgraph combination
# - detect graph pattern, consider regex
# - check auto graph
# Consider:
# - Similarities with:
#   - TypeInfer of Op
#   - TensorIntrinsic
# - VTA, GPU:TensorCore, Quantization, LayoutTransform

class OpConstraints(object):
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
                    dtype = TVMType(dtype)
                assert isinstance(dtype, TVMType)
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
        return 'OpConstraints[idtypes={}, odtypes={}]'.format(self.idtypes, self.odtypes)


class HardwareDescription(object):
    def __init__(self):
        self._op_constraints = defaultdict(list)

    def __getitem__(self, op_name):
        return self._op_constraints[op_name]

    @property
    def ops(self):
        return self._op_constraints.keys()


def create_accelerator_description():
    desc = HardwareDescription()
    desc['add'].append(OpConstraints(idtypes=['int8', 'int8'], odtypes=['int16']))
    desc['add'].append(OpConstraints(idtypes=['int8', 'int8'], odtypes=['int32']))
    desc['add'].append(OpConstraints(idtypes=['int16', 'int16'], odtypes=['int32']))
    # TODO(ziheng) enable int32 addition will lead to overflow easily
    #  - add output_bit constraint to restrict the using for output bit-width
    # desc['add'].append(OpConstraints(idtypes=['int32', 'int32'], odtypes=['int32']))
    # TODO(ziheng) enable int16 conv2d will lead to overflow easily
    #  - add input_bit constraint to restrict the using for output bit-width
    # desc['nn.conv2d'].append(OpConstraints(idtypes=['int8', 'int8'], odtypes=['int16']))
    desc['nn.conv2d'].append(OpConstraints(idtypes=['int8', 'int8'], odtypes=['int32']))
    # desc['nn.conv2d'].append(OpConstraints(idtypes=['int16', 'int16'], odtypes=['int32']))
    desc['nn.global_avg_pool2d'].append(OpConstraints(idtypes=['float32']), odtypes=['float32'])  # force dequantize
    return desc
