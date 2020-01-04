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
class HardwareDescription(object):
    def __init__(self):
        self._op_constraints = defaultdict(list)

    def __getitem__(self, op_name):
        return self._op_constraints[op_name]

    @property
    def ops(self):
        return self._op_constraints.keys()


def create_accelerator_description():
    # TODO: change to DataType
    desc = HardwareDescription()
    desc['add'].append(([8, 8], [16]))
    desc['add'].append(([8, 8], [32]))
    desc['add'].append(([16, 16], [32]))
    desc['add'].append(([32, 32], [32]))
    desc['nn.conv2d'].append(([8, 8], [16]))
    # desc['nn.conv2d'].append(([8, 8], [32]))
    return desc

