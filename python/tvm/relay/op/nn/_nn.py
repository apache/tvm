#pylint: disable=invalid-name
"""Backend compiler related feature registration"""
import tvm
import topi
from .. import register

def dense_compiler(attrs, inputs, output_type):
    assert len(inputs) == 2
    return [topi.nn.dense(inputs[0], inputs[1])]

def dense_schedule(outputs, target):
    assert len(outputs) == 1
    return tvm.create_schedule(outputs[0].op)

register("nn.dense", "FTVMCompute", dense_compiler)
register("nn.dense", "FTVMSchedule", dense_schedule)
