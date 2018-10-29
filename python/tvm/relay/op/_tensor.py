#pylint: disable=invalid-name, unused-argument
"""Backend compiler related feature registration"""
import tvm
import topi
from . import register

def add_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.add(inputs[0], inputs[1])]

def add_schedule(outputs, target):
    assert len(outputs) == 1
    return tvm.create_schedule(outputs[0].op)

register("add", "FTVMCompute", add_compute)
register("add", "FTVMSchedule", add_schedule)

def subtract_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.subtract(inputs[0], inputs[1])]

def subtract_schedule(outputs, target):
    assert len(outputs) == 1
    return tvm.create_schedule(outputs[0].op)

register("subtract", "FTVMCompute", subtract_compute)
register("subtract", "FTVMSchedule", subtract_schedule)

def multiply_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.multiply(inputs[0], inputs[1])]

def multiply_schedule(outputs, target):
    assert len(outputs) == 1
    return tvm.create_schedule(outputs[0].op)

register("multiply", "FTVMCompute", multiply_compute)
register("multiply", "FTVMSchedule", multiply_schedule)

def equal_compute(attrs, inputs, output_type, target):
    assert len(inputs) == 2
    return [topi.equal(inputs[0], inputs[1])]

def equal_schedule(outputs, target):
    assert len(outputs) == 1
    return tvm.create_schedule(outputs[0].op)

register("equal", "FTVMCompute", equal_compute)
register("equal", "FTVMSchedule", equal_schedule)