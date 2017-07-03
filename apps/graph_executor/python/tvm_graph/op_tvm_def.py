"""NNVM operator definitions."""
import tvm

@tvm.register_func("tvm_graph.compute.add")
def compute_add(a, b):
    return tvm.compute(a.shape, lambda *i: a(*i) + b(*i))

@tvm.register_func("tvm_graph.compute.exp")
def compute_exp(a):
    return tvm.compute(a.shape, lambda *i: tvm.exp(a(*i)))

@tvm.register_func("tvm_graph.schedule.ewise")
def schedule_ewise(outs, target):
    s = tvm.create_schedule([x.op for x in outs])
    tvm.schedule.AutoInlineElemWise(s)
    return s
