import tvm

from tvm import relay
from tvm.relay.ir_builder import scalar_type, convert, tensor_type


def make_rel(name, args, num_inputs=None, attrs=None):
    func = tvm.get_env_func("tvm.relay.type_relation." + name)
    if num_inputs is None:
        num_inputs = len(args) - 1
    return relay.ty.TypeRelation(func, args, num_inputs, attrs)

def make_solver():
    solver = relay._ir_pass._test_type_solver()
    solver.Solve = solver("Solve")
    solver.Unify = solver("Unify")
    solver.Resolve = solver("Resolve")
    solver.AddConstraint = solver("AddConstraint")

    def gen_type(name, args, out=None):
        out = out if out else relay.ty.IncompleteType()
        solver.AddConstraint(make_rel(name, args + [out]))
        return out

    solver.gen_type = gen_type
    return solver


def test_bcast():
    solver = make_solver()
    t0 = relay.ty.TensorType((10, 20), "float32")
    t1 = relay.ty.TensorType((10, 1), "float32")
    tc = relay.ty.TensorType((10, 1, 1), "float32")
    t2 = solver.gen_type("Broadcast", [t0, t1])
    t3 = solver.gen_type("Identity", [t2])
    t4 = solver.gen_type("Broadcast", [t3, tc])
    assert solver.Solve()
    assert solver.Resolve(t2) == relay.ty.TensorType((10, 20), "float32")
    assert solver.Resolve(t4) == relay.ty.TensorType((10, 10, 20), "float32")


def test_backward_solving():
    solver = make_solver()
    t0 = relay.ty.TensorType((10, 20), "float32")
    tc = relay.ty.TensorType((10, 1, 1), "float32")
    t1 = relay.ty.IncompleteType()
    t3 = solver.gen_type("Broadcast", [t0, t1])
    t2 = solver.gen_type("Identity", [t1], out=tc)
    assert solver.Solve()
    assert solver.Resolve(t3) == relay.ty.TensorType((10, 10, 20), "float32")



if __name__ == "__main__":
    test_bcast()
    test_backward_solving()
