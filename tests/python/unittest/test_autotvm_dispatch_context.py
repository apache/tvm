"""Test dispatcher.
The dispatcher can choose which template to use according
to the parameters of workload"""

from collections import namedtuple
from tvm.autotvm.task import dispatcher, DispatchContext

SimpleWorkload = namedtuple("SimpleWorkload", ["key"])
SimpleConfig = namedtuple("SimpleConfig", ["template_key"])

def test_dispatch():
    @dispatcher
    def my_dispatcher(a, b):
        return SimpleWorkload(key=a + b)

    @my_dispatcher.register("spatial_pack")
    def _sp_pack_add(cfg, a, b):
        return b + 100

    @my_dispatcher.register("im2col")
    def _im2col_add(cfg, a, b):
        return a + 1

    class SimpleDispatcher(DispatchContext):
        def query(self, target, workload):
            tkey = "spatial_pack" if workload.key > 2 else "im2col"
            return SimpleConfig(tkey)

    with SimpleDispatcher():
        # im2col
        assert my_dispatcher(1, 0) == 2
        # spack
        assert my_dispatcher(1, 100) == 200

if __name__ == "__main__":
    test_dispatch()
