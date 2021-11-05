import tvm
from tvm import relay
from tvm.ir.instrument import PassTimingInstrument
from tvm.relay.analysis import list_op_freqs
from typing import Union, Sequence

from module import TVMModule


@tvm.transform.module_pass(opt_level=0, name="TutorialRelayPass")
class TutorialRelayPass:
    """Count the number of node appearances."""

    def transform_module(self, mod, ctx):
        freq = list_op_freqs(mod)
        print("### TUTORIAL RELAY PASS OUTPUT ###")
        print(freq)
        return mod


@tvm.tir.transform.prim_func_pass(opt_level=0, name="TutorialTIRPass")
class TutorialTIRPass:
    """Count the number of for nodes."""

    def __init__(self):
        self.for_count = 0

    def transform_function(self, func, mod, ctx):
        def fvisit(stmt):
            if isinstance(stmt, tvm.tir.For):
                self.for_count += 1

        tvm.tir.stmt_functor.post_order_visit(func.body, fvisit)
        print("### TUTORIAL TIR PASS OUTPUT ###")
        print(self.for_count)

        return func


@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def __init__(self, names: Union[str, Sequence[str]] = []):
        self.names = names

    def run_after_pass(self, mod, info):
        if self.names == "all" or str(info.name) in self.names:
            print("Mod after pass:", info.name)
            # print(mod)


def main():
    # Load module
    m = TVMModule(torch_model=True)
    relay_mod, relay_params = m.load()

    # Manual compiler passes
    target = tvm.target.Target("llvm")

    use_manual_passes = True
    if use_manual_passes:
        print("### BEFORE PASSES ###")
        print(relay_mod["main"])

        timing_inst = PassTimingInstrument()
        with tvm.transform.PassContext(
            opt_level=0,
            # required_pass=["SimplifyExpr"],
            # disabled_pass=["FoldScaleAxis"],
            instruments=[timing_inst],
        ):
            # relay_mod = relay.transform.InferType()(relay_mod)
            # relay_mod = relay.transform.SimplifyInference()(relay_mod)
            # relay_mod = TutorialPass()(relay_mod)
            # relay_mod, relay_params = relay.optimize(relay_mod, target, relay_params)

            print("### PASSES ###")
            print(timing_inst.render())

        print("### AFTER PASSES ###")
        print(relay_mod["main"])

    # Compile module
    print("Compile module...")
    with tvm.transform.PassContext(
        config={"tir.add_lower_pass": [(0, TutorialTIRPass())]},
        opt_level=0,
        instruments=[PrintIR("all")],
    ):
        lib = relay.build(relay_mod, target=target, params=relay_params)

    # Benchmark module
    m.benchmark(lib)


if __name__ == "__main__":
    main()
