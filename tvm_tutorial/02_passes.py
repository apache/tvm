import tvm
from tvm import relay
from typing import Union, Sequence

from module import TVMModule


@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def __init__(self, names: Union[str, Sequence[str]] = []):
        self.names = names

    def run_after_pass(self, mod, info):
        if self.names == "all" or str(info.name) in self.names:
            print("Mod after pass:", info.name)
            print(mod)


def main():
    # Load module
    m = TVMModule()
    relay_mod, relay_params = m.load()

    # Compile module
    target = tvm.target.Target("llvm")
    with tvm.transform.PassContext(
        opt_level=0, instruments=[PrintIR("all")],
    ):
        lib = relay.build(relay_mod, target=target, params=relay_params)

    # Benchmark module
    m.benchmark(lib)


if __name__ == "__main__":
    main()
