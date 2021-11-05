import tvm
from tvm import relay

from module import TVMModule

def main():
    # Load module
    m = TVMModule(torch_model=True)
    relay_mod, relay_params = m.load()

    # Compile module
    print("Compile module...")
    # target = tvm.target.Target("llvm -mcpu=core-avx2")
    target = tvm.target.Target("llvm")
    with tvm.transform.PassContext(opt_level=0):
        lib = relay.build(relay_mod, target=target, params=relay_params)

    # Benchmark module
    m.benchmark(lib)


if __name__ == "__main__":
    main()
