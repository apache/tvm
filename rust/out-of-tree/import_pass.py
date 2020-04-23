import tvm
import tvm.relay

test_func = tvm.relay.Function([], tvm.relay.var("x"))

pass_dylib = "/Users/jroesch/Git/tvm/rust/target/debug/libmy_pass.dylib"

def load_rust_extension(ext_dylib):
    load_so = tvm.get_global_func("runtime.module.loadfile_so")
    mod = load_so(ext_dylib)
    mod.get_function("initialize")()


def load_pass(pass_name, dylib):
    pass

load_rust_extension(pass_dylib)

# print(hex(test_func.handle.value))
res = tvm.get_global_func("__rust_pass")(test_func)
# rust_pass = load_pass("out_of_tree.Pass", pass_dylib)

# print(rust_pass)


test_func = tvm.relay.Function([], tvm.relay.var("x"))

pass_dylib = "/Users/jroesch/Git/tvm/rust/target/debug/libmy_pass.dylib"

def load_rust_extension(ext_dylib):
    load_so = tvm.get_global_func("runtime.module.loadfile_so")
    mod = load_so(ext_dylib)
    mod.get_function("initialize")()


def load_pass(pass_name, dylib):
    pass

load_rust_extension(pass_dylib)

rust_pass = tvm.get_global_func("out_of_tree.Pass", pass_dylib)
