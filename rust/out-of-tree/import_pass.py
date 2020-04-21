import tvm

pass_dylib = "/Users/jroesch/Git/tvm/rust/target/debug/libmy_pass.dylib"

def load_pass(pass_name, pass_dylib):
    load_so = tvm.get_global_func("runtime.module.loadfile_so")
    mod = load_so(pass_dylib)
    mod.get_function("initialize")()
    return tvm.get_global_func(pass_name)()

rust_pass = load_pass("RustPass", pass_dylib)

print(rust_pass)
