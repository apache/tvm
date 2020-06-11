import tvm
import tvm.relay
from tvm.ir.transform import PassContext

x = tvm.relay.var("x", shape=(10,))
test_func = tvm.relay.Function([x], x)
test_mod = tvm.IRModule.from_expr(test_func)

pass_dylib = "/Users/jroesch/Git/tvm/rust/target/debug/libmy_pass.dylib"

def load_rust_extension(ext_dylib):
    load_so = tvm.get_global_func("runtime.module.loadfile_so")
    mod = load_so(ext_dylib)
    mod.get_function("initialize")()


def load_pass(pass_name, dylib):
    load_rust_extension(dylib)
    return tvm.get_global_func(pass_name)

MyPass = load_pass("out_of_tree.Pass", pass_dylib)
ctx = PassContext()
import pdb; pdb.set_trace()
f = MyPass(test_func, test_mod, ctx)
mod = MyPass()(test_mod)

print(mod)
