import traceback
import tvm
from tvm import relay
from tvm import runtime
from tvm.relay import transform
from tvm.relay.op.contrib.register import get_pattern_table
import numpy as np
import time
import os
from tvm.contrib.cutlass import gen_gemm


M = 1820
N = 768
K = 768


def get_cpu_op_count(mod):
    """Traverse graph counting ops offloaded to TVM."""

    class Counter(tvm.relay.ExprVisitor):
        def __init__(self):
            super().__init__()
            self.count = 0

        def visit_call(self, call):
            if isinstance(call.op, tvm.ir.Op):
                self.count += 1

            super().visit_call(call)

    c = Counter()
    c.visit(mod["main"])
    return c.count


data = relay.var("data", shape=(M, K), dtype="float16")
weight = relay.var("weight", shape=(N, K), dtype="float16")
bias = relay.var("bias", shape=(N,), dtype="float16")
gemm_out = relay.nn.dense(data, weight)
# gemm_out = relay.nn.bias_add(gemm_out, bias)
# gemm_out = relay.nn.relu(gemm_out)
# gemm_out = relay.nn.gelu(gemm_out)
# gemm_out = relay.nn.dense(gemm_out, weight)
out = gemm_out

mod = tvm.IRModule.from_expr(out)
### dataflow rewrite
mod = transform.MergeComposite(get_pattern_table("cutlass"))(mod)
mod = transform.AnnotateTarget(["cutlass"])(mod)
# mod = transform.MergeCompilerRegions()(mod) // we don't need this in byoc cutlass
mod = transform.PartitionGraph()(mod)
# mod = transform.InferType()(mod)
is_nt = False
print(mod)


class GemmCollector(tvm.relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.signature = {}

    def visit_call(self, call):
        op = call.op
        if isinstance(op, relay.Function) and "PartitionedFromPattern" in op.attrs:
            self.signature["op_type"] = op.attrs["Composite"]
            for i, arg in enumerate(op.params):
                self.signature["arg%d_shape" % i] = arg.checked_type.shape
                self.signature["arg%d_dtype" % i] = arg.checked_type.dtype
            self.signature["ret_shape"] = op.ret_type.shape
            self.signature["ret_dtype"] = op.ret_type.dtype


cutlass_profiler = gen_gemm.CutlassGemmProfiler("75", "../../../3rdparty/cutlass", "./temp")

for var in mod.get_global_vars():
    fun_name = var.name_hint
    func = mod[fun_name]
    collector = GemmCollector()
    if "cutlass" in fun_name:
        collector.visit(func)
        # call cutlass profiler to find best settings, update attr
        new_attrs = {}
        new_attrs.update(collector.signature)
        for key in func.attrs.keys():
            new_attrs[key] = func.attrs[key]
        # call profiler
        arg0_shape = new_attrs["arg0_shape"]
        arg1_shape = new_attrs["arg1_shape"]
        MM = arg0_shape[0]
        KK = arg0_shape[1]
        NN = arg1_shape[0]
        out = cutlass_profiler.profile(
            "GenerateSM75_TensorOp_1688", new_attrs["arg0_dtype"], MM, NN, KK
        )
        if new_attrs["op_type"] == "cutlass.dense":
            new_attrs["cutlass_op_def"] = out["opdef"]
        elif new_attrs["op_type"] == "cutlass.dense_bias":
            new_attrs["cutlass_op_def"] = out["opdef_bias"]
        elif new_attrs["op_type"] == "cutlass.dense_bias_relu":
            new_attrs["cutlass_op_def"] = out["opdef_bias_relu"]
        elif new_attrs["op_type"] == "cutlass.dense_bias_gelu":
            new_attrs["cutlass_op_def"] = out["opdef_bias_gelu"]
        else:
            raise ValueError("%s pattern is not implemented." % new_attrs["op_type"])
        new_attrs["cutlass_op_name"] = out["name"]

        print("The best kernel is " + new_attrs["cutlass_op_name"])
        if new_attrs["cutlass_op_name"].find("_tn_align") > 0:
            new_attrs["lda"] = "K"
            new_attrs["ldb"] = "K"
            new_attrs["ldc"] = "N"
        elif new_attrs["cutlass_op_name"].find("_nt_align") > 0:
            new_attrs["lda"] = "M"
            new_attrs["ldb"] = "N"
            new_attrs["ldc"] = "N"
            is_nt = True
        else:
            raise ValueError("%s unsupported operation" % new_attrs["cutlass_op_name"])
        new_attrs = tvm.ir.make_node("DictAttrs", **new_attrs)
        new_func = relay.Function(
            func.params,
            func.body,
            ret_type=func.ret_type,
            type_params=func.type_params,
            attrs=new_attrs,
        )
        mod.update_func(var, new_func)

target = "cuda"
np_data = np.random.uniform(-1, 1, (M, K)).astype("float16")
np_weight = np.random.uniform(-1, 1, (N, K)).astype("float16")
np_bias = np.random.uniform(-1, 1, (N,)).astype("float16")

if is_nt:
    tvm_data = np_data.T
    tvm_weight = np_weight.T
    tvm_bias = np_bias
else:
    tvm_data = np_data
    tvm_weight = np_weight
    tvm_bias = np_bias

params = {"weight": tvm_weight, "bias": tvm_bias}

print("compiling...")
with tvm.transform.PassContext(opt_level=3):
    # json, lib, param = relay.build(mod, target=target, params=params)
    # print("====================")
    # print(lib.imported_modules[1].get_source())
    lib = relay.build(mod, target=target, params=params)


lib_path = "compiled.so"
# cutlass_path = "../../../3rdparty/cutlass/include"
# cutlass_util_path = "../../../3rdparty/cutlass/tools/util/include"
cutlass_path = "/home/masa/projects/dev/tvm/3rdparty/cutlass/include"
cutlass_util_path = "/home/masa/projects/dev/tvm/3rdparty/cutlass/tools/util/include"
workdir = "tmp"

os.makedirs(workdir, exist_ok=True)

kwargs = {}
kwargs["cc"] = "nvcc"
kwargs["options"] = [
    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
    "-gencode=arch=compute_75,code=[sm_75,compute_75]",
    "-Xcompiler=-fPIC",
    "-Xcompiler=-Wconversion",
    "-Xcompiler=-fno-strict-aliasing",
    "-O3",
    "-std=c++14",
    "-I" + cutlass_path,
    "-I" + cutlass_util_path,
]
lib.export_library(lib_path, workspace_dir=workdir, **kwargs)
lib = runtime.load_module(lib_path)


ctx = tvm.gpu()
# rt_mod = tvm.contrib.graph_runtime.create(json, lib, ctx)
rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](ctx))

x = tvm.nd.array(tvm_data, device=ctx)
rt_mod.set_input("data", x)

print("Running for the first time...")
rt_mod.run()
y = rt_mod.get_output(0)

print("np computing...")
np_out = np.dot(np_data, np_weight.T)
# np_out = np.dot(np_out, np_weight.T)
# np_out = np_out + np_bias
# np_out = np_out * (np_out > 0)
# np_out = np_out*(0.5+erf(np_out * np.sqrt(0.5)) * 0.5)


try:
    np.testing.assert_allclose(y.asnumpy(), np_out, atol=1e-2, rtol=1e-2)
    print("Accuracy test passed...")
except:
    traceback.print_exc()
    print("Accuracy test failed...")


times = []
for i in range(100):
    start = time.time()
    rt_mod.run()
    ctx.sync()  # wait for the device to finish
    times.append(time.time() - start)
print("Latency:", 1000.0 * np.mean(times), "ms")
print("TFLOPS:", 2 * M * N * K / np.mean(times) / 1e12)
