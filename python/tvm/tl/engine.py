import tvm
from tvm import tir, tl, relay
from tvm.contrib import nvcc
import os.path as osp

def is_device_call(fn: tir.PrimFunc):
    if fn.attrs and "calling_conv" in fn.attrs and fn.attrs["calling_conv"] == 2:
        return True
    else:
        return False

def is_host_call(fn: tir.PrimFunc):
    return not is_device_call(fn)

@tvm.register_func("tvm_tl_cuda_compile", override=True)
def tvm_callback_cuda_compile(code, target):
    tvm_root = osp.join(osp.dirname(__file__), "../../..")
    tl_template_path = osp.abspath(osp.join(tvm_root, "src/tl"))
    ptx = nvcc.compile_cuda(code, target_format="ptx", options=["-std=c++17", "-I"+tl_template_path])
    return ptx

def extrac_params(fn: tir.PrimFunc):
    buffers = [fn.buffer_map[var] for var in fn.params]
    tensor_types = [relay.TensorType(buffer.shape, buffer.dtype) for buffer in buffers]
    return tensor_types

def compile(fn):
    params = extrac_params(fn)
    target_host = tvm.target.Target("llvm -keys=cpu")
    target = tvm.target.Target("cuda", target_host)

    mod = tvm.IRModule({fn.attrs["global_symbol"]: fn})

    mod = tl.transform.FrontendLegalize()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tl.transform.LayoutInference()(mod)
    mod = tl.transform.LowerTileOp()(mod)

    mod = tir.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    mod = tl.transform.PipelinePlanning()(mod)
    mod = tl.transform.InjectSoftwarePipeline()(mod)
    mod = tir.transform.LowerOpaqueBlock()(mod)
    mod = tir.transform.FlattenBuffer()(mod)
    mod = tir.transform.NarrowDataType(32)(mod)
    mod = tir.transform.Simplify()(mod)

    mod = tir.transform.VectorizeLoop()(mod)
    mod = tir.transform.StorageRewrite()(mod)
    mod = tir.transform.UnrollLoop()(mod)
    mod = tir.transform.RenormalizeSplitPattern()(mod)
    mod = tir.transform.Simplify()(mod)
    mod = tir.transform.RemoveNoOp()(mod)
    mod = tir.transform.RewriteUnsafeSelect()(mod)
    mod = tir.transform.HoistIfThenElse()(mod)

    mod = tir.transform.BindTarget(target)(mod)
    mod = tir.transform.VerifyMemory()(mod)
    mod = tir.transform.AnnotateEntryFunc()(mod)
    mod = tir.transform.ThreadSync("shared")(mod)
    mod = tir.transform.ThreadSync("shared.dyn")(mod)
    mod = tir.transform.MergeDynamicSharedMemoryAllocations()(mod)
    mod = tir.transform.InjectPTXAsyncCopy()(mod)

    mod = tir.transform.AnnotateDeviceRegions()(mod)
    mod = tir.transform.SplitHostDevice()(mod)
    mod = tir.transform.MakePackedAPI()(mod)
    mod = tir.transform.LowerDeviceKernelLaunch()(mod)

    host_mod = tir.transform.Filter(is_host_call)(mod)
    host_mod = tir.transform.BindTarget(target_host)(host_mod)
    host_mod = tir.transform.FP8StorageLegalize()(host_mod)
    host_mod = tir.transform.BF16StorageLegalize()(host_mod)
    host_mod = tir.transform.LowerTVMBuiltin()(host_mod)
    host_mod = tir.transform.LowerCustomDatatypes()(host_mod)
    host_mod = tir.transform.LowerIntrin()(host_mod)
    host_mod = tir.transform.LowerDeviceStorageAccessInfo()(host_mod)
    host_mod = tir.transform.CombineContextCall()(host_mod)
    host_mod = tvm._ffi.get_global_func("target.build.llvm")(host_mod, target)

    device_mod = tir.transform.Filter(is_device_call)(mod)
    device_mod = tir.transform.LowerDeviceStorageAccessInfo()(device_mod)
    device_mod = tir.transform.LowerIntrin()(device_mod)
    device_mod = tir.transform.Simplify()(device_mod)
    # code = tvm._ffi.get_global_func("target.build.tl_debug_codegen")(device_mod, target)
    # print(code)
    device_mod = tvm._ffi.get_global_func("target.build.tl")(device_mod, target)

    host_mod.import_module(device_mod)
    return host_mod, params
