import numpy as np
import tvm
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.relax.frontend.torch import from_exported_program
from tvm import IRModule, relax
from tvm.contrib.download import download_testdata
import torch
from torch.export import export
from torchvision.models.resnet import ResNet18_Weights, resnet18
import os, sys
from tvm.relax.frontend.change_datatype import ChangeDatatype
from PIL import Image
import time
from tvm import relax, tir, IRModule
from sch_handed import add_parallel_directives_to_all_functions
from register import _posit_registered
from tvm.relax.transform import ToMixedPrecision


_posit_registered()
    
def get_cat_image(): # Download and preprocess a cat image for testing
    url = "https://gist.githubusercontent.com/zhreshold/bcda4716699ac97ea44f791c24310193/raw/fa7ef0e9c9a5daea686d6473a62aacd1a5885849/cat.png"
    dst = "cat.png"
    real_dst = download_testdata(url, dst, module="data")
    img = Image.open(real_dst).resize((224, 224))
    
    # Preprocess the image using PyTorch's ResNet-18 weights
    weights = ResNet18_Weights.DEFAULT
    preprocess = weights.transforms()
    img_tensor = preprocess(img).unsqueeze(0) #(3, 224, 224) -> (1, 3, 224, 224)
    
    # Convert to numpy array with the correct data type
    img_array = img_tensor.numpy()
    return np.asarray(img_array, dtype="float32")

def export_resnet18():
    model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()
    example_args = (torch.randn(1, 3, 224, 224, dtype=torch.float32),)
    with torch.no_grad():
        exported_program = export(model, example_args)
        relax_module = from_exported_program(exported_program, keep_params_as_input=True)
    relax_module, params = relax.frontend.detach_params(relax_module)
    relax_module = relax.transform.DecomposeOpsForInference()(relax_module)
    return relax_module, params

def convert_ndarray_in_relax(dst_dtype, array, dev=tvm.cpu(), target="llvm"):
    """
    input = numpy.ndarray or tvm.nd.NDArray：
      - if input == numpy.ndarray -> tvm.nd.NDArray
      - Relax VM uses astype return tvm.nd.NDArray
    """
    if isinstance(array, list):
        return [convert_ndarray_in_relax(dst_dtype, v, dev, target) for v in array]
    if isinstance(array, np.ndarray):
        array = tvm.runtime.tensor(array, dev)
    shape = array.shape
    src_dtype = str(array.dtype)
    mod = IRModule()
    x = relax.Var("x",
        relax.TensorStructInfo(shape=shape, dtype=src_dtype)
    )
    body = relax.op.astype(x, dst_dtype)
    ret_sinfo = relax.TensorStructInfo(shape=shape, dtype=dst_dtype)
    func = relax.Function(
        params=[x],
        body=body,
        ret_struct_info=ret_sinfo
    )
    mod["main"] = func
    with tvm.transform.PassContext(config={"tir.disable_vectorize": True}):
        rt_mod = relax.build(mod, target=target)
    vm = relax.VirtualMachine(rt_mod, dev)
    return vm["main"](array)

def ChangeDatatypeInRelaxAndParams(mod: tvm.IRModule, params: dict, src_dtype: str , dst_dtype: str):
    """
    Convert the data type of all tensors in the Relax module to the specified dtype.
    """
    dtype_mutator = ChangeDatatype(src_dtype, dst_dtype, mod)
    new_main = dtype_mutator.visit_expr(mod["main"])
    new_mod = IRModule({"main": new_main})
    """
    Convert the data type of all input and params to the specified dtype.
    """
    params = {k: convert_ndarray_in_relax(dst_dtype, v) for k, v in params.items()}
    return new_mod, params

def run_inference(mod: tvm.IRModule, params: dict, input_data, target="llvm"):
    # Convert input data to the specified dtype
    dev = tvm.device(target, 0)
    # input_data = convert_ndarray_in_relax("float16", input_data, dev=dev)
    # Run inference on the Relax module with the given parameters and input data
    print("="*60)
    print("Start Building IRModule...")
    start_time = time.time()
    rt_mod = relax.build(mod, target=target)
    end_time = time.time()
    print("Build Time:", end_time - start_time, "seconds")
    print("-"*60)

    # Run the model
    print("Start Running Inference...")
    start_time = time.time()
    vm = relax.VirtualMachine(rt_mod, dev)
    output = vm["main"](input_data, *params["main"])
    end_time = time.time()
    print("Inference Time:", end_time - start_time, "seconds")
    print("="*60)
    return output

def benchmark_inference_float32():
    target = "llvm"
    mod, params = export_resnet18()
    input_data = get_cat_image() 
    dev = tvm.device(target, 0)
    rt_mod = relax.build(mod, target=target)
    vm = relax.VirtualMachine(rt_mod, dev)
    input_data = tvm.runtime.tensor(input_data, dev)
    output = vm["main"](input_data, *params["main"])
    return output

def main():
    src_dtype = "float32"
    dst_dtype = "custom[posites2]16"
    # dst_dtype = "float16"

    mod, params = export_resnet18()
    # mod = ToMixedPrecision(out_dtype="float16")(mod)
    print(mod)
    mod, params = ChangeDatatypeInRelaxAndParams(mod, params, src_dtype=src_dtype, dst_dtype=dst_dtype)  # Change data type to posit
    # mod, params = ChangeDatatypeInRelaxAndParams(mod, params, src_dtype="float32", dst_dtype="float16")  # Change data type to posit
    print(mod)

    mod = tvm.relax.pipeline.get_pipeline()(mod) # Lower to TIR
    shc = add_parallel_directives_to_all_functions(mod)  # Add parallel directives to all functions
    # # print(shc)
    input_data = convert_ndarray_in_relax(dst_dtype=dst_dtype, array= get_cat_image())
    output = run_inference(shc, params, input_data , target="llvm --num-cores=8")  # Run inference
    output = convert_ndarray_in_relax("float32", output[0])
    print("Output:\n", output)
    float32_out = benchmark_inference_float32()[0]
    np.testing.assert_allclose(
        float32_out.numpy(), output.numpy(), rtol=1e-5, atol=1e-5
    )
if __name__ == "__main__":
    main()
