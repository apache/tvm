# Example inference on microtvm
# Using framework prequantized models in TVM
import pathlib
import tarfile
import subprocess
import shutil
import sys

from PIL import Image

import numpy as np

import torch
from torchvision.models.quantization import mobilenet as qmobilenet
from torchvision.models.quantization import shufflenetv2 as qshufflenet

import tvm
from tvm import relay
from tvm.contrib.download import download_testdata


def get_transform():
    import torchvision.transforms as transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )


def get_real_image(im_height, im_width):
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_path = download_testdata(img_url, "cat.png", module="data")

    return Image.open(img_path).resize((im_height, im_width))


def get_imagenet_input():
    im = get_real_image(224, 224)
    preprocess = get_transform()
    pt_tensor = preprocess(im)
    return np.expand_dims(pt_tensor.numpy(), 0)


def get_synset():
    synset_url = "".join(
        [
            "https://gist.githubusercontent.com/zhreshold/",
            "4d0b62f3d01426887599d4f7ede23ee5/raw/",
            "596b27d23537e5a1b5751d2b0481ef172f58b539/",
            "imagenet1000_clsid_to_human.txt",
        ]
    )
    synset_name = "imagenet1000_clsid_to_human.txt"
    synset_path = download_testdata(synset_url, synset_name, module="data")
    with open(synset_path) as f:
        return eval(f.read())


synset = get_synset()
inp = get_imagenet_input()


def quantize_model(model, inp):
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)
    # Dummy calibration
    model(inp)
    torch.quantization.convert(model, inplace=True)

print("An introduction to microtvm")
print("")
qmodel = qmobilenet.mobilenet_v2(pretrained=True).eval()

pt_inp = torch.from_numpy(inp)
quantize_model(qmodel, pt_inp)


script_module = torch.jit.trace(qmodel, pt_inp).eval()

with torch.no_grad():
    pt_result = script_module(pt_inp).numpy()


input_name = "input"
input_shapes = [(input_name, (1, 3, 224, 224))]
mod, params = relay.frontend.from_pytorch(script_module, input_shapes)


#target="llvm -mcpu=skylake"
target = "c"
target += " -runtime=c --link-params --executor=aot --unpacked-api=1"

with tvm.transform.PassContext(
    opt_level=3, 
    config={"tir.disable_vectorize": True}, 
    disabled_pass=["AlterOpLayout"]
):
    module = relay.build(mod, target=target, params=params)


#for source_module in module.get_lib().imported_modules:
#    source_code = source_module.get_source()
#
#    print("Source Code:")
#    print(source_code)



# Model Library Format: http://tvm.apache.org/docs/arch/model_library_format.html

model_library_format_tar_path = pathlib.Path("build/08b/module.tar")
model_library_format_tar_path.unlink(missing_ok=True)
model_library_format_tar_path.parent.mkdir(parents=True, exist_ok=True)

tvm.micro.export_model_library_format(module, model_library_format_tar_path)

print("Built MLF Library: ")
with tarfile.open(model_library_format_tar_path, "r:*") as tar_f:
    print("\n".join(f" - {m.name}" for m in tar_f.getmembers()))
    tar_f.extractall(model_library_format_tar_path.parent)

#repo_root = pathlib.Path(
#    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], encoding="utf-8").strip()
#)
#template_project_path = repo_root / "src" / "runtime" / "crt" / "host"
#project_options = {}

template_project_path = pathlib.Path(".").absolute() / "microtvm_template"
project_options = {'project_type': 'aot', 'verbose': True}

generated_project_dir = model_library_format_tar_path.parent.absolute() / "generated-project"
if generated_project_dir.exists():
    shutil.rmtree(generated_project_dir, ignore_errors=True)


generated_project = tvm.micro.generate_project(
    template_project_path, module, generated_project_dir, project_options
)

def create_header_file(tensor_name, npy_data, output_path):
    """
    This method generates a header file containing the data contained in the numpy array provided.
    It is used to capture the tensor data (for both inputs and expected outputs) to be bundled into the standalone application.
    """
    file_path = pathlib.Path(f"{output_path}/" + tensor_name).resolve()
    # create header file
    raw_path = file_path.with_suffix(".h").resolve()
    with open(raw_path, "w") as header_file:
        header_file.write("#include <stddef.h>\n")
        header_file.write("#include <stdint.h>\n")
        header_file.write("#include <dlpack/dlpack.h>\n")
        header_file.write(f"const size_t {tensor_name}_len = {npy_data.size};\n")

        if npy_data.dtype == "int8":
            header_file.write(f"int8_t {tensor_name}[] =")
        elif npy_data.dtype == "int32":
            header_file.write(f"int32_t {tensor_name}[] = ")
        elif npy_data.dtype == "uint8":
            header_file.write(f"uint8_t {tensor_name}[] = ")
        elif npy_data.dtype == "float32":
            header_file.write(f"float {tensor_name}[] = ")

        header_file.write("{")
        for i in np.ndindex(npy_data.shape):
            header_file.write(f"{npy_data[i]}, ")
        header_file.write("};\n\n")

create_header_file(input_name, inp, generated_project_dir / "src")
out = np.zeros((1,1000), dtype=np.float32)
create_header_file("output", out, generated_project_dir / "src")


# Build and flash the project
generated_project.build()

