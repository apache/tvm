# Example inference on microtvm
# Using framework prequantized models in TVM
import pathlib
import tarfile
import subprocess
import shutil

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
target += " -system-lib=1 -runtime=c"

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

model_library_format_tar_path = pathlib.Path("build/07a/module.tar")
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
project_options = {'project_type': 'host_driven'}

generated_project_dir = model_library_format_tar_path.parent.absolute() / "generated-project"
if generated_project_dir.exists():
    shutil.rmtree(generated_project_dir, ignore_errors=True)

generated_project = tvm.micro.generate_project(
    template_project_path, module, generated_project_dir, project_options
)

# Build and flash the project
generated_project.build()
generated_project.flash()

with tvm.micro.Session(transport_context_manager=generated_project.transport()) as session:
    graph_mod = tvm.micro.create_local_graph_executor(
        module.get_graph_json(), session.get_system_lib(), session.device
    )

    graph_mod.set_input(**module.get_params())

    graph_mod.set_input(input_name, inp)
    graph_mod.run()

    tvm_result = graph_mod.get_output(0).numpy()

    pt_top3_labels = np.argsort(pt_result[0])[::-1][:3]
    tvm_top3_labels = np.argsort(tvm_result[0])[::-1][:3]

    print("PyTorch top3 labels:", ", ".join([f'"{synset[label]}"' for label in pt_top3_labels]))
    print("TVM top3 labels:", ", ".join([f'"{synset[label]}"' for label in tvm_top3_labels]))


# Unfortunately need to rebuild for debug runtime
if generated_project_dir.exists():
    shutil.rmtree(generated_project_dir, ignore_errors=True)

generated_project = tvm.micro.generate_project(
    template_project_path, module, generated_project_dir, project_options
)

# Build and flash the project
generated_project.build()
generated_project.flash()

with tvm.micro.Session(transport_context_manager=generated_project.transport()) as session:
    graph_mod = tvm.micro.create_local_debug_executor(
        module.get_graph_json(), session.get_system_lib(), session.device
    )
    graph_mod.set_input(**module.get_params())

    debug_profile = graph_mod.profile()
    
    print("Per operator profile breakdown")
    print(debug_profile)    