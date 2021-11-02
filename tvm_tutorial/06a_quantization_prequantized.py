# Using framework prequantized models in TVM
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


def run_tvm_model(mod, params, input_name, inp, target="llvm", opt_level=3, instrument_passes=False, profile=False):
    instruments = []
    if instrument_passes:
        timing_inst = tvm.ir.instrument.PassTimingInstrument()
        instruments.append(timing_inst)

    with tvm.transform.PassContext(opt_level=opt_level, instruments=instruments):
        lib = relay.build(mod, target=target, params=params)

        if instrument_passes:
            pass_profiles = timing_inst.render()
            print(f"TVM passes (O{opt_level}):")
            print(pass_profiles)

    if profile:
        dev = tvm.device(target, 0)
        runtime = tvm.contrib.debugger.debug_executor.GraphModuleDebug(lib["debug_create"]("default", dev), [dev], lib.get_graph_json(), None)
        runtime.set_input(input_name, inp)
        debug_profile = runtime.profile()
        print("Per operator profile breakdown")
        print(debug_profile)
    else:
        runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.device(target, 0)))
        runtime.set_input(input_name, inp)
        runtime.run()
    return runtime.get_output(0).numpy(), runtime



synset = get_synset()
inp = get_imagenet_input()


def quantize_model(model, inp):
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)
    # Dummy calibration
    model(inp)
    torch.quantization.convert(model, inplace=True)

print("Using a framework prequantized model in TVM")
print("")
qmodel = qmobilenet.mobilenet_v2(pretrained=True).eval()
#qmodel = qshufflenet.shufflenet_v2_x1_0(pretrained=True).eval()
print("Torch model(unquantized):")
print(qmodel)   

pt_inp = torch.from_numpy(inp)
quantize_model(qmodel, pt_inp)

print("Torch model (quantized):")
print(qmodel)

script_module = torch.jit.trace(qmodel, pt_inp).eval()

with torch.no_grad():
    pt_result = script_module(pt_inp).numpy()


input_name = "input"  # the input name can be be arbitrary for PyTorch frontend.
input_shapes = [(input_name, (1, 3, 224, 224))]
mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

print("Relay module (quantized)")
print(mod)


target="llvm -mcpu=skylake"
#target="llvm"

print(f"Running on target: {target}")
tvm_result, rt_mod = run_tvm_model(mod, params, input_name, inp, target=target)


pt_top3_labels = np.argsort(pt_result[0])[::-1][:3]
tvm_top3_labels = np.argsort(tvm_result[0])[::-1][:3]

print("PyTorch top3 labels:", ", ".join([f'"{synset[label]}"' for label in pt_top3_labels]))
print("TVM top3 labels:", ", ".join([f'"{synset[label]}"' for label in tvm_top3_labels]))

n_repeat = 100  # should be bigger to make the measurement more accurate
dev = tvm.cpu(0)

print("Execution time summary (quantized):")
print(rt_mod.benchmark(dev, number=1, repeat=n_repeat))

print("Doing per operator profiling")
tvm_result, rt_mod = run_tvm_model(mod, params, input_name, inp, target=target, profile=True)
