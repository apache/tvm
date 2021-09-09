# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
""" Tests on quantized torch model conversion """
import os

from PIL import Image

import numpy as np

import torch
from torch import nn
from torch.quantization import QuantStub, DeQuantStub
from torch.quantization import fuse_modules, QuantWrapper

import tvm
import tvm.testing
from tvm import relay
from tvm.relay.frontend.pytorch_utils import is_version_greater_than
from tvm.contrib.download import download_testdata
from tvm.relay.op.contrib.register import register_pattern_table, get_pattern_table


def torch_version_check():
    from packaging import version

    return version.parse(torch.__version__) > version.parse("1.4.0")


def get_tvm_runtime(script_module, input_name, ishape):
    input_shapes = [(input_name, ishape)]
    mod, params = relay.frontend.from_pytorch(script_module, input_shapes)

    with tvm.transform.PassContext(opt_level=3):
        # test on only cpu for now, torch cannot run quant models on cuda
        # also not to make CI too slow
        lib = relay.build(mod, target="llvm", params=params)

    runtime = tvm.contrib.graph_executor.GraphModule(lib["default"](tvm.cpu(0)))
    return runtime


def get_qconfig(per_channel):
    from torch.quantization.observer import MovingAverageMinMaxObserver
    from torch.quantization.observer import default_weight_observer

    if per_channel:
        return torch.quantization.get_default_qconfig("fbgemm")
    else:
        act = MovingAverageMinMaxObserver.with_args(reduce_range=False)
        return torch.quantization.QConfig(activation=act, weight=default_weight_observer)


def quantize_model(model, inp, per_channel=False):
    model.fuse_model()
    model.qconfig = get_qconfig(per_channel)
    torch.quantization.prepare(model, inplace=True)
    model(inp)
    torch.quantization.convert(model, inplace=True)


class ConvBn(nn.Module):
    def __init__(self, with_relu=False):
        super().__init__()
        layers = [nn.Conv2d(3, 32, 3, bias=True), nn.BatchNorm2d(32)]
        if with_relu:
            layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)
        self.quant_wrap = QuantWrapper(self.conv)
        self.with_relu = with_relu

    def forward(self, x):
        return self.quant_wrap(x)

    def fuse_model(self):
        indices = ["0", "1"]
        if self.with_relu:
            indices.append("2")
        fuse_modules(self.conv, indices, inplace=True)


class ConvTranspose(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.ConvTranspose2d(3, 32, 3, bias=True)]
        self.conv = nn.Sequential(*layers)
        self.quant_wrap = QuantWrapper(self.conv)

    def forward(self, x):
        return self.quant_wrap(x)

    def fuse_model(self):
        pass


class Linear(nn.Module):
    def __init__(self, with_relu=False):
        super().__init__()
        layers = [nn.Linear(16, 32)]
        if with_relu:
            layers.append(nn.ReLU())
        self.fc = nn.Sequential(*layers)
        self.quant_wrap = QuantWrapper(self.fc)
        self.with_relu = with_relu

    def forward(self, x):
        return self.quant_wrap(x)

    def fuse_model(self):
        if self.with_relu:
            fuse_modules(self.fc, ["0", "1"], inplace=True)


class ReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = QuantWrapper(nn.ReLU())

    def forward(self, x):
        return self.relu(x)

    def fuse_model(self):
        pass


# Mobilenet V3 related modules
class Hsigmoid(nn.Module):
    def __init__(self, add_stub=False):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.add_stub = add_stub
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        if self.add_stub:
            x = self.quant(x)
        x = self.hsigmoid(x)
        if self.add_stub:
            x = self.dequant(x)
        return x

    def fuse_model(self):
        pass


class Hswish(nn.Module):
    def __init__(self, add_stub=False):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.add_stub = add_stub
        self.hswish = nn.Hardswish()

    def forward(self, x):
        if self.add_stub:
            x = self.quant(x)
        x = self.hswish(x)
        if self.add_stub:
            x = self.dequant(x)
        return x

    def fuse_model(self):
        pass


class SqueezeExcite(nn.Module):
    def __init__(self, channel, reduction=4, add_stub=False):
        super(SqueezeExcite, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid(add_stub=False),
        )
        self.fmul = nn.quantized.FloatFunctional()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.add_stub = add_stub

    def forward(self, x):
        b, c, _, _ = x.size()
        if self.add_stub:
            x = self.quant(x)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        out = self.fmul.mul(x, y.expand_as(x))
        if self.add_stub:
            return self.dequant(out)
        else:
            return out

    def fuse_model(self):
        fuse_modules(self.fc, ["0", "1"], inplace=True)


# test on quantized::mul_scalar with negative scale
class MulScalarNegative(nn.Module):
    def __init__(self):
        super().__init__()
        self.float_op = nn.quantized.FloatFunctional()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        mul = self.float_op.mul_scalar(x, -0.3)
        return self.dequant(mul)

    def fuse_model(self):
        pass


class UpsamplingBilinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        upsample = nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return self.dequant(upsample)

    def fuse_model(self):
        pass


class AvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = QuantWrapper(nn.AvgPool2d(kernel_size=2))

    def forward(self, x):
        return self.pool(x)

    def fuse_model(self):
        pass


class AdaptiveAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = QuantWrapper(nn.AdaptiveAvgPool2d((1, 1)))

    def forward(self, x):
        return self.pool(x)

    def fuse_model(self):
        pass


def test_quantized_modules():
    imagenet_ishape = (1, 3, 224, 224)

    qmodules = [
        ("relu", imagenet_ishape, ReLU(), False),
        ("upsample bilinear", (1, 3, 64, 64), UpsamplingBilinear(), False),
        ("avgpool", imagenet_ishape, AvgPool2d(), False),
    ]

    for per_channel in [False, True]:
        if per_channel:
            postfix = ", per_channel"
        else:
            postfix = ""

        qmodules += [
            ("conv_bn" + postfix, imagenet_ishape, ConvBn(), per_channel),
            ("conv_bn_relu" + postfix, imagenet_ishape, ConvBn(with_relu=True), per_channel),
            ("linear" + postfix, (16, 16), Linear(), per_channel),
            ("linear_relu" + postfix, (16, 16), Linear(with_relu=True), per_channel),
            ("conv_transpose", imagenet_ishape, ConvTranspose(), False),
            ("hsigmoid", imagenet_ishape, Hsigmoid(add_stub=True), False),
            ("hswish", imagenet_ishape, Hswish(add_stub=True), False),
            ("semodule", (1, 16, 64, 64), SqueezeExcite(16, add_stub=True), False),
            ("semodule, per_channel", (1, 16, 64, 64), SqueezeExcite(16, add_stub=True), True),
            ("mul_scalar negative", imagenet_ishape, MulScalarNegative(), False),
        ]

    for (module_name, ishape, raw_module, per_channel) in qmodules:
        raw_module.eval()
        inp = torch.rand(ishape)

        # quantized conv_transpose2d is supported only with qnnpack engine before torch v1.8.0.
        if module_name == "conv_transpose" and not is_version_greater_than("1.7.1"):
            prev_engine = torch.backends.quantized.engine
            torch.backends.quantized.engine = "qnnpack"
            quantize_model(raw_module, inp, per_channel=per_channel)
            torch.backends.quantized.engine = prev_engine
        else:
            quantize_model(raw_module, inp, per_channel=per_channel)

        script_module = torch.jit.trace(raw_module, inp).eval()

        with torch.no_grad():
            pt_result = script_module(inp.clone()).numpy()

        input_name = "input"
        runtime = get_tvm_runtime(script_module, input_name, ishape)
        runtime.set_input(input_name, inp.numpy().copy())
        runtime.run()
        tvm_result = runtime.get_output(0).numpy()

        max_abs_diff = np.max(np.abs(tvm_result - pt_result))
        mean_abs_diff = np.mean(np.abs(tvm_result - pt_result))
        num_identical = np.sum(tvm_result == pt_result)
        match_ratio = num_identical / float(np.prod(tvm_result.shape))

        print(module_name, max_abs_diff, mean_abs_diff, match_ratio)

        # sample outputs
        """
        relu 0.0039215684 2.6052087e-08 0.9999933567176871
        upsample bilinear 0.0 0.0 1.0
        conv_bn 0.22062653 0.011478779 0.6909348115006899
        conv_bn_relu 0.3700896 0.010921672 0.7489366477964451
        linear 0.15987062 0.009231662 0.794921875
        linear_relu 0.14180502 0.0053220326 0.8828125
        conv_transpose 0.0033792555 4.4658788e-07 0.9998678439971806
        conv_bn, per_channel 0.01654929 2.9486866e-06 0.9998218235127019
        conv_bn_relu, per_channel 0.009089053 1.4926576e-06 0.9998357732732732
        linear, per_channel 0.0 0.0 1.0
        linear_relu, per_channel 0.0 0.0 1.0
        hsigmoid 0.002614379 0.00020525524 0.9214896896258503
        hswish 0.0052286386 0.00063522335 0.7587359162414966
        semodule, per_channel 0.0039885044 0.0008620687 0.7838592529296875
        mul_scalar negative 0.0011764616 7.815566e-09 0.9999933567176871
        """

        # we cannot make any guarantee on how close the raw output is to torch
        # tvm.testing.assert_allclose(tvm_result, pt_result, rtol=1e-1, atol=1e-1)


def test_quantized_imagenet():
    def get_transform():
        import torchvision.transforms as transforms

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
        )

    def get_real_image(im_height, im_width):
        repo_base = "https://github.com/dmlc/web-data/raw/main/tensorflow/models/InceptionV1/"
        img_name = "elephant-299.jpg"
        image_url = os.path.join(repo_base, img_name)
        img_path = download_testdata(image_url, img_name, module="data")
        return Image.open(img_path).resize((im_height, im_width))

    def get_imagenet_input():
        im = get_real_image(224, 224)
        preprocess = get_transform()
        pt_tensor = preprocess(im)
        return np.expand_dims(pt_tensor.numpy(), 0)

    from torchvision.models.quantization import resnet as qresnet
    from torchvision.models.quantization import mobilenet as qmobilenet
    from torchvision.models.quantization import inception as qinception
    from torchvision.models.quantization import googlenet as qgooglenet

    qmodels = []

    for per_channel in [False, True]:
        qmodels += [
            ("resnet18", qresnet.resnet18(pretrained=True), per_channel),
            ("mobilenet_v2", qmobilenet.mobilenet_v2(pretrained=True), per_channel),
            # disable inception test for now, since loading it takes ~5min on torchvision-0.5 due to scipy bug
            # See https://discuss.pytorch.org/t/torchvisions-inception-v3-takes-much-longer-to-load-than-other-models/68756
            # ("inception_v3", qinception.inception_v3(pretrained=True), per_channel),
            # tracing quantized googlenet broken as of v1.6
            # ("googlenet", qgooglenet(pretrained=True), per_channel),
        ]

    if is_version_greater_than("1.7.1"):
        from torchvision.models.quantization import mobilenet_v3_large as qmobilenet_v3_large

        qmodels.append(
            ("mobilenet_v3_large", qmobilenet_v3_large(pretrained=True, quantize=True).eval(), True)
        )

    results = []

    for (model_name, raw_model, per_channel) in qmodels:
        raw_model.eval()

        if per_channel:
            model_name += ", per channel quantization"
        else:
            model_name += ", per tensor quantization"

        inp = get_imagenet_input()
        pt_inp = torch.from_numpy(inp)

        if "mobilenet_v3_large" not in model_name:
            # mv3 was qat-ed, quantize=True option above makes it already quantized
            quantize_model(raw_model, pt_inp, per_channel=per_channel)

        script_module = torch.jit.trace(raw_model, pt_inp).eval()

        with torch.no_grad():
            pt_result = script_module(pt_inp).numpy()

        input_name = "image"
        runtime = get_tvm_runtime(script_module, input_name, (1, 3, 224, 224))
        runtime.set_input(input_name, inp)
        runtime.run()

        tvm_result = runtime.get_output(0).numpy()

        results.append((model_name, pt_result[0], tvm_result[0]))

    for (model_name, pt_result, tvm_result) in results:
        max_abs_diff = np.max(np.abs(tvm_result - pt_result))
        mean_abs_diff = np.mean(np.abs(tvm_result - pt_result))
        num_identical = np.sum(tvm_result == pt_result)
        pt_top3_labels = np.argsort(pt_result)[::-1][:3]
        tvm_top3_labels = np.argsort(tvm_result)[::-1][:3]

        print("\nModel name: %s" % model_name)
        print("PyTorch top3 label:", pt_top3_labels)
        print("TVM top3 label:", tvm_top3_labels)
        print("max abs diff:", max_abs_diff)
        print("mean abs_diff:", mean_abs_diff)
        print("%d in 1000 raw outputs identical." % num_identical)

        assert set(pt_top3_labels) == set(tvm_top3_labels)

        # sample outputs
        """
        Model name: resnet18, per tensor quantization
        PyTorch top3 label: [386 101 385]
        TVM top3 label: [386 101 385]
        max abs diff: 0.65681696
        mean abs_diff: 0.14055882
        236 in 1000 raw outputs identical.

        Model name: mobilenet_v2, per tensor quantization
        PyTorch top3 label: [101 386 385]
        TVM top3 label: [101 386 385]
        max abs diff: 2.1262953
        mean abs_diff: 0.41025686
        101 in 1000 raw outputs identical.

        Model name: inception_v3, per tensor quantization
        PyTorch top3 label: [101 386 385]
        TVM top3 label: [101 386 385]
        max abs diff: 0.9994669
        mean abs_diff: 0.098697364
        272 in 1000 raw outputs identical.

        Model name: googlenet, per tensor quantization
        PyTorch top3 label: [101 386 385]
        TVM top3 label: [101 386 385]
        max abs diff: 0.28248847
        mean abs_diff: 0.0634469
        274 in 1000 raw outputs identical.

        Model name: resnet18, per channel quantization
        PyTorch top3 label: [101 386 385]
        TVM top3 label: [101 386 385]
        max abs diff: 0.65908074
        mean abs_diff: 0.1274223
        469 in 1000 raw outputs identical.

        Model name: mobilenet_v2, per channel quantization
        PyTorch top3 label: [101 386 385]
        TVM top3 label: [101 386 385]
        max abs diff: 0.71120834
        mean abs_diff: 0.15883648
        423 in 1000 raw outputs identical.

        Model name: inception_v3, per channel quantization
        PyTorch top3 label: [386 101 385]
        TVM top3 label: [386 101 385]
        max abs diff: 1.3372154
        mean abs_diff: 0.1225224
        401 in 1000 raw outputs identical.

        Model name: googlenet, per channel quantization
        PyTorch top3 label: [101 386 385]
        TVM top3 label: [101 386 385]
        max abs diff: 0.34015465
        mean abs_diff: 0.054197952
        558 in 1000 raw outputs identical.
        """


def test_serialized_modules():
    ishape = (1, 16, 64, 64)
    raw_module = AdaptiveAvgPool2d().eval()
    inp = torch.rand(ishape)

    quantize_model(raw_module, inp)
    script_module = torch.jit.trace(raw_module, inp).eval()

    fname = "tmp.pt"
    torch.jit.save(script_module, fname)
    loaded = torch.jit.load(fname)
    os.remove(fname)

    with torch.no_grad():
        pt_result = loaded(inp.clone()).numpy()

    input_name = "input"
    runtime = get_tvm_runtime(loaded, input_name, ishape)
    runtime.set_input(input_name, inp.numpy().copy())
    runtime.run()
    tvm_result = runtime.get_output(0).numpy()

    # with 0.5ish results, 1e-2 is relative accuracy close to 2**-6.
    # for simple layers like here this should be achievable
    # with 8 bit quantization
    # we only require 90% match just to be sure
    num_identical = np.sum(np.abs(tvm_result - pt_result) < 1e-2)
    match_ratio = num_identical / float(np.prod(tvm_result.shape))
    assert match_ratio > 0.90


def test_quantize_dynamic():
    # A wrapper is required for quantize_dynamic to work correctly
    class LinearWrapper(nn.Module):
        def __init__(self, in_dim, hidden_dim):
            super().__init__()
            self.linear = nn.Linear(in_dim, hidden_dim)

        def forward(self, inp):
            return self.linear(inp)

    torch.manual_seed(0)
    mod = LinearWrapper(16, 32)

    for qconfig in [
        torch.quantization.per_channel_dynamic_qconfig,
        torch.quantization.default_dynamic_qconfig,
    ]:
        for ishape in [(16, 16), (10, 16, 16)]:
            qspec = {nn.Linear: qconfig}
            qmod = torch.quantization.quantize_dynamic(mod, qconfig_spec=qspec, dtype=torch.qint8)

            inp = torch.randn(*ishape)
            script_module = torch.jit.trace(qmod, inp).eval()

            with torch.no_grad():
                pt_result = script_module(inp.clone()).numpy()

            input_name = "input"
            runtime = get_tvm_runtime(script_module, "input", inp.shape)
            runtime.set_input(input_name, inp.numpy().copy())
            runtime.run()
            tvm_result = runtime.get_output(0).numpy()

            # Only compare with the PyTorch result for version v1.6 or newer
            # Have seen a strange accuracy problem from PyTorch 1.4 and 1.5
            # Even with the manual random seed set, the same PyTorch
            # version can outputs slightly different results depending on an environment.
            # Outputs from v1.6 seem reliable. TVM's outputs are always the same
            if is_version_greater_than("1.5.1"):
                tvm.testing.assert_allclose(tvm_result, pt_result, rtol=1e-4, atol=1e-4)


def make_qnn_add_pattern():
    from tvm.relay.dataflow_pattern import wildcard, is_op

    lhs = wildcard()
    rhs = wildcard()
    lhs_scale = wildcard()
    lhs_zero_point = wildcard()
    rhs_scale = wildcard()
    rhs_zero_point = wildcard()
    output_scale = wildcard()
    output_zero_point = wildcard()
    qadd = is_op("qnn.add")(
        lhs,
        rhs,
        lhs_scale,
        lhs_zero_point,
        rhs_scale,
        rhs_zero_point,
        output_scale,
        output_zero_point,
    )
    return qadd.optional(is_op("clip"))


@register_pattern_table("test_table")
def pattern_table():
    return [
        ("qnn_add", make_qnn_add_pattern()),
    ]


def run_qnn_mergecomposite(script_module, input_name, ishape):
    input_shapes = [(input_name, ishape)]
    mod, params = relay.frontend.from_pytorch(script_module, input_shapes)
    pattern_table = get_pattern_table("test_table")
    with tvm.transform.PassContext(opt_level=3):
        pass_list = [
            tvm.relay.transform.SimplifyInference(),
            tvm.relay.transform.MergeComposite(pattern_table),
        ]
        composite_partition = tvm.transform.Sequential(pass_list)
        partitioned = composite_partition(mod)


def test_qnn_mergecomposite():
    from torchvision.models.quantization import resnet as qresnet

    model = qresnet.resnet18(pretrained=True)
    model.eval()

    inp = torch.zeros((1, 3, 224, 224))
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
    torch.quantization.prepare(model, inplace=True)
    model(inp)
    torch.quantization.convert(model, inplace=True)
    script_module = torch.jit.trace(model, inp).eval()

    input_name = "image"
    run_qnn_mergecomposite(script_module, input_name, inp.shape)
