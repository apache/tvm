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
"""Workloads in Relay IR"""
from enum import Enum
from typing import Dict, Tuple

import tvm.relay.testing  # pylint: disable=unused-import
from tvm import relay
from tvm.ir import IRModule
from tvm.runtime import NDArray

# Model types supported in Torchvision
class MODEL_TYPE(Enum):  # pylint: disable=invalid-name
    IMAGE_CLASSIFICATION = (1,)
    VIDEO_CLASSIFICATION = (2,)
    SEGMENTATION = (3,)
    OBJECT_DETECTION = (4,)
    TEXT_CLASSIFICATION = (5,)


# Specify the type of each model
MODEL_TYPES = {
    "resnet18": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "mobilenet_v2": MODEL_TYPE.IMAGE_CLASSIFICATION,
    "bert_base": MODEL_TYPE.TEXT_CLASSIFICATION,
}


def get_torch_model(
    model_name: str,
    input_shape: Tuple[int, ...],
    output_shape: Tuple[int, int],  # pylint: disable=unused-argument
    dtype: str = "float32",
) -> Tuple[IRModule, Dict[str, NDArray]]:
    """Load model from torch model zoo
    Parameters
    ----------
    model_name : str
        The name of the model to load
    input_shape: Tuple[int, ...]
        Tuple for input shape
    output_shape: Tuple[int, int]
        Tuple for output shape
    dtype: str
        Tensor data type
    """

    assert dtype == "float32"

    import torch  # type: ignore # pylint: disable=import-error,import-outside-toplevel
    from torchvision import models  # type: ignore # pylint: disable=import-error,import-outside-toplevel
    import transformers  # type: ignore # pylint: disable=import-error,import-outside-toplevel
    import os  # type: ignore # pylint: disable=import-error,import-outside-toplevel

    def do_trace(model, inp):
        model.eval()
        model_trace = torch.jit.trace(model, inp)
        model_trace.eval()
        return model_trace

    # Load model from torchvision
    if MODEL_TYPES[model_name] == MODEL_TYPE.TEXT_CLASSIFICATION:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model = transformers.BertModel(
            transformers.BertConfig(
                num_hidden_layers=12,
                hidden_size=768,
                intermediate_size=3072,
                num_attention_heads=12,
                return_dict=False,
            )
        )
        model.eval()
        input_data = torch.randint(10000, input_shape)
        shape_list = [("input_ids", input_shape)]
        scripted_model = torch.jit.trace(model, [input_data], strict=False)
    elif MODEL_TYPES[model_name] == MODEL_TYPE.IMAGE_CLASSIFICATION:
        model = getattr(models, model_name)()
        # Setup input
        input_data = torch.randn(input_shape).type(torch.float32)
        shape_list = [("input0", input_shape)]
        # Get trace. Depending on the model type, wrapper may be necessary.
        scripted_model = do_trace(model, input_data)
    else:
        raise ValueError("Unsupported model in Torch model zoo.")

    # Convert torch model to relay module
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    return mod, params


def get_network(
    name: str,
    batch_size: int,
    layout: str = "NHWC",
    dtype: str = "float32",
) -> Tuple[IRModule, Dict[str, NDArray], Tuple[int, int, int, int], Tuple[int, int]]:
    """Get the symbol definition and random weight of a network"""
    # meta-schedule prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape: Tuple[int, int, int, int] = (batch_size,) + image_shape
    output_shape: Tuple[int, int] = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        from mxnet.gluon.model_zoo.vision import get_model  # type: ignore  # pylint: disable=import-outside-toplevel

        assert layout == "NCHW"
        block = get_model("resnet50_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = IRModule.from_expr(net)
    return mod, params, input_shape, output_shape
