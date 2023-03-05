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
"""Creates a simple TVM modules."""

import argparse
import os
import pathlib

from tvm import relay
import tvm
from tvm import runtime as tvm_runtime
import logging
from tvm.relay.backend import Runtime
from tvm.contrib import cc as _cc

RUNTIMES = [
    (Runtime("crt", {"system-lib": True}), "{name}_c.{ext}"),
    (Runtime("cpp", {"system-lib": True}), "{name}_cpp.{ext}"),
]


def build_module(opts):
    dshape = (1, 3, 224, 224)
    from mxnet.gluon.model_zoo.vision import get_model

    block = get_model("mobilenet0.25", pretrained=True)
    shape_dict = {"data": dshape}
    mod, params = relay.frontend.from_mxnet(block, shape_dict)
    func = mod["main"]
    func = relay.Function(
        func.params, relay.nn.softmax(func.body), None, func.type_params, func.attrs
    )

    for runtime, file_format_str in RUNTIMES:
        with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
            graph, lib, params = relay.build(func, "llvm", runtime=runtime, params=params)

        build_dir = os.path.abspath(opts.out_dir)
        if not os.path.isdir(build_dir):
            os.makedirs(build_dir)
        ext = "tar" if str(runtime) == "crt" else "o"
        lib_file_name = os.path.join(build_dir, file_format_str.format(name="model", ext=ext))
        if str(runtime) == "crt":
            lib.export_library(lib_file_name)
        else:
            # NOTE: at present, export_libarary will always create _another_ shared object, and you
            # can't stably combine two shared objects together (in this case, init_array is not
            # populated correctly when you do that). So for now, must continue to use save() with the
            # C++ library.
            # TODO(areusch): Obliterate runtime.cc and replace with libtvm_runtime.so.
            lib.save(lib_file_name)
        with open(
            os.path.join(build_dir, file_format_str.format(name="graph", ext="json")), "w"
        ) as f_graph_json:
            f_graph_json.write(graph)
        with open(
            os.path.join(build_dir, file_format_str.format(name="params", ext="bin")), "wb"
        ) as f_params:
            f_params.write(tvm_runtime.save_param_dict(params))


def build_test_module(opts):
    import numpy as np

    x = relay.var("x", shape=(10, 5))
    y = relay.var("y", shape=(1, 5))
    z = relay.add(x, y)
    func = relay.Function([x, y], z)
    x_data = np.random.rand(10, 5).astype("float32")
    y_data = np.random.rand(1, 5).astype("float32")
    params = {"y": y_data}

    for runtime, file_format_str in RUNTIMES:
        with tvm.transform.PassContext(opt_level=3, config={"tir.disable_vectorize": True}):
            graph, lib, lowered_params = relay.build(
                tvm.IRModule.from_expr(func),
                "llvm",
                runtime=runtime,
                params=params,
            )

        build_dir = os.path.abspath(opts.out_dir)
        if not os.path.isdir(build_dir):
            os.makedirs(build_dir)
        ext = "tar" if str(runtime) == "crt" else "o"
        lib_file_name = os.path.join(build_dir, file_format_str.format(name="test_model", ext=ext))
        if str(runtime) == "crt":
            lib.export_library(lib_file_name)
        else:
            # NOTE: at present, export_libarary will always create _another_ shared object, and you
            # can't stably combine two shared objects together (in this case, init_array is not
            # populated correctly when you do that). So for now, must continue to use save() with the
            # C++ library.
            # TODO(areusch): Obliterate runtime.cc and replace with libtvm_runtime.so.
            lib.save(lib_file_name)
        with open(
            os.path.join(build_dir, file_format_str.format(name="test_graph", ext="json")), "w"
        ) as f_graph_json:
            f_graph_json.write(graph)
        with open(
            os.path.join(build_dir, file_format_str.format(name="test_params", ext="bin")), "wb"
        ) as f_params:
            f_params.write(tvm_runtime.save_param_dict(lowered_params))
        with open(
            os.path.join(build_dir, file_format_str.format(name="test_data", ext="bin")), "wb"
        ) as fp:
            fp.write(x_data.astype(np.float32).tobytes())
        x_output = x_data + y_data
        with open(
            os.path.join(build_dir, file_format_str.format(name="test_output", ext="bin")), "wb"
        ) as fp:
            fp.write(x_output.astype(np.float32).tobytes())


def build_inputs(opts):
    from tvm.contrib import download
    from PIL import Image
    import numpy as np

    build_dir = os.path.abspath(opts.out_dir)

    # Download test image
    image_url = "https://homes.cs.washington.edu/~moreau/media/vta/cat.jpg"
    image_fn = os.path.join(build_dir, "cat.png")
    download.download(image_url, image_fn)
    image = Image.open(image_fn).resize((224, 224))

    def transform_image(image):
        image = np.array(image) - np.array([123.0, 117.0, 104.0])
        image /= np.array([58.395, 57.12, 57.375])
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :]
        return image

    x = transform_image(image)
    print("x", x.shape)
    with open(os.path.join(build_dir, "cat.bin"), "wb") as fp:
        fp.write(x.astype(np.float32).tobytes())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out-dir", default=".")
    parser.add_argument("-t", "--test", action="store_true")
    opts = parser.parse_args()

    if opts.test:
        build_test_module(opts)
    else:
        build_module(opts)
        build_inputs(opts)
