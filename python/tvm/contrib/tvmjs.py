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
"""Namespace to store utilities for building web runtime."""
# pylint: disable=unused-import
import sys
import os
import json
from typing import Mapping, Union

import numpy as np

import tvm
from .emcc import create_tvmjs_wasm


def _convert_f32_to_bf16(value):
    cap = np.finfo("float32").max
    assert -np.finfo("float32").max == np.finfo("float32").min
    bf16_limit = ((np.array([cap.view("uint32")]) >> 16) << 16).view("float32")[0]
    # When the value is in [-bf16_limit, bf16_limit], round to nearest even.
    # We can afford to do it in dumping phase to reduce overall rounding error.
    #
    # When the value is out of bound(usually mask values in attention), use truncation
    # so it is equivalent to clip to the limit values
    data = value.view("uint32")
    rounding_bias = np.where(
        np.logical_and(value < bf16_limit, value > -bf16_limit),
        ((data >> 16) & 1) + 0x7FFF,
        np.zeros_like(data),
    )
    return ((data + rounding_bias) >> 16).astype("uint16")


def _convert_bf16_to_f32(value):
    data = value.view("uint16")
    return (data.astype("uint32") << 16).view("float32")


def dump_ndarray_cache(
    params: Mapping[str, Union[np.ndarray, tvm.runtime.NDArray]],
    cachedir: str,
    encode_format="f32-to-bf16",
    meta_data=None,
):
    """Dump parameters to NDArray cache.

    Parameters
    ----------
    params: Mapping[str, tvm.runtime.NDArray],
        The parameter dictionary

    cachedir: str
        The path to the cache

    encode_format: {"f32-to-bf16", "raw"}
        Encoding format.

    meta_data: json-compatible-struct
        Extra meta_data to be stored in the cache json file.
    """
    if encode_format not in ("raw", "f32-to-bf16"):
        raise ValueError(f"Invalie encode_format {encode_format}")

    meta_data = {} if meta_data is None else meta_data
    records = []
    total = len(params)
    counter = 0
    max_out_length = 0

    if not os.path.exists(cachedir):
        os.makedirs(cachedir)

    f32_to_bf16_triggered = False

    print("Start storing to cache %s" % cachedir)
    for k, v in params.items():
        fname = k + ".bin"
        out_path = os.path.join(cachedir, fname)
        shape = list(v.shape)

        if not isinstance(v, np.ndarray):
            v = v.numpy()

        # convert fp32 to bf16
        if encode_format == "f32-to-bf16" and v.dtype == "float32":
            _convert_f32_to_bf16(v).tofile(out_path)
            dtype = "bfloat16"
            f32_to_bf16_triggered = True
        else:
            v.tofile(out_path)

        dtype = str(v.dtype)
        records.append(
            {"name": k, "shape": shape, "dtype": dtype, "dataPath": fname, "format": encode_format}
        )
        counter += 1
        last_cmd = "[%04d/%04d] saving %s" % (counter, total, out_path)
        flush = "\r" + (" " * max_out_length) + "\r"
        max_out_length = max(len(last_cmd), max_out_length)
        sys.stdout.write(flush + last_cmd)

    nd_cache_json = os.path.join(cachedir, "ndarray-cache.json")

    with open(nd_cache_json, "w") as outfile:
        json.dump({"metadata": meta_data, "records": records}, outfile, indent=4)
    print("\nAll finished, record saved to %s" % nd_cache_json)

    if f32_to_bf16_triggered:
        rec_bf16 = []
        for item in records:
            if item["dtype"] == "float32":
                item["format"] = "raw"
                item["dtype"] = "bfloat16"
            rec_bf16.append(item)
        b16_nd_cache_json = os.path.join(cachedir, "ndarray-cache-b16.json")
        # also dump a file that contains bf16
        with open(b16_nd_cache_json, "w") as outfile:
            json.dump({"metadata": meta_data, "records": rec_bf16}, outfile, indent=4)
        print("Also saved a bf16 record to %s" % b16_nd_cache_json)


def load_ndarray_cache(cachepath: str, device: tvm.runtime.Device):
    """Load the ndarray cache from the directory or json.


    Parameters
    ----------
    cachepath: str
        Path to the location or json file.

    device: tvm.runtime.Device
        The device we would like to load the data from.
    """
    if not cachepath.endswith(".json"):
        cachepath = os.path.join(cachepath, "ndarray-cache.json")

    cachedir = os.path.dirname(cachepath)
    json_info = json.loads(open(cachepath, "r").read())
    result_dict = {}

    for rec in json_info["records"]:
        name = rec["name"]
        shape = rec["shape"]
        dtype = rec["dtype"]
        encode_format = rec["format"]
        data_path = rec["dataPath"]

        arr = tvm.nd.empty(shape, dtype, device=device)
        full_data_path = os.path.join(cachedir, data_path)

        if encode_format == "f32-to-bf16":
            data = np.fromfile(full_data_path, dtype="uint16").reshape(shape)
            arr.copyfrom(_convert_bf16_to_f32(data))
        else:
            data = np.fromfile(full_data_path, dtype=dtype).reshape(shape)
            arr.copyfrom(data)
        result_dict[name] = arr
    return result_dict, json_info["metadata"]
