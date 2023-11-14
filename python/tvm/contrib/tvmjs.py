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
import hashlib
import json
import os
import shutil

# pylint: disable=unused-import
import sys
from typing import Mapping, Union

import numpy as np

import tvm
from tvm._ffi.libinfo import find_lib_path

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


def _calculate_md5(filename):
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class NDArrayCacheShardingManager:
    """Internal helper to shard ndarrays."""

    def __init__(self, cache_dir: str, prefix: str, shard_cap_nbytes: int):
        self.cache_dir = cache_dir
        self.prefix = prefix
        self.curr_records = []
        self.curr_data = bytearray()
        self.shard_records = []
        self.shard_cap_nbytes = shard_cap_nbytes
        self.counter = 0

    def append(self, data, name, shape, dtype, encode_format):
        """Commit a record to the manager.

        Parameters
        ----------
        data: bytes
            Raw bytes to be appended.

        name: str
            The name of the parameter

        shape: tuple
            The shape of the array

        dtype: str
            The dtype information

        encode_format:
            The encode format of the entry
        """
        rec = {
            "name": name,
            "shape": shape,
            "dtype": dtype,
            "format": encode_format,
            "nbytes": len(data),
        }

        if self.pending_nbytes + len(data) >= self.shard_cap_nbytes:
            if len(data) * 2 >= self.shard_cap_nbytes:
                # out of band data
                rec["byteOffset"] = 0
                self._commit_internal(data, [rec])
                return
            self.commit()
        rec["byteOffset"] = self.pending_nbytes
        self.curr_records.append(rec)
        self.curr_data += data

    def commit(self):
        """Commit a record"""
        if self.pending_nbytes != 0:
            self._commit_internal(self.curr_data, self.curr_records)
            self.curr_data = bytearray()
            self.curr_records = []

    def finish(self):
        """Finish building and return shard records."""
        self.commit()
        return self.shard_records

    def _commit_internal(self, data, records):
        data_path = f"{self.prefix}_{self.counter}.bin"
        full_path = os.path.join(self.cache_dir, data_path)
        self.counter += 1
        with open(full_path, "wb") as outfile:
            outfile.write(data)

        shard_record = {
            "dataPath": data_path,
            "format": "raw-shard",
            "nbytes": len(data),
            "records": records,
            "md5sum": _calculate_md5(full_path),
        }
        self.shard_records.append(shard_record)

    @property
    def pending_nbytes(self):
        """Return total bytes stored so far"""
        return len(self.curr_data)


def dump_ndarray_cache(
    params: Mapping[str, Union[np.ndarray, tvm.runtime.NDArray]],
    cache_dir: str,
    encode_format="f32-to-bf16",
    meta_data=None,
    shard_cap_mb=32,
):
    """Dump parameters to NDArray cache.

    Parameters
    ----------
    params: Mapping[str, tvm.runtime.NDArray],
        The parameter dictionary

    cache_dir: str
        The path to the cache

    encode_format: {"f32-to-bf16", "raw"}
        Encoding format.

    meta_data: json-compatible-struct
        Extra meta_data to be stored in the cache json file.

    shard_cap_mb: int
        Maxinum number of MB to be kept per shard
    """
    if encode_format not in ("raw", "f32-to-bf16"):
        raise ValueError(f"Invalie encode_format {encode_format}")

    meta_data = {} if meta_data is None else meta_data
    records = []
    total = len(params)
    counter = 0
    max_out_length = 0

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    f32_to_bf16_triggered = False

    print("Start storing to cache %s" % cache_dir)
    shard_cap_nbytes = shard_cap_mb * (1 << 20)

    shard_manager = NDArrayCacheShardingManager(cache_dir, "params_shard", shard_cap_nbytes)

    for k, origin_v in params.items():
        shape = list(origin_v.shape)
        v = origin_v
        if not isinstance(v, np.ndarray):
            v = v.numpy()

        # prefer to preserve original dtype, especially if the format was bfloat16
        dtype = str(origin_v.dtype) if isinstance(origin_v, tvm.nd.NDArray) else str(v.dtype)

        # convert fp32 to bf16
        if encode_format == "f32-to-bf16" and dtype == "float32":
            data = _convert_f32_to_bf16(v).tobytes()
            f32_to_bf16_triggered = True
        else:
            data = v.tobytes()

        shard_manager.append(data, name=k, shape=shape, dtype=dtype, encode_format=encode_format)

        counter += 1
        last_cmd = "[%04d/%04d] saving %s" % (counter, total, k)
        flush = "\r" + (" " * max_out_length) + "\r"
        max_out_length = max(len(last_cmd), max_out_length)
        sys.stdout.write(flush + last_cmd)

    records = shard_manager.finish()

    nd_cache_json = os.path.join(cache_dir, "ndarray-cache.json")

    with open(nd_cache_json, "w") as outfile:
        json.dump({"metadata": meta_data, "records": records}, outfile, indent=4)
    print(
        "\nAll finished, %d total shards committed, record saved to %s"
        % (shard_manager.counter, nd_cache_json)
    )

    if f32_to_bf16_triggered:
        for shard in records:
            for item in shard["records"]:
                if item["dtype"] == "float32":
                    item["format"] = "raw"
                    item["dtype"] = "bfloat16"
        b16_nd_cache_json = os.path.join(cache_dir, "ndarray-cache-b16.json")
        # also dump a file that contains bf16
        with open(b16_nd_cache_json, "w") as outfile:
            json.dump({"metadata": meta_data, "records": records}, outfile, indent=4)
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

    for shard_rec in json_info["records"]:
        data_path = shard_rec["dataPath"]
        full_data_path = os.path.join(cachedir, data_path)
        raw_data = open(full_data_path, "rb").read()
        assert shard_rec["format"] == "raw-shard"
        assert shard_rec["nbytes"] == len(raw_data)

        for rec in shard_rec["records"]:
            name = rec["name"]
            shape = rec["shape"]
            dtype = rec["dtype"]
            encode_format = rec["format"]
            offset = rec["byteOffset"]
            nbytes = rec["nbytes"]

            arr = tvm.nd.empty(shape, dtype, device=device)
            assert offset + nbytes <= len(raw_data)
            buffer_source = raw_data[offset : offset + nbytes]
            if encode_format == "f32-to-bf16" and dtype == "float32":
                data = np.frombuffer(buffer_source, dtype="uint16").reshape(shape)
                arr.copyfrom(_convert_bf16_to_f32(data))
            elif dtype == "bfloat16":
                data = np.frombuffer(buffer_source, dtype="uint16").reshape(shape)
                arr.copyfrom(data)
            else:
                data = np.frombuffer(buffer_source, dtype=dtype).reshape(shape)
                arr.copyfrom(data)
            result_dict[name] = arr
    return result_dict, json_info["metadata"]


def export_runtime(runtime_dir):
    """Export TVMJS runtime to the runtime_dir

    Parameters
    ----------
    runtime_dir: str
        The runtime directory
    """
    web_hint = (
        "make sure you setup tvm web runtime correctly."
        + " obtain a copy of TVM source code, set TVM_HOME env variable:\n"
        + " cd /path/to/tvm/web; make; npm run bundle"
    )

    jsbundle = find_lib_path("tvmjs.bundle.js", optional=True)
    if not jsbundle:
        raise RuntimeError("Cannot find tvmjs.bundle.js, " + web_hint)

    wasi = find_lib_path("tvmjs_runtime.wasi.js", optional=True)
    if not wasi:
        raise RuntimeError("Cannot find tvmjs_runtime.wasi.js, " + web_hint)

    print(f"Copy {jsbundle[0]} to {runtime_dir}")
    shutil.copy(jsbundle[0], runtime_dir)
    print(f"Copy {wasi[0]} to {runtime_dir}")
    shutil.copy(wasi[0], runtime_dir)
