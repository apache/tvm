#!/usr/bin/env bash

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
# ruff: noqa: E501

import logging
import urllib.request

LOGGER = None


# To update this list, run https://github.com/apache/tvm/actions/workflows/upload_ci_resource.yml
# with the URL to download and the SHA-256 hash of the file.
BASE = "https://tvm-ci-resources.s3.us-west-2.amazonaws.com"
URL_MAP = {
    "https://github.com/onnx/models/raw/131c99da401c757207a40189385410e238ed0934/vision/classification/mobilenet/model/mobilenetv2-7.onnx": f"{BASE}/onnx/models/raw/131c99da401c757207a40189385410e238ed0934/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
}


class TvmRequestHook(urllib.request.Request):
    def __init__(self, url, *args, **kwargs):
        LOGGER.info(f"Caught access to {url}")
        url = url.strip()
        if url not in URL_MAP and not url.startswith(BASE):
            # Dis-allow any accesses that aren't going through S3
            msg = (
                f"Uncaught URL found in CI: {url}. "
                "A committer must upload the relevant file to S3 via "
                "https://github.com/apache/tvm/actions/workflows/upload_ci_resource.yml "
                "and add it to the mapping in tests/scripts/request_hook/request_hook.py"
            )
            raise RuntimeError(msg)

        new_url = URL_MAP[url]
        LOGGER.info(f"Mapped URL {url} to {new_url}")
        super().__init__(new_url, *args, **kwargs)


def init():
    global LOGGER
    urllib.request.Request = TvmRequestHook
    LOGGER = logging.getLogger("tvm_request_hook")
    LOGGER.setLevel(logging.DEBUG)
    fh = logging.FileHandler("redirected_urls.log")
    fh.setLevel(logging.DEBUG)
    LOGGER.addHandler(fh)
