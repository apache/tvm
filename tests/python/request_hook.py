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

import logging
import urllib.request

LOGGER = None


URL_MAP = {}


class TvmRequestHook(urllib.request.Request):
    def __init__(self, url, *args, **kwargs):
        LOGGER.info(f"Caught access to {url}")
        url = url.strip()
        if url not in URL_MAP:
            # Disallow network accesses without an explicitly maintained mirror.
            msg = (
                f"Uncaught URL found in CI: {url}. "
                "Avoid network access or arrange a stable project-managed mirror, "
                "then add it to URL_MAP in tests/python/request_hook.py."
            )
            raise RuntimeError(msg)

        new_url = URL_MAP[url]
        LOGGER.info(f"Mapped URL {url} to {new_url}")
        super().__init__(new_url, *args, **kwargs)


def init():
    global LOGGER
    if urllib.request.Request is TvmRequestHook:
        return
    urllib.request.Request = TvmRequestHook
    LOGGER = logging.getLogger("tvm_request_hook")
    LOGGER.setLevel(logging.DEBUG)
    fh = logging.FileHandler("redirected_urls.log")
    fh.setLevel(logging.DEBUG)
    LOGGER.addHandler(fh)
