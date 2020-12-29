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
"""VTA specific bitstream management library."""
from __future__ import absolute_import as _abs

import os
import sys

from tvm.contrib.download import download
from .environment import get_env

if sys.version_info >= (3,):
    import urllib.error as urllib2
else:
    import urllib2

# bitstream repo
BITSTREAM_URL = "https://github.com/uwsampl/vta-distro/raw/master/bitstreams/"


def get_bitstream_path():
    """Returns the path to the cached bitstream corresponding to the current config

    Returns
    -------
    bit_path: str
        Corresponding to the filepath of the bitstream
    """

    env = get_env()

    # Derive destination path
    cache_dir = os.getenv("VTA_CACHE_PATH", os.path.join(os.getenv("HOME"), ".vta_cache/"))
    cache_dir = os.path.join(cache_dir, env.TARGET)
    cache_dir = os.path.join(cache_dir, env.HW_VER.replace(".", "_"))
    # Create the directory if it didn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    bit_path = os.path.join(cache_dir, env.BITSTREAM) + ".bit"

    return bit_path


def download_bitstream():
    """Downloads a cached bitstream corresponding to the current config"""

    env = get_env()

    success = False
    bit = get_bitstream_path()
    url = os.path.join(BITSTREAM_URL, env.TARGET)
    url = os.path.join(url, env.HW_VER)
    url = os.path.join(url, env.BITSTREAM + ".bit")

    try:
        download(url, bit)
    except urllib2.HTTPError as err:
        if err.code == 404:
            raise RuntimeError(
                # Raise error - the solution when this happens it to build your
                # own bitstream and add it to your $VTA_CACHE_PATH
                "{} is not available. It appears that this configuration \
bistream has not been cached. Please compile your own bitstream (see hardware \
compilation guide to get Xilinx toolchains setup) and add it to your \
$VTA_CACHE_PATH. Alternatively edit your config.json back to its default \
settings. You can see the list of available bitstreams under {}".format(
                    url, BITSTREAM_URL
                )
            )
        raise RuntimeError(
            # This could happen when trying to access the URL behind a proxy
            "Something went wrong when trying to access {}. Check your \
internet connection or proxy settings.".format(
                url
            )
        )

    return success
