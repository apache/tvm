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
BITSTREAM_URL = "https://github.com/uwsaml/vta-distro/raw/master/bitstreams/"

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
    # Create the directory if it didn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    bit_path = os.path.join(cache_dir, env.BITSTREAM)

    return bit_path


def download_bitstream():
    """Downloads a cached bitstream corresponding to the current config
    """

    env = get_env()

    success = False
    bit = get_bitstream_path()
    url = os.path.join(BITSTREAM_URL, env.TARGET)
    url = os.path.join(url, env.HW_VER)
    url = os.path.join(url, env.BITSTREAM)

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
settings. You can see the list of available bitstreams under {}"
                .format(url, BITSTREAM_URL))
        else:
            raise RuntimeError(
                # This could happen when trying to access the URL behind a proxy
                "Something went wrong when trying to access {}. Check your \
internet connection or proxy settings."
                .format(url))

    return success
