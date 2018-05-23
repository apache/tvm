"""VTA specific bitstream management library."""
from __future__ import absolute_import as _abs

import os
import urllib
from .environment import get_env

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
    # Check that the bitstream is accessible from the server
    if urllib.urlopen(url).getcode() == 404:
        # Raise error - the solution when this happens it to build your own bitstream and add it
        # to your VTA_CACHE_PATH
        raise RuntimeError(
            "Error: {} is not available. It appears that this configuration has not been built."
            .format(url))
    else:
        urllib.urlretrieve(url, bit)
        success = True

    return success
