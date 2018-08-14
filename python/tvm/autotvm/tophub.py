"""
TopHub: Tensor Operator Hub
To get the best performance, we typically need auto-tuning for the specific devices.
TVM releases pre-tuned parameters in TopHub for some common networks and hardware targets.
TVM will download these parameters for you when you create the target for the first time.
"""
# pylint: disable=invalid-name

import logging
import os
import json
import sys

from .task import ApplyHistoryBest
from .. import target as _target
from ..contrib.util import tempdir
from ..contrib.download import download

AUTOTVM_TOPHUB_ROOT_PATH = os.path.join(os.path.expanduser('~'), ".tvm", "tophub")

logger = logging.getLogger('autotvm')

def _alias(name):
    """convert alias for some packages"""
    table = {
        'vtacpu': 'vta',
    }
    return table.get(name, name)


def context(target, extra_files=None, allow_fallback=False):
    """Return the dispatch context with pre-tuned parameters.
    The corresponding downloaded *.log files under tophub root path will be loaded.
    Users can also add their own files in argument `extra_files`.

    Parameters
    ----------
    target: Target
        The compilation target
    extra_files: list of str, optional
        Extra log files to load
    allow_fallback: bool
        Whether allow to use a fallback configuration if cannot find
        tuned result.
    """
    rootpath = AUTOTVM_TOPHUB_ROOT_PATH
    best_context = ApplyHistoryBest([], allow_fallback=allow_fallback)

    if isinstance(target, str):
        target = _target.create(target)

    big_target = str(target).split()[0]
    if os.path.isfile(os.path.join(rootpath, big_target + ".log")):
        best_context.load(os.path.join(rootpath, big_target + ".log"))

    for opt in target.options:
        if opt.startswith("-device"):
            model = _alias(opt[8:])
            if os.path.isfile(os.path.join(rootpath, model) + ".log"):
                best_context.load(os.path.join(rootpath, model) + ".log")

    if extra_files:
        for filename in extra_files:
            best_context.load(filename)

    return best_context


def download_package(backend):
    """Download pre-tuned parameters of operators for a backend

    Parameters
    ----------
    backend: str
        The name of package
    """
    rootpath = AUTOTVM_TOPHUB_ROOT_PATH

    if not os.path.isdir(rootpath):
        # make directory
        splits = os.path.split(rootpath)
        for j in range(1, len(splits)+1):
            path = os.path.join(*splits[:j])
            if not os.path.isdir(path):
                os.mkdir(path)

    backend = _alias(backend)
    logger.info("Download pre-tuned parameters for %s", backend)
    download("https://raw.githubusercontent.com/uwsaml/tvm-distro/master/tophub/%s.log" % backend,
             os.path.join(rootpath, backend + ".log"), True, verbose=0)


def check_package(backend):
    """Check whether have pre-tuned parameters of the certain target.
    If not, will download it.

    Parameters
    ----------
    backend: str
        The name of package
    """
    backend = _alias(backend)

    if os.path.isfile(os.path.join(AUTOTVM_TOPHUB_ROOT_PATH, backend + ".log")):
        return

    if sys.version_info >= (3,):
        import urllib.request as urllib2
    else:
        import urllib2
    try:
        download_package(backend)
    except urllib2.URLError:
        logging.warning("Failed to download tophub package for %s", backend)


def list_packages():
    """List all available pre-tuned op parameters for targets

    Returns
    -------
    ret: List
        All available packets
    """
    path = tempdir()
    filename = path.relpath("info.json")
    logger.info("Download meta info for pre-tuned parameters")
    download("https://raw.githubusercontent.com/uwsaml/tvm-distro/master/tophub/info.json",
             filename, True, verbose=0)

    with open(filename, "r") as fin:
        text = "".join(fin.readlines())
    info = json.loads(text)
    keys = list(info.keys())
    keys.sort()

    return [(k, info[k]) for k in keys]
