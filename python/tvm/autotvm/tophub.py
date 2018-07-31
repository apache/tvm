"""
TopHub: Tensor Operator Hub
To get the best performance, we typically need auto-tuning for the specific devices.
TVM releases pre-tuned parameters in TopHub for some common networks and hardware targets.
TVM will download these parameters for you when you create the target for the first time.
"""

import logging
import os
import json

from .task import DispatchContext
from .record import ApplyHistoryBest
from ..contrib.util import tempdir
from ..contrib.download import download

AUTOTVM_TOPHUB_ROOT_PATH = os.path.join(os.path.expanduser('~'), ".tvm", "tophub")


class EmptyContext(object):
    """An empty context"""
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def context(target, extra_files=None):
    """Return the dispatch context with pre-tuned parameters.
    The corresponding downloaded *.log files under tophub root path will be loaded.
    Users can also add their own files in argument `extra_files`.

    Parameters
    ----------
    target: Target
        The compilation target
    extra_files: list of str, optional
        Extra log files to load
    """
    rootpath = AUTOTVM_TOPHUB_ROOT_PATH
    best_context = ApplyHistoryBest([])

    big_target = str(target).split()[0]
    if os.path.isfile(os.path.join(rootpath, big_target + ".log")):
        best_context.load(os.path.join(rootpath, big_target + ".log"))

    for opt in target.options:
        if opt.startswith("-device"):
            model = opt[8:]
            if os.path.isfile(os.path.join(rootpath, model) + ".log"):
                best_context.load(os.path.join(rootpath, model) + ".log")

    if extra_files:
        for filename in extra_files:
            best_context.load(filename)

    return best_context


def load_context(target, extra_files=None):
    """Load the dispatch context with pre-tuned parameters.
    The corresponding downloaded *.log files under tophub root path will be loaded.
    Users can also add their own files in argument `extra_files`.

    Parameters
    ----------
    target: Target
        The compilation target
    extra_files: list of str, optional
        Extra log files to load
    """
    best_context = context(target, extra_files)
    assert not DispatchContext.current, "Cannot load pre-tuned parameters inside a dispatch context"
    best_context.__enter__()


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

    logging.info("Download pre-tuned parameters for %s", backend)
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
    if os.path.isfile(os.path.join(AUTOTVM_TOPHUB_ROOT_PATH, backend + ".log")):
        return

    download_package(backend)


def list_packages():
    """List all available pre-tuned op parameters for targets

    Returns
    -------
    ret: List
        All available packets
    """
    path = tempdir()
    filename = path.relpath("info.json")
    logging.info("Download meta info for pre-tuned parameters")
    download("https://raw.githubusercontent.com/uwsaml/tvm-distro/master/tophub/info.json",
             filename, True, verbose=0)

    with open(filename, "r") as fin:
        text = "".join(fin.readlines())
    info = json.loads(text)
    keys = list(info.keys())
    keys.sort()

    return [(k, info[k]) for k in keys]
