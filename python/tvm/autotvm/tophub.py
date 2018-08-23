"""
TopHub: Tensor Operator Hub
To get the best performance, we typically need auto-tuning for the specific devices.
TVM releases pre-tuned parameters in TopHub for some common networks and hardware targets.
TVM will download these parameters for you when you create the target for the first time.
"""
# pylint: disable=invalid-name

import logging
import os
import sys

from .task import ApplyHistoryBest
from .. import target as _target
from ..contrib.download import download

# root path to store TopHub files
AUTOTVM_TOPHUB_ROOT_PATH = os.path.join(os.path.expanduser('~'), ".tvm", "tophub")

# the version of each package
PACKAGE_VERSION = {
    'vta':     "v0.01",
    'arm_cpu': "v0.01",
    'cuda':    "v0.01",
}

logger = logging.getLogger('autotvm')

def _alias(name):
    """convert alias for some packages"""
    table = {
        'vtacpu': 'vta',
    }
    return table.get(name, name)


def context(target, extra_files=None):
    """Return the dispatch context with pre-tuned parameters.
    This function will load the corresponding *.log files in AUTOTVM_TOPHUB_ROOT_PATH.
    If cannot find them, it will download them from TopHub github repo.
    Users can also add their own files in argument `extra_files`.

    Parameters
    ----------
    target: Target
        The compilation target
    extra_files: list of str, optional
        Extra log files to load
    """
    best_context = ApplyHistoryBest([])

    if isinstance(target, str):
        target = _target.create(target)

    possible_names = [str(target).split()[0]]
    for opt in target.options:
        if opt.startswith("-device"):
            device = _alias(opt[8:])
            possible_names.append(device)

    all_packages = list(PACKAGE_VERSION.keys())
    for name in possible_names:
        if name in all_packages:
            check_backend(name)

            filename = "%s_%s.log" % (name, PACKAGE_VERSION[name])
            best_context.load(os.path.join(AUTOTVM_TOPHUB_ROOT_PATH, filename))

    if extra_files:
        for filename in extra_files:
            best_context.load(filename)

    return best_context


def check_backend(backend):
    """Check whether have pre-tuned parameters of the certain target.
    If not, will download it.

    Parameters
    ----------
    backend: str
        The name of backend.
    """
    backend = _alias(backend)
    assert backend in PACKAGE_VERSION, 'Cannot find backend "%s" in TopHub' % backend

    version = PACKAGE_VERSION[backend]
    package_name = "%s_%s.log" % (backend, version)
    if os.path.isfile(os.path.join(AUTOTVM_TOPHUB_ROOT_PATH, package_name)):
        return

    if sys.version_info >= (3,):
        import urllib.request as urllib2
    else:
        import urllib2
    try:
        download_package(package_name)
    except urllib2.URLError as e:
        logging.warning("Failed to download tophub package for %s: %s", backend, e)


def download_package(package_name):
    """Download pre-tuned parameters of operators for a backend

    Parameters
    ----------
    package_name: str
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

    logger.info("Download pre-tuned parameters package %s", package_name)
    download("https://raw.githubusercontent.com/uwsaml/tvm-distro/master/tophub/%s"
             % package_name, os.path.join(rootpath, package_name), True, verbose=0)
