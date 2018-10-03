"""
TopHub: Tensor Operator Hub
To get the best performance, we typically need auto-tuning for the specific devices.
TVM releases pre-tuned parameters in TopHub for some common networks and hardware targets.
TVM will download these parameters for you when you call nnvm.compiler.build_module .
"""
# pylint: disable=invalid-name

import logging
import os
import sys

from .task import ApplyHistoryBest
from .. import target as _target
from ..contrib.download import download
from .record import load_from_file

# root path to store TopHub files
AUTOTVM_TOPHUB_ROOT_PATH = os.path.join(os.path.expanduser('~'), ".tvm", "tophub")

# the version of each package
PACKAGE_VERSION = {
    'arm_cpu': "v0.03",

    'cuda':    "v0.02",
    'rocm':    "v0.01",
    'opencl':  "v0.01",
    'mali':    "v0.02",

    'vta':     "v0.01",
}

logger = logging.getLogger('autotvm')

def _alias(name):
    """convert alias for some packages"""
    table = {
        'vtacpu': 'vta',

        'metal': 'opencl',
        'nvptx': 'cuda',
    }
    return table.get(name, name)


def context(target, extra_files=None):
    """Return the dispatch context with pre-tuned parameters.
    This function will load the corresponding *.log files in AUTOTVM_TOPHUB_ROOT_PATH.
    If cannot find them, it will download them from TopHub github repo.
    Users can also add their own files in argument `extra_files`.

    Parameters
    ----------
    target: Target or List of Target
        The compilation target
    extra_files: list of str, optional
        Extra log files to load
    """
    best_context = ApplyHistoryBest([])

    targets = target if isinstance(target, (list, tuple)) else [target]

    for tgt in targets:
        if isinstance(tgt, str):
            tgt = _target.create(tgt)

        possible_names = []
        for opt in tgt.options:
            if opt.startswith("-device"):
                device = _alias(opt[8:])
                possible_names.append(device)
        possible_names.append(tgt.target_name)

        all_packages = list(PACKAGE_VERSION.keys())
        for name in possible_names:
            name = _alias(name)
            if name in all_packages:
                check_backend(name)

                filename = "%s_%s.log" % (name, PACKAGE_VERSION[name])
                best_context.load(os.path.join(AUTOTVM_TOPHUB_ROOT_PATH, filename))
                break   # only load one file to avoid some fallback template mismatch problem

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


# global cache for load_reference_log
REFERENCE_LOG_CACHE = {}

def load_reference_log(backend, model, workload_name, template_key):
    """ Load reference log from TopHub to support fallback in template.
    Template will use these reference logs to choose fallback config.

    Parameters
    ----------
    backend: str
        The backend name
    model: str
        The name of the model
    workload_name: str
        The name of the workload. (The first item in the workload tuple)
    template_key: str
        The template key
    """

    backend = _alias(backend)
    version = PACKAGE_VERSION[backend]
    package_name = "%s_%s.log" % (backend, version)
    filename = os.path.join(AUTOTVM_TOPHUB_ROOT_PATH, package_name)

    global REFERENCE_LOG_CACHE
    key = (backend, model, workload_name, template_key)

    if key not in REFERENCE_LOG_CACHE:
        tmp = []
        if os.path.isfile(os.path.join(AUTOTVM_TOPHUB_ROOT_PATH, package_name)):
            find = False
            inp = None
            for inp, res in load_from_file(filename):
                if model == inp.target.model:
                    find = True
                    break
            if not find and inp:
                model = inp.target.model

            for inp, res in load_from_file(filename):
                if (model == inp.target.model and inp.task.workload[0] == workload_name and
                        inp.config.template_key == template_key):
                    tmp.append((inp, res))
        REFERENCE_LOG_CACHE[key] = tmp

    return REFERENCE_LOG_CACHE[key]
