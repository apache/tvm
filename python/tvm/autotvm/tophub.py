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
"""
TopHub: Tensor Operator Hub
To get the best performance, we typically need auto-tuning for the specific devices.
TVM releases pre-tuned parameters in TopHub for some common networks and hardware targets.
TVM will download these parameters for you when you call relay.build.
"""
# pylint: disable=invalid-name

import logging
from os import getenv
import sys
from pathlib import Path
from tvm.ir.container import Array

from .task import ApplyHistoryBest
from ..target import Target
from ..contrib.download import download
from .record import load_from_file
from .utils import EmptyContext

# environment variable to read TopHub location
AUTOTVM_TOPHUB_LOC_VAR = "TOPHUB_LOCATION"

# default location of TopHub
AUTOTVM_TOPHUB_DEFAULT_LOC = "https://raw.githubusercontent.com/tlc-pack/tophub/main/tophub"

# value of AUTOTVM_TOPHUB_LOC_VAR to specify to not read from TopHub
AUTOTVM_TOPHUB_NONE_LOC = "NONE"

# root path to store TopHub files
AUTOTVM_TOPHUB_ROOT_PATH = Path(Path("~").expanduser(), ".tvm", "tophub")

# the version of each package
PACKAGE_VERSION = {
    "arm_cpu": "v0.08",
    "llvm": "v0.04",
    "cuda": "v0.10",
    "rocm": "v0.05",
    "opencl": "v0.04",
    "mali": "v0.06",
    "intel_graphics": "v0.02",
    "vta": "v0.10",
    "amd_apu": "v0.01",
}

logger = logging.getLogger("autotvm")


def _alias(name):
    """convert alias for some packages"""
    table = {
        "vtacpu": "vta",
        "webgpu": "opencl",
        "vulkan": "opencl",
        "nvptx": "cuda",
        "amd_apu": "amd_apu",
    }
    return table.get(name, name)


def _get_tophub_location():
    location = getenv(AUTOTVM_TOPHUB_LOC_VAR, None)
    return AUTOTVM_TOPHUB_DEFAULT_LOC if location is None else location


def context(target, extra_files=None):
    """Return the dispatch context with pre-tuned parameters.
    This function will load the corresponding *.log files in AUTOTVM_TOPHUB_ROOT_PATH.
    If cannot find them, it will download them from TopHub github repo.
    Users can also add their own files in argument `extra_files`.

    Parameters
    ----------
    target: Target or List of Target
        The compilation targets
    extra_files: list of str, optional
        Extra log files to load
    """
    tophub_location = _get_tophub_location()
    if tophub_location == AUTOTVM_TOPHUB_NONE_LOC:
        return EmptyContext()

    best_context = ApplyHistoryBest([])

    targets = target if isinstance(target, (Array, list, tuple)) else [target]

    for tgt in targets:
        if isinstance(tgt, str):
            tgt = Target(tgt)

        possible_names = []
        device = tgt.attrs.get("device", "")
        if device != "":
            possible_names.append(_alias(device))
        possible_names.append(tgt.kind.name)

        all_packages = list(PACKAGE_VERSION.keys())
        for name in possible_names:
            name = _alias(name)
            if name in all_packages:
                if not check_backend(tophub_location, name):
                    continue

                filename = "%s_%s.log" % (name, PACKAGE_VERSION[name])
                best_context.load(Path(AUTOTVM_TOPHUB_ROOT_PATH, filename))
                break  # only load one file to avoid some fallback template mismatch problem

    if extra_files:
        for filename in extra_files:
            best_context.load(filename)

    return best_context


def check_backend(tophub_location, backend):
    """Check whether have pre-tuned parameters of the certain target.
    If not, will download it.

    Parameters
    ----------
    backend: str
        The name of backend.

    Returns
    ----------
    success: bool
        Whether the check is successful.
    """
    backend = _alias(backend)
    assert backend in PACKAGE_VERSION, 'Cannot find backend "%s" in TopHub' % backend

    version = PACKAGE_VERSION[backend]
    package_name = "%s_%s.log" % (backend, version)
    if Path(AUTOTVM_TOPHUB_ROOT_PATH, package_name).is_file():
        return True

    # pylint: disable=import-outside-toplevel
    if sys.version_info >= (3,):
        import urllib.request as urllib2
    else:
        import urllib2
    try:
        download_package(tophub_location, package_name)
        return True
    except urllib2.URLError as e:
        logging.warning("Failed to download tophub package for %s: %s", backend, e)
        return False


def download_package(tophub_location, package_name):
    """Download pre-tuned parameters of operators for a backend

    Parameters
    ----------
    tophub_location: str
        The location to download TopHub parameters from

    package_name: str
        The name of package
    """
    rootpath = Path(AUTOTVM_TOPHUB_ROOT_PATH)
    rootpath.mkdir(parents=True, exist_ok=True)

    download_url = "{0}/{1}".format(tophub_location, package_name)
    logger.info("Download pre-tuned parameters package from %s", download_url)
    download(download_url, Path(rootpath, package_name), overwrite=True)


# global cache for load_reference_log
REFERENCE_LOG_CACHE = {}


def load_reference_log(backend, model, workload_name):
    """Load reference log from TopHub to support fallback in template.
    Template will use these reference logs to choose fallback config.

    Parameters
    ----------
    backend: str
        The backend name
    model: str
        The name of the device model
    workload_name: str
        The name of the workload. (The first item in the workload tuple)
    """

    backend = _alias(backend)
    if backend not in PACKAGE_VERSION:
        return []
    version = PACKAGE_VERSION[backend]
    package_name = "%s_%s.log" % (backend, version)
    filename = Path(AUTOTVM_TOPHUB_ROOT_PATH, package_name)

    global REFERENCE_LOG_CACHE
    key = (backend, model, workload_name)

    if key not in REFERENCE_LOG_CACHE:
        tmp = []
        # If TOPHUB_LOCATION is not AUTOTVM_TOPHUB_NONE_LOC,
        # Download the config file from tophub if not exists.
        if not Path(filename).exists():
            tophub_location = _get_tophub_location()
            if tophub_location != AUTOTVM_TOPHUB_NONE_LOC:
                download_package(tophub_location, package_name)
        if Path(filename).is_file():  # in case download failed
            find = False
            inp = None
            counts = {}
            for inp, res in load_from_file(filename):
                counts[inp.target.model] = counts.get(inp.target.model, 0) + 1
                if model == inp.target.model:
                    find = True
                    break
            # if device model is not find, use the device model with the most tuned workloads
            if not find and counts:
                model = max(counts.items(), key=lambda k: k[1])[0]

            for inp, res in load_from_file(filename):
                if model == inp.target.model and inp.task.workload[0] == workload_name:
                    tmp.append((inp, res))
        REFERENCE_LOG_CACHE[key] = tmp

    return REFERENCE_LOG_CACHE[key]
