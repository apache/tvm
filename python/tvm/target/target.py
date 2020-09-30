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
"""Target data structure."""
import os
import re
import json
import warnings
import tvm._ffi

from tvm.runtime import Object
from tvm._ffi import register_func as _register_func
from . import _ffi_api


@tvm._ffi.register_object
class TargetKind(Object):
    """Kind of a compilation target"""


@tvm._ffi.register_object
class Target(Object):
    """Target device information, use through TVM API.

    Note
    ----
    You can create target using the constructor or the following functions

    - :py:func:`tvm.target.arm_cpu` create arm_cpu target
    - :py:func:`tvm.target.cuda` create CUDA target
    - :py:func:`tvm.target.rocm` create ROCM target
    - :py:func:`tvm.target.mali` create Mali target
    - :py:func:`tvm.target.intel_graphics` create Intel Graphics target
    """

    def __init__(self, tag_or_str_or_dict):
        """Construct a TVM target object from
        1) Raw target string
        2) Target config dict
        3) Target tag

        Parameters
        ----------
        tag_or_str_or_dict : Union[str, Dict[str, Any]]
            Can be one of a literal target string, a json string describing
            a configuration, or a dictionary of configuration options.
            When using a dictionary or json string to configure target, the
            possible values are:

            kind :  str (required)
                Which codegen path to use, for example 'llvm' or 'cuda'.
            keys : List of str (optional)
                A set of strategies that can be dispatched to. When using
                "kind=opencl" for example, one could set keys to ["mali", "opencl", "gpu"].
            device : str (optional)
                A single key that corresponds to the actual device being run on.
                This will be effectively appended to the keys.
            libs : List of str (optional)
                The set of external libraries to use. For example ['cblas', 'mkl'].
            system-lib : bool (optional)
                If True, build a module that contains self registered functions.
                Useful for environments where dynamic loading like dlopen is banned.
            mcpu : str (optional)
                The specific cpu being run on. Serves only as an annotation.
            model : str (optional)
                An annotation indicating what model a workload came from.
            runtime : str (optional)
                An annotation indicating which runtime to use with a workload.
            mtriple : str (optional)
                The llvm triplet describing the target, for example "arm64-linux-android".
            mattr : List of str (optional)
                The llvm features to compile with, for example ["+avx512f", "+mmx"].
            mfloat-abi : str (optional)
                An llvm setting that is one of 'hard' or 'soft' indicating whether to use
                hardware or software floating-point operations.
        """
        if not isinstance(tag_or_str_or_dict, (dict, str, Target)):
            raise ValueError("target has to be a string or dictionary.")
        self.__init_handle_by_constructor__(_ffi_api.Target, tag_or_str_or_dict)

    def __enter__(self):
        _ffi_api.TargetEnterScope(self)
        return self

    def __exit__(self, ptype, value, trace):
        _ffi_api.TargetExitScope(self)

    def export(self):
        return _ffi_api.TargetExport(self)

    @staticmethod
    def current(allow_none=True):
        """Returns the current target.

        Parameters
        ----------
        allow_none : bool
            Whether allow the current target to be none

        Raises
        ------
        ValueError if current target is not set.
        """
        return _ffi_api.TargetCurrent(allow_none)

    @property
    def max_num_threads(self):
        return int(self.attrs["max_num_threads"])

    @property
    def thread_warp_size(self):
        return int(self.attrs["thread_warp_size"])

    @property
    def device_name(self):
        return str(self.attrs.get("device", ""))

    @property
    def model(self):
        """Returns model from the target if it exists."""
        return str(self.attrs.get("model", "unknown"))

    @property
    def mcpu(self):
        """Returns the mcpu from the target if it exists."""
        return str(self.attrs.get("mcpu", ""))

    @property
    def mattr(self):
        """Returns the mattr from the target if it exists."""
        return list(self.attrs.get("mattr", []))

    @property
    def libs(self):
        return list(self.attrs.get("libs", []))


# TODO(@tvm-team): Deprecate the helper functions below. Encourage the usage of config dict instead.


def _merge_opts(opts, new_opts):
    """Helper function to merge options"""
    if isinstance(new_opts, str):
        new_opts = new_opts.split()
    if new_opts:
        opt_set = set(opts)
        new_opts = [opt for opt in new_opts if opt not in opt_set]
        return opts + new_opts
    return opts


def cuda(model="unknown", options=None):
    """Returns a cuda target.

    Parameters
    ----------
    model: str
        The model of cuda device (e.g. 1080ti)
    options : str or list of str
        Additional options
    """
    opts = _merge_opts(["-model=%s" % model], options)
    return Target(" ".join(["cuda"] + opts))


def rocm(model="unknown", options=None):
    """Returns a ROCM target.

    Parameters
    ----------
    model: str
        The model of this device
    options : str or list of str
        Additional options
    """
    opts = _merge_opts(["-model=%s" % model], options)
    return Target(" ".join(["rocm"] + opts))


def mali(model="unknown", options=None):
    """Returns a ARM Mali GPU target.

    Parameters
    ----------
    model: str
        The model of this device
    options : str or list of str
        Additional options
    """
    opts = ["-device=mali", "-model=%s" % model]
    opts = _merge_opts(opts, options)
    return Target(" ".join(["opencl"] + opts))


def intel_graphics(model="unknown", options=None):
    """Returns an Intel Graphics target.

    Parameters
    ----------
    model: str
        The model of this device
    options : str or list of str
        Additional options
    """
    opts = ["-device=intel_graphics", "-model=%s" % model, "-thread_warp_size=16"]
    opts = _merge_opts(opts, options)
    return Target(" ".join(["opencl"] + opts))


def micro(hardware="unknown", options=None):
    """Returns a microTVM target.

    Parameters
    ----------
    hardware : str
        Canonically identifies the target device; typicaly one of cortex-mX, or a specific SoC model
        when that model has been tested to work with microTVM.
    options : str or list of str
        Additional options
    """
    trans_table = {"host": ["-mcpu=native"]}
    opts = _merge_opts(trans_table[hardware] + ["-runtime=c", "--system-lib"], options)

    # NOTE: in the future, the default micro target will be LLVM except when
    # external dependencies are present.
    return Target(" ".join(["c"] + opts))


def arm_cpu(model="unknown", options=None):
    """Returns a ARM CPU target.
    This function will also download pre-tuned op parameters when there is none.

    Parameters
    ----------
    model: str
        SoC name or phone name of the arm board.
    options : str or list of str
        Additional options
    """
    trans_table = {
        "pixel2": ["-model=snapdragon835", "-mtriple=arm64-linux-android", "-mattr=+neon"],
        "mate10": ["-model=kirin970", "-mtriple=arm64-linux-android", "-mattr=+neon"],
        "mate10pro": ["-model=kirin970", "-mtriple=arm64-linux-android", "-mattr=+neon"],
        "p20": ["-model=kirin970", "-mtriple=arm64-linux-android", "-mattr=+neon"],
        "p20pro": ["-model=kirin970", "-mtriple=arm64-linux-android", "-mattr=+neon"],
        "rasp3b": ["-model=bcm2837", "-mtriple=armv7l-linux-gnueabihf", "-mattr=+neon"],
        "rasp4b": [
            "-model=bcm2711",
            "-mtriple=armv8l-linux-gnueabihf",
            "-mattr=+neon",
            "-mcpu=cortex-a72",
        ],
        "rasp4b64": [
            "-model=bcm2711",
            "-mtriple=aarch64-linux-gnu",
            "-mattr=+neon",
            "-mcpu=cortex-a72",
        ],
        "rk3399": ["-model=rk3399", "-mtriple=aarch64-linux-gnu", "-mattr=+neon"],
        "pynq": ["-model=pynq", "-mtriple=armv7a-linux-eabi", "-mattr=+neon"],
        "ultra96": ["-model=ultra96", "-mtriple=aarch64-linux-gnu", "-mattr=+neon"],
        "beagleai": [
            "-model=beagleai",
            "-mtriple=armv7a-linux-gnueabihf",
            "-mattr=+neon,+vfp4,+thumb2",
            "-mcpu=cortex-a15",
        ],
        "stm32mp1": [
            "-model=stm32mp1",
            "-mtriple=armv7a-linux-gnueabihf",
            "-mattr=+neon,+vfp4,+thumb2",
        ],
        "thunderx": [
            "-model=thunderx",
            "-mtriple=aarch64-linux-gnu",
            "-mattr=+neon,+crc,+lse",
            "-mcpu=thunderxt88",
        ],
    }
    pre_defined_opt = trans_table.get(model, ["-model=%s" % model])

    opts = ["-device=arm_cpu"] + pre_defined_opt
    opts = _merge_opts(opts, options)
    return Target(" ".join(["llvm"] + opts))


def rasp(options=None):
    """Return a Raspberry 3b target.

    Parameters
    ----------
    options : str or list of str
        Additional options
    """
    warnings.warn(
        "tvm.target.rasp() is going to be deprecated. " 'Please use tvm.target.arm_cpu("rasp3b")'
    )
    return arm_cpu("rasp3b", options)


def vta(model="unknown", options=None):
    opts = ["-device=vta", "-keys=vta,cpu", "-model=%s" % model]
    opts = _merge_opts(opts, options)
    return Target(" ".join(["ext_dev"] + opts))


def bifrost(model="unknown", options=None):
    """Return an ARM Mali GPU target (Bifrost architecture).

    Parameters
    ----------
    options : str or list of str
        Additional options
    """
    opts = ["-device=bifrost", "-model=%s" % model]
    opts = _merge_opts(opts, options)
    return Target(" ".join(["opencl"] + opts))


def hexagon(cpu_ver="v66", sim_args=None, llvm_args=None, hvx=128):
    """Returns a Hexagon target.

    Parameters
    ----------
    cpu_ver : str
        CPU version used for code generation. Not all allowed cpu str
        will be valid, LLVM will throw an error.
    sim_args : str or list of str
        User defined sim arguments. CPU version defaults to cpu_ver.
        Otherwise, separate versions are used for codegen and sim. Not
        all allowed cpu strings will be valid, simulator will throw an
        error if invalid. Does not affect codegen.
    llvm_args : str or list of str
        User defined compiler arguments.
    hvx : int
        Size of hvx register. Value of 0 indicates disabled hvx.
    """
    # Example compiler arguments
    # llvm -mtriple=hexagon -mcpu=hexagonv66 -mattr=+hvxv66,+hvx-length128b

    # Check for valid codegen cpu
    valid_hex = ["v60", "v62", "v65", "v66", "v67", "v67t"]
    try:
        cpu_ver = cpu_ver[cpu_ver.index("v") :].lower()
        assert 3 <= len(cpu_ver) <= 4
    except:
        msg = "{} is not a valid Hexagon version\nvalid versions include {}"
        raise ValueError(msg.format(cpu_ver, valid_hex)) from None

    assert hvx in [0, 64, 128]

    # Target string
    def create_target(cpu_ver):
        target = " -mtriple=hexagon"
        mcpu = " -mcpu=hexagon" + cpu_ver
        mattr = ""
        # HVX enable
        if hvx:
            mattr = " -mattr=+hvx" + cpu_ver + ",+hvx-length" + str(hvx) + "b"
        return target + mcpu + mattr

    # Simulator string
    def create_sim(cpu_ver, sim_args):
        def validate_hvx_length(codegen_hvx, sim_args):
            if sim_args and "--hvx_length" in sim_args:
                # If --hvx_length was specified, check HVX length of sim
                # vs codegen
                i = sim_args.index("hvx_length") + len("hvx_length") + 1
                sim_hvx = sim_args[i : i + 3]
                if sim_hvx != str(codegen_hvx):
                    print(
                        "WARNING: sim hvx {} and codegen hvx {} mismatch!".format(
                            sim_hvx, codegen_hvx
                        )
                    )
            elif codegen_hvx != 0:
                # If --hvx_length was not given, add it if HVX is enabled
                sim_args = sim_args + " " if isinstance(sim_args, str) else ""
                sim_args += "--hvx_length " + str(codegen_hvx)
            return sim_args or ""

        if not sim_args:
            return cpu_ver + " " + validate_hvx_length(hvx, sim_args)

        sim_cpu = cpu_ver + " "

        # Add user defined args
        if isinstance(sim_args, list):
            sim_args = " ".join(sim_args)

        # Check for supplied sim cpu version
        if "v6" in sim_args:
            sim_cpu = ""

            # Regex match for allowed cpus
            valid_cpu_str_regex = (
                r"(?P<pre>--.*\s)?(--m)?"
                + r"(?P<base_version>v6[25678])(?P<sub_version>[a-z])?"
                + r"(?P<l2_size>_[0-9]+)?(?P<rev>_rev[0-9])?\s?(?P<post>--.*)?"
            )
            m = re.match(valid_cpu_str_regex, sim_args.lower())
            if not m:
                raise ValueError('Invalid simulator argument string "{}"'.format(sim_args))

            # Parse options into correct order
            cpu_attr = {x: str(m.groupdict()[x] or "") for x in m.groupdict()}
            sim_args = (
                cpu_attr["base_version"]
                + cpu_attr["sub_version"]
                + cpu_attr["l2_size"]
                + cpu_attr["rev"]
                + " "
                + cpu_attr["pre"]
                + cpu_attr["post"]
            )

        return sim_cpu + " " + validate_hvx_length(hvx, sim_args)

    # LLVM string
    def create_llvm(llvm_args):
        # TVM's option parser doesn't allow '=' in values, but '=' can
        # appear in LLVM flags. Replace it with '@', since it's unlikely
        # that '@' will be used in another context.
        if llvm_args is None or len(llvm_args.replace(" ", "")) == 0:
            return ""
        args = [s.replace("=", "@") for s in llvm_args.split()]
        return "--llvm-options=" + ",".join(args)

    # Sim args
    os.environ["HEXAGON_SIM_ARGS"] = create_sim(cpu_ver, sim_args)

    target_str = create_target(cpu_ver)
    llvm_str = create_llvm(llvm_args)
    args_list = target_str.split() + llvm_str.split()

    return Target(" ".join(["hexagon"] + args_list))


def create(target):
    """Deprecated. Use the constructor of :py:mod:`tvm.target.Target` directly."""
    warnings.warn("tvm.target.create() is being deprecated. Please use tvm.target.Target() instead")
    return Target(target)


@_register_func("target._load_config_dict")
def _load_config_dict(config_dict_str):
    try:
        config = json.loads(config_dict_str)
    except json.decoder.JSONDecodeError:
        return None
    if not isinstance(config, dict):
        return None
    for key in config.keys():
        if not isinstance(key, str):
            return None
    return config
