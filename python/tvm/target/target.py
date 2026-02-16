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
import re
import warnings
from typing import Union

import tvm_ffi
from tvm.runtime import Device
from tvm.runtime import Object, convert
from tvm.runtime.container import String
from tvm.ir.container import Map, Array

from . import _ffi_api


@tvm_ffi.register_object("target.TargetKind")
class TargetKind(Object):
    """Kind of a compilation target"""

    @property
    def options(self):
        """Returns the dict of available option names and types"""
        return dict(_ffi_api.ListTargetKindOptions(self))

    @staticmethod
    def options_from_name(kind_name: str):
        """Returns the dict of available option names and types from a name of TargetKind"""
        return dict(_ffi_api.ListTargetKindOptionsFromName(kind_name))


class TargetFeatures:
    def __init__(self, target):
        self.target = target

    def __getattr__(self, name: str):
        return _ffi_api.TargetGetFeature(self.target, name)


@tvm_ffi.register_object("target.Target")
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

    def __init__(self, target, host=None):
        """Construct a TVM target object from
        1) Raw target string
        2) Target config dict
        3) Target tag

        Parameters
        ----------
        target : Union[str, Dict[str, Any]]
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
            mabi : str (optional)
                An llvm setting. Generate code for the specified ABI, for example "lp64d".
            host : Union[str, Dict[str, Any]] (optional)
                Description for target host. Can be recursive. Similar to target.
        host : Optional[Union[str, Dict[str, Any]]]
            Similar to target but for target host. Can be one of a literal target host string,
            a json string describing a configuration, or a dictionary of configuration options.
            When using a dictionary or json string to configure target, the possible values are
            same as target.
        """
        if isinstance(target, (dict, str)):
            target = convert(target)
        if isinstance(host, (dict, str)):
            host = convert(host)
        if target is None or not isinstance(target, (Map, String, Target, str)):
            raise ValueError(f"target has to be a string or dictionary. instead get {type(target)}")
        if host is not None:
            if not isinstance(host, (Map, String, Target, str)):
                raise ValueError("target host has to be a string or dictionary.")
            self.__init_handle_by_constructor__(_ffi_api.Target, Target(target), Target(host))
        else:
            self.__init_handle_by_constructor__(_ffi_api.Target, target)

    def __enter__(self):
        _ffi_api.TargetEnterScope(self)
        return self

    def __exit__(self, ptype, value, trace):
        _ffi_api.TargetExitScope(self)

    def export(self):
        return _ffi_api.TargetExport(self)

    def with_host(self, host=None):
        return _ffi_api.WithHost(self, Target(host))

    @staticmethod
    def from_device(device: Union[str, Device]) -> "Target":
        """Detects Target associated with the given device. If the device does not exist,
        there will be an Error.

        Parameters
        ----------
        dev : Union[str, Device]
            The device to detect the target for.
            Supported device types: ["cuda", "metal", "rocm", "vulkan", "opencl", "cpu"]

        Returns
        -------
        target : Target
            The detected target.
        """
        from .detect_target import (  # pylint: disable=import-outside-toplevel
            detect_target_from_device,
        )

        return detect_target_from_device(device)

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
    def arch(self):
        """Returns the cuda arch from the target if it exists."""
        return str(self.attrs.get("arch", ""))

    @property
    def max_num_threads(self):
        """Returns the max_num_threads from the target if it exists."""
        return int(self.attrs["max_num_threads"])

    @property
    def max_block_size_x(self):
        """Returns the max block size in x-dimension from the target if it exists."""
        return int(self.attrs["max_block_size_x"])

    @property
    def max_block_size_y(self):
        """Returns the max block size in y-dimension from the target if it exists."""
        return int(self.attrs["max_block_size_y"])

    @property
    def thread_warp_size(self):
        """Returns the thread_warp_size from the target if it exists."""
        return int(self.attrs["thread_warp_size"])

    @property
    def max_shared_memory_per_block(self):
        return int(self.attrs["max_shared_memory_per_block"])

    @property
    def max_function_args(self):
        return int(self.attrs.get("max_function_args", 0))

    @property
    def vtcm_capacity(self):
        return int(self.attrs.get("vtcm-capacity", 0))

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
    def supports_integer_dot_product(self):
        if self.attrs.get("supports_integer_dot_product", []):
            return bool(self.attrs["supports_integer_dot_product"])
        if self.kind.name == "cuda":
            sm_version = int(self.arch.split("_")[1])
            if sm_version >= 61:
                return True
        return False

    @property
    def libs(self):
        return list(self.attrs.get("libs", []))

    @property
    def supports_cooperative_matrix(self):
        if self.attrs.get("supports_cooperative_matrix", []):
            return bool(self.attrs["supports_cooperative_matrix"])
        else:
            return False

    @property
    def features(self):
        return TargetFeatures(self)

    @property
    def l2_cache_size_bytes(self):
        return int(self.attrs.get("l2_cache_size_bytes", 0))

    def get_kind_attr(self, attr_name):
        """Get additional attribute about the target kind.

        Parameters
        ----------
        attr_name : str
            The attribute name.

        Returns
        -------
        value : object
            The attribute value
        """
        return _ffi_api.TargetKindGetAttr(self.kind, attr_name)

    def get_target_device_type(self):
        """Returns the device_type for this target."""
        return _ffi_api.TargetGetDeviceType(self)

    @staticmethod
    def list_kinds():
        """Returns the list of available target names."""
        return list(_ffi_api.ListTargetKinds())

    @staticmethod
    def canon_target(target):
        """Given a single target-like object, returns the TVM Target object representing it.
        Can convert from:
        - None (to None).
        - An existing TVM Target object.
        - A string, eg "cuda"
        - A Python dictionary, eg {"kind": "cuda", "arch": "sm_80" }
        """
        if target is None:
            return None
        if isinstance(target, Target):
            return target
        return Target(target)

    @staticmethod
    def canon_target_and_host(target, target_host=None):
        """Returns a TVM Target capturing target and target_host. Also returns the host in
        canonical form. The given target can be in any form recognized by
        Target.canon_target. If given, target_host can be in any form recognized by
        Target.canon_target. If target_host is given it will be set as the 'host' in the
        result Target object (and a warning given).

        Note that this method does not support heterogeneous compilation targets.
        """
        target = Target.canon_target(target)
        if target is None:
            assert target_host is None, "Target host is not empty when target is empty."
            return target, target_host
        if target.host is None and target_host is not None:
            warnings.warn(
                "target_host parameter is going to be deprecated. "
                "Please pass in tvm.target.Target(target, host=target_host) instead."
            )
            target_host = Target.canon_target(target_host)
            target = target.with_host(target_host)
        if target is not None:
            # In case the target already had a host, extract it here.
            target_host = target.host
        return target, target_host

    @staticmethod
    def canon_multi_target(multi_targets):
        """Given a single target-like object, or a collection-like object of target-like objects,
        returns a TVM Array of TVM Target objects representing then. Can convert from:
        - None (to None).
        - A single target-like object in a form recognized by canon_target.
        - A Python list or TVM Array of target-like objects in a form recognized by
        canon_target.
        - A Python dict or TVM Map from TVM IntImm objects representing device types to
        a target-like object in a form recognized by canon_target. (This is a legacy
        method to represent heterogeneous targets. The keys are ignored.)
        """
        if multi_targets is None:
            return None
        if isinstance(multi_targets, (dict, Map)) and "kind" not in multi_targets:
            # Convert legacy heterogeneous map representation to ordinary list of targets.
            return Target.canon_multi_target(list(multi_targets.values()))
        if isinstance(multi_targets, (list, Array)):
            # Multiple Target results.
            return convert([Target.canon_target(tgt) for tgt in multi_targets])
        # Single Target result.
        return convert([Target.canon_target(multi_targets)])

    @staticmethod
    def canon_multi_target_and_host(target, target_host=None):
        """Returns a TVM Array<Target> capturing target and target_host. The given target can be in
        any form recognized by Target.canon_multi_target. If given, target_host can be in
        any form recognized by Target.canon_target. If target_host is given it will be set
        as the 'host' in each result Target object (and a warning given).
        """
        # Convert target to Array<Target>, but not yet accounting for any host.
        raw_targets = Target.canon_multi_target(target)
        assert raw_targets is not None and len(raw_targets) > 0
        # Convert host to Target, if given.
        if raw_targets[0].host is None and target_host is not None:
            warnings.warn(
                "target_host parameter is going to be deprecated. "
                "Please pass in tvm.target.Target(target, host=target_host) instead."
            )
            # Make sure the (canonical) host is captured in all the (canonical) targets.
            target_host = Target.canon_target(target_host)
            raw_targets = convert([tgt.with_host(target_host) for tgt in raw_targets])
        return raw_targets

    @staticmethod
    def canon_target_map_and_host(target_map, target_host=None):
        """Returns target_map as a map from TVM Target's in canonical form to IRModules. The keys
        of the input target_map can be in any form recognized by Target.canon_target.
        Similarly, if given, target_host can be in any form recognized by
        Target.canon_target. The final target_map keys will capture the target_host in
        canonical form. Also returns the target_host in canonical form."""
        new_target_map = {}
        canonical_target_host = None
        for tgt, mod in target_map.items():
            tgt = Target.canon_target(tgt)
            assert tgt is not None
            if canonical_target_host is None:
                if tgt.host is not None:
                    canonical_target_host = tgt.host
                elif target_host is not None:
                    # No deprecation warning in this case since host may have been manufactured
                    # behind the scenes in build_module.py build.
                    canonical_target_host = Target.canon_target(target_host)
            if tgt.host is None and canonical_target_host is not None:
                tgt = tgt.with_host(canonical_target_host)
            new_target_map[tgt] = mod
        return new_target_map, canonical_target_host

    @staticmethod
    def target_or_current(target):
        """Returns target, or the current target in the environment if target is None"""
        if target is None:
            target = Target.current()
        if target is None:
            raise ValueError("Target is not set in env or passed as argument.")
        return target


def cuda(model="unknown", arch=None, **kwargs):
    """Returns a cuda target.

    Parameters
    ----------
    model: str
        The model of cuda device (e.g. 1080ti)
    arch: str
        The cuda architecture (e.g. sm_61)
    kwargs: dict
        Additional target attributes
    """
    config = {"kind": "cuda", "model": model}
    if arch:
        config["arch"] = arch
    else:
        warnings.warn("Try specifying cuda arch by adding arch='sm_xx' to your target.")
    config.update(kwargs)
    return Target(config)


def rocm(model="unknown", **kwargs):
    """Returns a ROCM target.

    Parameters
    ----------
    model: str
        The model of this device
    kwargs: dict
        Additional target attributes
    """
    config = {"kind": "rocm", "model": model}
    config.update(kwargs)
    return Target(config)


def mali(model="unknown", **kwargs):
    """Returns a ARM Mali GPU target.

    Parameters
    ----------
    model: str
        The model of this device
    kwargs: dict
        Additional target attributes
    """
    config = {"kind": "opencl", "device": "mali", "model": model}
    config.update(kwargs)
    return Target(config)


def intel_graphics(model="unknown", **kwargs):
    """Returns an Intel Graphics target.

    Parameters
    ----------
    model: str
        The model of this device
    kwargs: dict
        Additional target attributes
    """
    config = {
        "kind": "opencl",
        "device": "intel_graphics",
        "model": model,
        "thread_warp_size": 16,
    }
    config.update(kwargs)
    return Target(config)


MICRO_SUPPORTED_MODELS = {
    "host": {},
    "atsamd51": {"mcpu": "cortex-m4"},
    "cxd5602gg": {"mcpu": "cortex-m4"},
    "esp32": {},
    "imxrt10xx": {"mcpu": "cortex-m7"},
    "mps2_an521": {"mcpu": "cortex-m33"},
    "mps3_an547": {"mcpu": "cortex-m55"},
    "nrf52840": {"mcpu": "cortex-m4+nodsp"},
    "nrf5340dk": {"mcpu": "cortex-m33"},
    "rp2040": {"mcpu": "cortex-m0"},
    "sam3x8e": {"mcpu": "cortex-m3"},
    "stm32f746xx": {"mcpu": "cortex-m7", "march": "armv7e-m"},
    "stm32h7xx": {"mcpu": "cortex-m7"},
    "stm32l4r5zi": {"mcpu": "cortex-m4"},
    "stm32u5xx": {"mcpu": "cortex-m33"},
    "zynq_mp_r5": {"mcpu": "cortex-r5"},
}


def _parse_cli_opts_to_dict(opts):
    """Convert a list of CLI-style options (e.g. ['-mcpu=cortex-a72']) to a dict."""
    result = {}
    for opt in opts:
        opt = opt.lstrip("-")
        if "=" in opt:
            key, val = opt.split("=", 1)
            # Handle comma-separated values as lists
            if "," in val:
                val = val.split(",")
            result[key] = val
        else:
            result[opt] = True
    return result


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
        "pixel2": {"model": "snapdragon835", "mtriple": "arm64-linux-android", "mattr": ["+neon"]},
        "mate10": {"model": "kirin970", "mtriple": "arm64-linux-android", "mattr": ["+neon"]},
        "mate10pro": {"model": "kirin970", "mtriple": "arm64-linux-android", "mattr": ["+neon"]},
        "p20": {"model": "kirin970", "mtriple": "arm64-linux-android", "mattr": ["+neon"]},
        "p20pro": {"model": "kirin970", "mtriple": "arm64-linux-android", "mattr": ["+neon"]},
        "rasp3b": {
            "model": "bcm2837",
            "mtriple": "armv7l-linux-gnueabihf",
            "mattr": ["+neon"],
        },
        "rasp4b": {
            "model": "bcm2711",
            "mtriple": "armv8l-linux-gnueabihf",
            "mattr": ["+neon"],
            "mcpu": "cortex-a72",
        },
        "rasp4b64": {
            "model": "bcm2711",
            "mtriple": "aarch64-linux-gnu",
            "mattr": ["+neon"],
            "mcpu": "cortex-a72",
        },
        "rk3399": {"model": "rk3399", "mtriple": "aarch64-linux-gnu", "mattr": ["+neon"]},
        "pynq": {"model": "pynq", "mtriple": "armv7a-linux-eabi", "mattr": ["+neon"]},
        "ultra96": {"model": "ultra96", "mtriple": "aarch64-linux-gnu", "mattr": ["+neon"]},
        "beagleai": {
            "model": "beagleai",
            "mtriple": "armv7a-linux-gnueabihf",
            "mattr": ["+neon", "+vfp4", "+thumb2"],
            "mcpu": "cortex-a15",
        },
        "stm32mp1": {
            "model": "stm32mp1",
            "mtriple": "armv7a-linux-gnueabihf",
            "mattr": ["+neon", "+vfp4", "+thumb2"],
            "mcpu": "cortex-a7",
        },
        "thunderx": {
            "model": "thunderx",
            "mtriple": "aarch64-linux-gnu",
            "mattr": ["+neon", "+crc", "+lse"],
            "mcpu": "thunderxt88",
        },
    }
    pre_defined = trans_table.get(model, {"model": model})

    config = {
        "kind": "llvm",
        "keys": ["arm_cpu", "cpu"],
        "device": "arm_cpu",
    }
    config.update(pre_defined)
    if options:
        if isinstance(options, str):
            options = options.split()
        config.update(_parse_cli_opts_to_dict(options))
    return Target(config)


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


def bifrost(model="unknown", **kwargs):
    """Return an ARM Mali GPU target (Bifrost architecture).

    Parameters
    ----------
    model: str
        The model of this device
    kwargs: dict
        Additional target attributes
    """
    config = {"kind": "opencl", "device": "bifrost", "model": model}
    config.update(kwargs)
    return Target(config)


def riscv_cpu(model="sifive-u54", options=None):
    """Returns a RISC-V CPU target.
    Default: sifive-u54 rv64gc

    Parameters
    ----------
    model: str
        CPU name.
    options : str or list of str
        Additional options
    """
    trans_table = {
        "sifive-e31": {
            "model": "sifive-e31",
            "mtriple": "riscv32-unknown-linux-gnu",
            "mcpu": "sifive-e31",
            "mabi": "ilp32",
        },
        "sifive-e76": {
            "model": "sifive-e76",
            "mtriple": "riscv32-unknown-linux-gnu",
            "mcpu": "sifive-e76",
            "mabi": "ilp32",
        },
        "sifive-u54": {
            "model": "sifive-u54",
            "mtriple": "riscv64-unknown-linux-gnu",
            "mcpu": "sifive-u54",
            "mabi": "lp64d",
        },
        "sifive-u74": {
            "model": "sifive-u74",
            "mtriple": "riscv64-unknown-linux-gnu",
            "mcpu": "sifive-u74",
            "mabi": "lp64d",
        },
        "licheepi3a": {
            "num-cores": 8,
            "mtriple": "riscv64-unknown-linux-gnu",
            "mcpu": "spacemit-x60",
            "mfloat-abi": "hard",
            "mabi": "lp64d",
        },
    }
    pre_defined = trans_table.get(model, {"model": model})

    config = {
        "kind": "llvm",
        "keys": ["arm_cpu", "cpu"],
        "device": "arm_cpu",
    }
    config.update(pre_defined)
    if options:
        if isinstance(options, str):
            options = options.split()
        config.update(_parse_cli_opts_to_dict(options))
    return Target(config)


def hexagon(cpu_ver="v68", **kwargs):
    """Returns a Hexagon target.

    Parameters
    ----------
    cpu_ver : str (default: "v68")
        CPU version used for code generation. Not all allowed cpu str
        will be valid, LLVM will throw an error.

    Recognized keyword parameters
    -----------------------------
    hvx : int (default: 128)
        Size of HVX vector in bytes. Value of 0 disables HVX codegen.
    llvm_options : str or list of str (default: None)
        User defined compiler arguments.
    use_qfloat : bool (default: True for cpu_ver >= v68, False otherwise)
        Whether to use QFloat HVX instructions.
    use_ieee_fp : bool (default: False)
        Whether to use IEEE HVX instructions
    num_cores : int (default: 4)
        The number of HVX threads. This attribute is required by meta scheduler.
    vtcm_capacity: int (default: 0)
        Hexagon VTCM capacity limitation. If the value is 0, the capacity is treated as unbounded.

    Note: Floating point support in HVX requires LLVM 14+.
    """

    def get_arch_version(cpu_ver):
        m = re.match(r"v([0-9]+).*", cpu_ver)
        assert m
        return int(m.group(1))

    # Check for valid codegen cpu
    valid_hex = ["v65", "v66", "v67", "v67t", "v68", "v69", "v71", "v73", "v75"]
    try:
        cpu_ver = cpu_ver[cpu_ver.index("v") :].lower()
        assert cpu_ver in valid_hex
    except:
        msg = "{} is not a valid Hexagon version\nvalid versions include {}"
        raise ValueError(msg.format(cpu_ver, valid_hex)) from None

    def get_vtcm_capacity(cpu_ver):
        one_mb = 2**20
        default_vtcm_sizes = {
            "v65": one_mb // 4,
            "v66": one_mb // 4,
            "v68": 4 * one_mb,
            "v69": 8 * one_mb,
            "v73": 8 * one_mb,
            "v75": 8 * one_mb,
        }
        return default_vtcm_sizes.get(cpu_ver, 0)

    arch_version = get_arch_version(cpu_ver)
    local_config = {
        "hvx": 128,
        "llvm_options": None,
        "use_qfloat": arch_version >= 68,
        "use_ieee_fp": False,
        "vtcm_capacity": get_vtcm_capacity(cpu_ver),
    }
    local_config.update(kwargs)

    # Warn about obsolete parameter names.
    if local_config.get("sim_args") or local_config.get("sim_options"):
        msg = (
            "Setting simulator options in target is deprecated, set environment variable "
            "HEXAGON_SIM_ARGS instead"
        )
        warnings.warn(msg, stacklevel=2)
    if local_config.get("llvm_args"):
        msg = "The keyword parameter 'llvm_args' is deprecated, use 'llvm_options' instead"
        warnings.warn(msg, stacklevel=2)
        local_config.update({"llvm_options": local_config["llvm_args"]})

    # Build mattr list from config
    features_map = {
        "use_qfloat": "hvx-qfloat",
        "use_ieee_fp": "hvx-ieee-fp",
    }
    mattr = []
    if local_config["hvx"] > 0:
        valid_hvx = [0, 64, 128]
        if local_config["hvx"] not in valid_hvx:
            raise ValueError("Invalid hvx value, should be one of " + str(valid_hvx))
        mattr += ["+hvx" + cpu_ver, "+hvx-length" + str(local_config["hvx"]) + "b"]
    else:
        mattr += ["-hvx"]
    if arch_version >= 68:
        for f, feat_name in features_map.items():
            mattr.append(("-", "+")[local_config[f]] + feat_name)

    # Build llvm-options list
    llvm_options_list = []
    llvm_options = local_config["llvm_options"]
    if arch_version == 68:
        if not llvm_options:
            llvm_options = ""
        llvm_options += " -force-hvx-float"
    if llvm_options and len(llvm_options.strip()) > 0:
        llvm_options_list = [s.replace("=", "@") for s in llvm_options.split()]

    num_cores = local_config["num_cores"] if "num_cores" in kwargs else 4

    target_config = {
        "kind": "hexagon",
        "mtriple": "hexagon",
        "mcpu": "hexagon" + cpu_ver,
        "mattr": mattr,
        "num-cores": num_cores,
        "vtcm-capacity": local_config["vtcm_capacity"],
    }
    if llvm_options_list:
        target_config["llvm-options"] = llvm_options_list

    return Target(target_config)


STM32_SUPPORTED_SERIES = {
    # High-Performance
    "stm32H7xx": {"mcpu": "cortex-m7", "march": "armv7e-m"},
    "stm32F7xx": {"mcpu": "cortex-m7"},
    "stm32F4xx": {"mcpu": "cortex-m4"},
    "stm32F2xx": {"mcpu": "cortex-m3"},
    # Mainstream
    "stm32G0xx": {"mcpu": "cortex-m0+"},
    "stm32F0xx": {"mcpu": "cortex-m0"},
    "stm32F1xx": {"mcpu": "cortex-m3"},
    "stm32G4xx": {"mcpu": "cortex-m4"},
    "stm32F3xx": {"mcpu": "cortex-m4"},
    # Low-power
    "stm32U5xx": {"mcpu": "cortex-m33"},
    "stm32L5xx": {"mcpu": "cortex-m33"},
    "stm32L4xx": {"mcpu": "cortex-m4"},
    "stm32L1xx": {"mcpu": "cortex-m3"},
    "stm32L0xx": {"mcpu": "cortex-m0+"},
}


def stm32(series="unknown", options=None):
    """Returns a STM32 target.

    Parameters
    ----------
    series: str
        Series name of a STM32 board series, eg. stm32H7xx or stm32F4xx
    options : str or list of str
        Additional options
    """
    if series not in STM32_SUPPORTED_SERIES:
        raise ValueError(f"Series {series} is not supported by tvm.target.stm32.")
    config = {
        "kind": "c",
        "keys": ["arm_cpu", "cpu"],
        "device": "arm_cpu",
    }
    config.update(STM32_SUPPORTED_SERIES[series])
    if options:
        if isinstance(options, str):
            options = options.split()
        config.update(_parse_cli_opts_to_dict(options))
    return Target(config)


def adreno(model="unknown", options=None, cfg=None, backend="opencl"):
    """Returns a Qualcomm GPU target.
    Parameters
    ----------
    model: str
        The model of this device
    options : str or list of str
        Additional options
    cfg : str
        Additional hints for target pipeline behavior
    backend : str
            Backend API, can be "opencl" or "vulkan"
    """

    if backend not in ["opencl", "vulkan"]:
        raise ValueError(f"Unsupported API: {backend}. Must be 'opencl' or 'vulkan'.")

    keys = ["adreno", backend, "gpu"]
    if cfg:
        keys.append(cfg)

    config = {
        "kind": backend,
        "device": "adreno",
        "keys": keys,
        "model": model,
    }
    if options:
        if isinstance(options, str):
            options = options.split()
        config.update(_parse_cli_opts_to_dict(options))
    return Target(config)


def create(target):
    """Deprecated. Use the constructor of :py:mod:`tvm.target.Target` directly."""
    warnings.warn("tvm.target.create() is being deprecated. Please use tvm.target.Target() instead")
    return Target(target)
