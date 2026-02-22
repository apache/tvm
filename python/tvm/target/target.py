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

from typing import Union

import tvm_ffi

from tvm.ir.container import Map
from tvm.runtime import Device, Object, convert
from tvm.runtime.container import String

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

    Targets can be constructed from:

    - A JSON config dictionary: ``Target({"kind": "cuda", "arch": "sm_80"})``
    - A tag name: ``Target("nvidia/nvidia-a100")``
    - A tag with overrides: ``Target({"tag": "nvidia/nvidia-a100", "l2_cache_size_bytes": 12345})``
    - A kind name: ``Target("cuda")``

    Use ``target.attrs["key"]`` to access target attributes.

    Examples
    --------
    .. code-block:: python

        # From a tag
        target = Target("nvidia/nvidia-a100")

        # From a tag with attribute overrides
        target = Target({"tag": "qcom/hexagon-v68", "vtcm-capacity": 70000})

        # From a config dictionary
        target = Target({"kind": "cuda", "arch": "sm_80"})
    """

    def __init__(self, target, host=None):
        """Construct a TVM target object from
        1) Raw target string
        2) Target config dict
        3) Target tag string
        4) Tag with overrides dict

        Parameters
        ----------
        target : Union[str, Dict[str, Any]]
            Can be one of a literal target string, a json string describing
            a configuration, or a dictionary of configuration options.
            When using a dictionary or json string to configure target, the
            possible values are:

            tag : str (optional)
                A registered tag name (e.g. ``"nvidia/nvidia-a100"``).
                When ``tag`` is present, the tag's base config is loaded and
                any additional fields in the dict override the base values.
                The ``kind`` field is not needed when ``tag`` is specified.
            kind :  str (required unless tag is specified)
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
    def features(self):
        return TargetFeatures(self)

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
    def target_or_current(target):
        """Returns target, or the current target in the environment if target is None"""
        if target is None:
            target = Target.current()
        if target is None:
            raise ValueError("Target is not set in env or passed as argument.")
        return target
