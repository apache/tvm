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
# pylint: disable=invalid-name, missing-function-docstring
"""A utility method to run a TVM module on a remote device."""
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

from typing_extensions import Literal

if TYPE_CHECKING:
    import numpy as np

    from tvm.meta_schedule.runner import EvaluatorConfig, RPCConfig
    from tvm.runtime import Device, Module, NDArray

# pylint: disable=import-outside-toplevel,protected-access


def _args_to_device(args, device):
    import numpy as np

    from tvm.runtime.ndarray import NDArray, empty

    uploaded_args = []
    for arg in args:
        if isinstance(arg, (np.ndarray, NDArray)):
            uploaded_args.append(empty(arg.shape, dtype=arg.dtype, device=device).copyfrom(arg))
        elif isinstance(arg, (int, float)):
            uploaded_args.append(arg)
        else:
            raise ValueError(f"Unsupported input type: {type(arg)}")
    return uploaded_args


def _args_to_numpy(args):
    from tvm.runtime.ndarray import NDArray

    downloaded_args = []
    for arg in args:
        if isinstance(arg, NDArray):
            downloaded_args.append(arg.numpy())
        else:
            downloaded_args.append(arg)
    return downloaded_args


def _normalize_export_func(export_func, output_format) -> Tuple[Callable, str]:
    from tvm.contrib import ndk, tar

    def export_with(func):
        return lambda mod, path: mod.export_library(path, fcompile=func)

    if export_func == "tar":
        export_func = export_with(tar.tar)
        output_format = "tar"
    elif export_func == "ndk":
        export_func = export_with(ndk.create_shared)
        output_format = "so"
    elif callable(export_func):
        if output_format is None:
            raise ValueError("output_format must be specified if `export_func` is callable")
    else:
        raise ValueError(f"Unsupported export_func: {export_func}")
    return export_func, output_format


def local_run(  # pylint: disable=too-many-arguments,too-many-locals
    mod: "Module",
    device_type: str,
    args: List[Union["np.ndarray", "NDArray", int, float]],
    evaluator_config: Optional["EvaluatorConfig"] = None,
    export_func: Union[Callable[["Module", str], None], Literal["tar", "ndk"]] = "tar",
    output_format: Optional[str] = None,
):
    """Run a TVM module on a local device.

    Parameters
    ----------
    mod : Module
        The TVM module to run.
    device_type : str
        The device type to run the module on.
    args : List[Union[np.ndarray, NDArray, int, float]]
        The arguments to be fed to the module.
    evaluator_config : Optional[EvaluatorConfig]
        The evaluator configuration to use.
    export_func : Union[Callable[Module, str], Literal["tar", "ndk"]]
        The function to export the module to a file.
        If callable, it must be a function that takes two arguments: the module to export and the
        path to export to.
        If "tar", the module will be exported to a tar file.
        If "ndk", the module will be exported to a shared library.
    output_format : Optional[str]
        The format of the exported module.
        If not specified, it will be inferred from the `export_func` argument.

    Returns
    -------
    args : List[Union[np.ndarray, NDArray, int, float]]
        The results of running the module.
    profile_result : tvm.runtime.BenchmarkResult
        The profiling result of running the module.
    """
    import os.path as osp
    import tempfile

    from tvm.meta_schedule.runner import EvaluatorConfig
    from tvm.runtime import device, load_module

    evaluator_config = EvaluatorConfig._normalized(evaluator_config)
    export_func, output_format = _normalize_export_func(export_func, output_format)

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = osp.join(tmp_dir, "tvm_tmp_mod." + output_format)
        export_func(mod, artifact_path)
        device: Device = device(device_type, 0)

        try:
            args = _args_to_device(args, device)
            remote_mod = load_module(artifact_path)
            profile_result = remote_mod.time_evaluator(
                func_name=remote_mod.entry_name,
                dev=device,
                number=evaluator_config.number,
                repeat=evaluator_config.repeat,
                min_repeat_ms=evaluator_config.min_repeat_ms,
                f_preproc="cache_flush_cpu_non_first_arg"
                if evaluator_config.enable_cpu_cache_flush
                else "",
            )(*args)
            remote_mod(*args)
            args = _args_to_numpy(args)
        finally:
            pass

    return args, profile_result


def rpc_run(  # pylint: disable=too-many-arguments,too-many-locals
    mod: "Module",
    device_type: str,
    args: List[Union["np.ndarray", "NDArray", int, float]],
    evaluator_config: Optional["EvaluatorConfig"] = None,
    rpc_config: Optional["RPCConfig"] = None,
    export_func: Union[Callable[["Module", str], None], Literal["tar", "ndk"]] = "tar",
    output_format: Optional[str] = None,
):
    """Run a TVM module on a remote device.

    Parameters
    ----------
    mod : Module
        The TVM module to run.
    device_type : str
        The device type to run the module on.
    args : List[Union[np.ndarray, NDArray, int, float]]
        The arguments to be fed to the module.
    evaluator_config : Optional[EvaluatorConfig]
        The evaluator configuration to use.
    rpc_config : Optional[RPCConfig]
        The RPC configuration to connect to the remote device.
        If not specified, the default RPC configuration will be used, which reads the following
        environment variables:
        - TVM_TRACKER_HOST
        - TVM_TRACKER_PORT
        - TVM_TRACKER_KEY
    export_func : Union[Callable[Module, str], Literal["tar", "ndk"]]
        The function to export the module to a file.
        If callable, it must be a function that takes two arguments: the module to export and the
        path to export to.
        If "tar", the module will be exported to a tar file.
        If "ndk", the module will be exported to a shared library.
    output_format : Optional[str]
        The format of the exported module.
        If not specified, it will be inferred from the `export_func` argument.

    Returns
    -------
    args : List[Union[np.ndarray, NDArray, int, float]]
        The results of running the module.
    profile_result : tvm.runtime.BenchmarkResult
        The profiling result of running the module.
    """

    import os.path as osp
    import tempfile

    from tvm.meta_schedule.runner import EvaluatorConfig, RPCConfig

    evaluator_config = EvaluatorConfig._normalized(evaluator_config)
    rpc_config = RPCConfig._normalized(rpc_config)
    export_func, output_format = _normalize_export_func(export_func, output_format)

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifact_path = osp.join(tmp_dir, "tvm_tmp_mod." + output_format)
        _, remote_path = osp.split(artifact_path)
        session = rpc_config.connect_server()
        device: Device = session.device(device_type, 0)

        export_func(mod, artifact_path)
        try:
            session.upload(artifact_path, remote_path)
            args = _args_to_device(args, device)
            remote_mod = session.load_module(remote_path)
            profile_result = remote_mod.time_evaluator(
                func_name=remote_mod.entry_name,
                dev=device,
                number=evaluator_config.number,
                repeat=evaluator_config.repeat,
                min_repeat_ms=evaluator_config.min_repeat_ms,
                f_preproc="cache_flush_cpu_non_first_arg"
                if evaluator_config.enable_cpu_cache_flush
                else "",
            )(*args)
            remote_mod(*args)
            args = _args_to_numpy(args)
        finally:
            session.remove(remote_path)
            session.remove(remote_path + "." + output_format)
            session.remove("")

    return args, profile_result
