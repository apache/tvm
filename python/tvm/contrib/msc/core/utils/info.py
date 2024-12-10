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
"""tvm.contrib.msc.core.utils.info"""

from typing import List, Tuple, Dict, Any, Union
from packaging.version import parse
import numpy as np

import tvm
from tvm.contrib.msc.core import _ffi_api
from .namespace import MSCFramework


class MSCArray(object):
    """MSC wrapper for array like object

    Parameters
    ----------
    data: array_like: np.ndarray| torch.Tensor| tvm.ndarray| ...
        The data object.
    """

    def __init__(self, data: Any):
        self._meta_data = data
        self._framework, self._type, self._device = self._analysis(data)

    def __str__(self):
        return "<{} @{}>{}".format(self._framework, self._device, self.abstract())

    def _analysis(self, data: Any) -> Tuple[str, str, np.ndarray]:
        if isinstance(data, (list, tuple)) and all(isinstance(d, (int, float)) for d in data):
            return MSCFramework.MSC, "list", "cpu"
        if isinstance(data, np.ndarray):
            return MSCFramework.MSC, "tensor", "cpu"
        if isinstance(data, tvm.runtime.NDArray):
            device = tvm.runtime.Device.MASK2STR[data.device.device_type]
            if data.device.device_id:
                device += ":{}".format(data.device.device_id)
            return MSCFramework.TVM, "tensor", device
        if isinstance(data, tvm.relax.Var):
            return MSCFramework.TVM, "var", "cpu"
        try:
            import torch  # pylint: disable=import-outside-toplevel

            if isinstance(data, torch.Tensor):
                ref_dev = data.device
                if ref_dev.index:
                    device = "{}:{}".format(ref_dev.type, ref_dev.index)
                else:
                    device = ref_dev.type
                return MSCFramework.TORCH, "tensor", device
        except:  # pylint: disable=bare-except
            pass

        raise Exception("Unkonwn data {}({})".format(data, type(data)))

    def abstract(self) -> str:
        """Get abstract describe of the data"""

        data = self._to_ndarray()
        prefix = "[{},{}]".format(";".join([str(s) for s in data.shape]), data.dtype.name)
        if data.size < 10:
            return "{} {}".format(prefix, ",".join([str(i) for i in data.flatten()]))
        return "{} Max {:g}, Min {:g}, Avg {:g}".format(
            prefix, data.max(), data.min(), data.sum() / data.size
        )

    def _to_ndarray(self) -> np.ndarray:
        """Cast array like object to np.ndarray

        Returns
        -------
        data: np.ndarray
            The data as np.ndarray.
        """

        if self._framework == MSCFramework.MSC:
            if self._type == "list":
                return np.array(self._meta_data)
            return self._meta_data
        if self._framework == MSCFramework.TVM:
            if self._type == "var":
                shape = [int(s) for s in self._meta_data.struct_info.shape]
                return np.zeros(shape, dtype=self._meta_data.struct_info.dtype)
            return self._meta_data.asnumpy()
        if self._framework == MSCFramework.TORCH:
            return self._meta_data.detach().cpu().numpy()
        return self._meta_data

    def _to_device(self, device: str) -> Any:
        """Cast array like object to array like object

        Parameters
        ----------
        device: str
            The device for tensor.

        Returns
        -------
        output:
            The output as framework tensor.
        """

        if self._device == device:
            return self._meta_data
        if self._framework == MSCFramework.TORCH:
            return self._meta_data.to(self.get_device(device))
        if self._framework == MSCFramework.TVM:
            return tvm.nd.array(self._cast_data(), device=self.get_device(device))
        return self._meta_data

    def cast(self, framework: str, device: str = "cpu") -> Any:
        """Cast array like object to array like object

        Parameters
        ----------
        framework: str
            The target framework.
        device: str
            The device for tensor.

        Returns
        -------
        output:
            The output as framework tensor.
        """

        device = device or self._device
        if framework == self._framework and device == self._device and self._type == "tensor":
            return self._meta_data
        if framework == self._framework:
            return self._to_device(device)
        data = self._to_ndarray()
        if framework == MSCFramework.TORCH:
            import torch  # pylint: disable=import-outside-toplevel

            return torch.from_numpy(data).to(self.get_device(device, framework))
        if framework == MSCFramework.TVM:
            return tvm.nd.array(data, device=self.get_device(device, framework))
        return data

    def get_device(self, device: str, framework: str = None) -> Any:
        """Change device from name to device obj

        Parameters
        ----------
        device: str
            The device for tensor.
        framework: str
            The target framework.

        Returns
        -------
        device: any
            The device object.
        """

        framework = framework or self._framework
        if framework == MSCFramework.TVM:
            if device.startswith("cpu"):
                return tvm.cpu()
            if device.startswith("cuda"):
                dev_id = int(device.split(":")[1]) if ":" in device else 0
                return tvm.cuda(dev_id)
            raise TypeError("Unexpected tvm device " + str(device))
        if framework == MSCFramework.TORCH:
            import torch  # pylint: disable=import-outside-toplevel

            return torch.device(device)
        return device

    @classmethod
    def is_array(cls, data: Any) -> bool:
        """Check if the data is array like

        Parameters
        ----------
        data: array_like: np.ndarray| torch.Tensor| tvm.ndarray| ...
            The data object.

        Returns
        -------
        is_array: bool
            Whether the data is array like.
        """

        normal_types = (np.ndarray, tvm.runtime.NDArray, tvm.relax.Var)
        if isinstance(data, normal_types):
            return True
        if isinstance(data, (list, tuple)) and all(isinstance(d, (int, float)) for d in data):
            return True
        try:
            import torch  # pylint: disable=import-outside-toplevel

            if isinstance(data, torch.Tensor):
                return True
        except:  # pylint: disable=bare-except
            pass

        return False

    @property
    def framework(self):
        return self._framework

    @property
    def device(self):
        return self._device

    @property
    def type(self):
        return self._type


def is_array(data: Any) -> bool:
    """Check if the data is array

    Parameters
    ----------
    data: array_like: np.ndarray| torch.Tensor| tvm.ndarray| ...
        The data object.

    Returns
    -------
    is_array: bool
        Whether the data is array.
    """

    return MSCArray.is_array(data)


def cast_array(data: Any, framework: str = MSCFramework.MSC, device: str = "cpu") -> Any:
    """Cast array like object to np.ndarray

    Parameters
    ----------
    data: array_like: np.ndarray| torch.Tensor| tvm.ndarray| ...
        The data object.
    framework: str
        The target framework.
    device: str
        The device for tensor.

    Returns
    -------
    output: np.ndarray
        The output as numpy array or framework tensor(if given).
    """

    assert MSCArray.is_array(data), "{} is not array like".format(data)
    return MSCArray(data).cast(framework, device)


def inspect_array(data: Any, as_str: bool = True) -> Union[Dict[str, Any], str]:
    """Inspect the array

    Parameters
    ----------
    data: array like
        The data to inspect
    as_str: bool
        Whether inspect the array as string.

    Returns
    -------
    info: dict
        The data info.
    """

    if not MSCArray.is_array(data):
        return str(data)
    if as_str:
        return str(MSCArray(data))
    data = cast_array(data)
    return {
        "shape": list(data.shape),
        "dtype": data.dtype.name,
        "max": float(data.max()),
        "min": float(data.min()),
        "avg": float(data.sum() / data.size),
    }


def compare_arrays(
    golden: Dict[str, Any],
    datas: Dict[str, Any],
    atol: float = 1e-2,
    rtol: float = 1e-2,
    report_detail: bool = False,
) -> dict:
    """Compare elements in array

    Parameters
    ----------
    golden: dict<str, array_like>
        The golden datas.
    datas: dict<str, array_like>
        The datas to be compared.
    atol: float
        The atol for compare.
    rtol: float
        The rtol for compare.
    report_detail: bool
        Whether to report detail

    Returns
    -------
    report: dict
        The compare results.
    """

    assert golden.keys() == datas.keys(), "golden {} and datas {} mismatch".format(
        golden.keys(), datas.keys()
    )
    golden = {k: cast_array(v) for k, v in golden.items()}
    datas = {k: cast_array(v) for k, v in datas.items()}
    report = {"total": 0, "passed": 0, "info": {}}

    def _add_report(name: str, gol: Any, data: Any, passed: bool):
        diff = MSCArray(gol - data)
        if passed:
            if report_detail:
                report["info"][name] = {
                    "data": MSCArray(data).abstract(),
                    "d_pass": diff.abstract(),
                }
            else:
                report["info"][name] = "d_pass: {}".format(diff.abstract())
            report["passed"] += 1
        else:
            if report_detail:
                report["info"][name] = {
                    "gold": MSCArray(gol).abstract(),
                    "data": MSCArray(data).abstract(),
                    "d_fail": diff.abstract(),
                }
            else:
                report["info"][name] = "d_fail: {}".format(diff.abstract())

    for name, gol in golden.items():
        report["total"] += 1
        data = datas[name]
        if list(gol.shape) != list(data.shape):
            report["info"][name] = "fail: shape mismatch [G]{} vs [D]{}".format(
                gol.shape, data.shape
            )
            continue
        if gol.dtype != data.dtype:
            report["info"][name] = "fail: dtype mismatch [G]{} vs [D]{}".format(
                gol.dtype, data.dtype
            )
            continue
        if gol.dtype.name in ("int32", "int64"):
            passed = np.abs(gol - data), max() == 0
            _add_report(name, gol, data, passed)
            continue
        try:
            np.testing.assert_allclose(gol, data, rtol=rtol, atol=atol, verbose=False)
            _add_report(name, gol, data, True)
        except:  # pylint: disable=bare-except
            _add_report(name, gol, data, False)
    return report


def get_version(framework: str) -> List[int]:
    """Get the version list of framework.

    Parameters
    ----------
    framework: string
        Should be from MSCFramework.

    Returns
    -------
    version: list<int>
        The version in <major,minor,patch>.
    """

    try:
        if framework in (MSCFramework.MSC, MSCFramework.TVM):
            raw_version = tvm.__version__
        elif framework == MSCFramework.TORCH:
            import torch  # pylint: disable=import-outside-toplevel

            raw_version = torch.__version__
        elif framework == MSCFramework.TENSORFLOW:
            import tensorflow  # pylint: disable=import-outside-toplevel

            raw_version = tensorflow.__version__
        if framework == MSCFramework.TENSORRT:
            raw_version = ".".join(
                [str(v) for v in tvm.get_global_func("relax.get_tensorrt_version")()]
            )
        else:
            raw_version = "1.0.0"
    except:  # pylint: disable=bare-except
        raw_version = "1.0.0"
    version = parse(raw_version or "1.0.0")
    return [version.major, version.minor, version.micro]


def compare_version(given_version: List[int], target_version: List[int]) -> int:
    """Compare version

    Parameters
    ----------
    given_version: list<int>
        The version in <major,minor,patch>.

    target_version: list<int>
        The version in <major,minor,patch>.

    Returns
    -------
    compare_res: int
        The compare result: 0 for same version, 1 for greater version, -1 for less version
    """

    return int(_ffi_api.CompareVersion(given_version, target_version))
