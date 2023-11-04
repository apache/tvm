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

import os
import json
import copy
from typing import List, Tuple, Dict, Any
from distutils.version import LooseVersion
import numpy as np

import tvm
from .namespace import MSCFramework


class MSCArray(object):
    """MSC wrapper for array like object

    Parameters
    ----------
    data: array_like: np.ndarray| torch.Tensor| tvm.ndarray| ...
        The data object.
    """

    def __init__(self, data: Any):
        self._type, self._data = self._analysis(data)

    def __str__(self):
        return "<{}>{}".format(self._type, self.abstract())

    def _analysis(self, data: Any) -> Tuple[str, np.ndarray]:
        if isinstance(data, np.ndarray):
            return "np", data
        if isinstance(data, tvm.runtime.NDArray):
            return "tvm", data.asnumpy()
        try:
            import torch  # pylint: disable=import-outside-toplevel

            if isinstance(data, torch.Tensor):
                return "torch", data.detach().cpu().numpy()
        except:  # pylint: disable=bare-except
            pass

        raise Exception("Unkonwn data {}({})".format(data, type(data)))

    def abstract(self) -> str:
        """Get abstract describe of the data"""
        return "[S:{},D:{}] Max {:g}, Min {:g}, Avg {:g}".format(
            ";".join([str(s) for s in self._data.shape]),
            self._data.dtype.name,
            self._data.max(),
            self._data.min(),
            self._data.sum() / self._data.size,
        )

    @property
    def type(self):
        return self._type

    @property
    def data(self):
        return self._data


def cast_array(data: Any):
    """Cast array like object to np.ndarray

    Parameters
    ----------
    data: array_like: np.ndarray| torch.Tensor| tvm.ndarray| ...
        The data object.

    Returns
    -------
    output: np.ndarray
        The output as numpy array.
    """

    return MSCArray(data).data


def compare_arrays(
    golden: Dict[str, np.ndarray],
    datas: Dict[str, np.ndarray],
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> dict:
    """Compare elements in array

    Parameters
    ----------
    golden: dict<str, np.ndarray>
        The golden datas.
    datas: dict<str, np.ndarray>
        The datas to be compared.
    atol: float
        The atol for compare.
    rtol: float
        The rtol for compare.

    Returns
    -------
    report: dict
        The compare results.
    """

    assert golden.keys() == datas.keys(), "golden {} and datas {} mismatch".format(
        golden.keys(), datas.keys()
    )
    report = {"total": 0, "passed": 0, "info": {}}
    for name, gol in golden.items():
        report["total"] += 1
        data = datas[name]
        if list(gol.shape) != list(data.shape):
            report["info"][name] = "<Fail> shape mismatch [G]{} vs [D]{}".format(
                gol.shape, data.shape
            )
            continue
        if gol.dtype != data.dtype:
            report["info"][name] = "<Fail> dtype mismatch [G]{} vs [D]{}".format(
                gol.dtype, data.dtype
            )
            continue
        diff = MSCArray(gol - data)
        try:
            np.testing.assert_allclose(gol, data, rtol=rtol, atol=atol, verbose=False)
            report["info"][name] = "<Pass> diff {}".format(diff.abstract())
            report["passed"] += 1
        except:  # pylint: disable=bare-except
            report["info"][name] = "<Fail> diff {}".format(diff.abstract())
    return report


def load_dict(str_dict: str, flavor: str = "json") -> dict:
    """Load the string/file to dict.

    Parameters
    ----------
    str_dict: string
        The file_path or string object.
    flavor: str
        The flavor for load.

    Returns
    -------
    dict_obj: dict
        The loaded dict.
    """

    if isinstance(str_dict, str) and os.path.isfile(str_dict):
        with open(str_dict, "r") as f:
            dict_obj = json.load(f)
    elif isinstance(str_dict, str):
        dict_obj = json.loads(str_dict)
    assert flavor == "json", "Unexpected flavor for load_dict: " + str(flavor)
    return dict_obj


def dump_dict(dict_obj: dict, flavor: str = "dmlc") -> str:
    """Dump the config to string.

    Parameters
    ----------
    src_dict: dict
        The source dict.
    flavor: str
        The flavor for dumps.

    Returns
    -------
    str_dict: string
        The dumped string.
    """

    if not dict_obj:
        return ""
    if flavor == "dmlc":
        return json.dumps({k: int(v) if isinstance(v, bool) else v for k, v in dict_obj.items()})
    if flavor == "table":

        def _get_lines(value, indent=0):
            lines = []
            for k, v in value.items():
                if isinstance(v, dict):
                    lines.append("{}{}:".format(indent * " ", k))
                    lines.extend(_get_lines(v, indent + 2))
                elif isinstance(v, (tuple, list)) and len(str(v)) > 100:
                    lines.append("{}{}:".format(indent * " ", k))
                    lines.extend(
                        [
                            "{}<{}>{}".format((indent + 2) * " ", idx, ele)
                            for idx, ele in enumerate(v)
                        ]
                    )
                elif isinstance(v, bool):
                    lines.append("{}{}: {}".format(indent * " ", k, "true" if v else "false"))
                elif isinstance(v, np.ndarray):
                    lines.append("{}{}: {}".format(indent * " ", k, MSCArray(v).abstract()))
                else:
                    lines.append("{}{}: {}".format(indent * " ", k, v))
            return lines

        return "\n".join(_get_lines(dict_obj))
    return json.dumps(dict_obj)


def dict_equal(dict_a: dict, dict_b: dict) -> bool:
    """Check if two dicts are the same.

    Parameters
    ----------
    dict_a: dict
        The A dict.
    dict_b: dict
        The B dict.

    Returns
    -------
    equal: bool
        Whether two dicts are the same.
    """

    if not isinstance(dict_a, dict) or not isinstance(dict_b, dict):
        return False
    if dict_a.keys() != dict_b.keys():
        return False
    for k, v in dict_a.items():
        if not isinstance(v, type(dict_b[k])):
            return False
        if isinstance(v, dict) and not dict_equal(v, dict_b[k]):
            return False
        if v != dict_b[k]:
            return False
    return True


def copy_dict(dict_obj: dict) -> dict:
    """Deepcopy dict object

    Parameters
    ----------
    dict_obj: dict
        The source dict.

    Returns
    -------
    dict_obj: dict
        The copied dict.
    """

    if not dict_obj:
        return {}
    return copy.deepcopy(dict_obj)


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

    return LooseVersion(raw_version).version
