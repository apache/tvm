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
"""Converts (codegen) a JSON object to Android NNAPI source code
"""
import copy
from .stages import STAGES


DEFAULT_OPTIONS = {
    "class": {
        "base_path": "/sdcard/nnapi_result",
        "name": "AnnGraph",
    },
    "model": {
        "name": "model",
    },
    "compilation": {
        "name": "compilation",
    },
    "execution": {
        "name": "run",
        "end_event_name": "run_end",
    },
}


def convert(export_obj, options={}):  # pylint: disable=dangerous-default-value
    """Convert export_obj to NNAPI codes

    Parameters
    ----------
    export_obj: dict
        The json representation of a NNAPI model.

    options["class"]["base_path"]: str
        The base path of file accesses. Defaults to "/sdcard/nnapi_result".

    options["class"]["name"]: str
        The name of the generated C++ class wrapping around NNAPI codes. Defaults to "AnnGraph".

    options["model"]["name"]: str
        The name of the `ANeuralNetworksModel*` created. Defaults to "model".

    options["compilation"]["name"]: str
        The name of the `ANeuralNetworksCompilation*` created. Defaults to "compilation".

    options["execution"]["name"]: str
        The name of the `ANeuralNetworksExecution*` created. Defaults to "run".

    options["execution"]["end_event_name"]: str
        The name of the `ANeuralNetworksEvent*` used to wait for execution completion.
        Defaults to "run_end".

    Returns
    -------
    code: str
        The generated code
    """
    lines = {
        "tmp": {
            "model_creation": [],
            "set_execution_io": [],
            "wrapper_class": [],
        },
        "result": "",
    }
    options = _set_options(options)
    _export_obj = copy.deepcopy(export_obj)

    for s in STAGES:
        lines, _export_obj = s(lines, _export_obj, options)

    return lines["result"]


def _set_options(options):
    """Set options

    Parameters
    ----------
    options: dict
        The options to be set.

    Returns
    -------
    options: dict
        The updated options.
    """

    def _recursive_merge(cur_opts, def_opts):
        for k, v in def_opts.items():
            if k in cur_opts:
                if isinstance(v, dict):
                    assert isinstance(cur_opts[k], dict)
                    _recursive_merge(cur_opts[k], v)
                else:
                    assert isinstance(cur_opts[k], (float, int, str))
            else:
                cur_opts[k] = copy.deepcopy(v)

    _recursive_merge(options, DEFAULT_OPTIONS)

    return options
