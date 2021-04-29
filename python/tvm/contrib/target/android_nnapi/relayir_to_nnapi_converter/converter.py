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
"""Converts a Relay IR Function into Android NNAPI C++ class
"""
import copy
import tvm
from . import transform
from . import json_to_nnapi
from .function_to_json_converter import FunctionToJsonConverter


class Converter:
    """Converts a Relay IR Function into Android NNAPI C++ class

    Parameters
    ----------------------
    options: dict
        The converter option dict

    """

    DEFAULT_OPTIONS = {
        "class": {
            # The base_path option is here for loading weights from external storage directly
            # However, the feature is disabled for now due to its complexity to setup
            "base_path": "/sdcard/r2n/AnnGraph/",
            "self": {
                "name": "AnnGraph",
            },
        },
        "target": {
            "api_level": 29,
        },
    }

    def __init__(self, options):
        self._options = self._expand_options(options)

    def convert(self, func):
        """Converts a Relay IR Function into Android NNAPI C++ class source code
        Parameters
        ----------
        func: tvm.relay.Function
            The Relay IR Function to be converted

        Returns
        -------
        code: str
            The C++ class source code describing func in Android NNAPI

        """
        assert isinstance(func, tvm.relay.Function)
        func = transform.FixIllegalPatternForNnapi()(func)

        mod = tvm.IRModule({"main": func})
        export_obj = FunctionToJsonConverter(self._options)(mod["main"])

        ret = json_to_nnapi.convert(
            export_obj=export_obj.asjson(),
            options={
                "class": {
                    "base_path": self._options["class"]["base_path"],
                    "name": self._options["class"]["self"]["name"],
                },
            },
        )
        return ret

    @classmethod
    def _expand_options(cls, options):
        ret = copy.deepcopy(options)

        def _recursive_merge(cur_opts, def_opts):
            for k, v in def_opts.items():
                if k in cur_opts:
                    if isinstance(v, dict):
                        assert isinstance(cur_opts[k], dict)
                        _recursive_merge(cur_opts[k], v)
                    else:
                        # type(cur_opts[k]) should be a basic type
                        assert isinstance(cur_opts[k], (float, int, str))
                else:  # option k does not exist in current options, so copy from default options
                    cur_opts[k] = copy.deepcopy(v)

        _recursive_merge(ret, cls.DEFAULT_OPTIONS)

        return ret
