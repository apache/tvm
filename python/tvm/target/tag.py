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
"""Target tags"""
from typing import Any, Dict, Optional
from . import _ffi_api
from .target import Target


def list_tags() -> Optional[Dict[str, Target]]:
    """Returns a dict of tags, which maps each tag name to its corresponding target.

    Returns
    -------
    tag_dict : Optional[Dict[str, Target]]
        The dict of tags mapping each tag name to its corresponding target.
        None if TVM is built in runtime-only mode.
    """
    if hasattr(_ffi_api, "TargetTagListTags"):
        return _ffi_api.TargetTagListTags()
    return None


def register_tag(name: str, config: Dict[str, Any], override: bool = False) -> Optional[Target]:
    """Add a user-defined tag into the target tag registry.

    Parameters
    ----------
    name: str
        Name of the target, e.g. "nvidia/gtx1080ti"
    config : Dict[str, Any]
        The config dict used to create the target
    override: bool
        A boolean flag indicating if overriding existing tags are allowed.
        If False and the tag has been registered already, an exception will be thrown.

    Returns
    -------
    target : Optional[Target]
        The target corresponding to the tag
        None if TVM is built in runtime-only mode.

    Examples
    --------
    .. code-block:: python

        register_tag("nvidia/gtx1080ti", config={
            "kind": "cuda",
            "arch": "sm_61",
        })
    """
    if hasattr(_ffi_api, "TargetTagAddTag"):
        return _ffi_api.TargetTagAddTag(name, config, override)
    return None


# We purposely maintain all tags in the C++ side to support pure C++ use cases,
# and the Python API is only used for fast prototyping.
register_tag(
    "nvidia/gtx1080ti",
    config={
        "kind": "cuda",
        "arch": "sm_61",
    },
)

# To check the correctness of all registered tags, the call is made in library loading time.
list_tags()
