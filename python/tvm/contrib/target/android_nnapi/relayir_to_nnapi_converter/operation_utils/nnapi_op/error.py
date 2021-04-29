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
# pylint: disable=invalid-name,wildcard-import,unused-wildcard-import
"""Namespace for errors encountered during checks of outputting
Android NNAPI operations
"""
from ...error import *


class AndroidNNAPICompilerBadNNAPIOperationError(AndroidNNAPICompilerError):
    """Error caused by unexpected parse result of the Relay AST

    Parameters
    ----------
    msg: str
        The error message

    """


def assert_nnapi_op_check(boolean, *msg):
    """Check for True or raise an AndroidNNAPICompilerBadNNAPIOperationError

    Parameters
    ----------
    boolean: bool
        The condition to be checked

    msg: str
        Optional error message to be raised

    """
    if not boolean:
        raise AndroidNNAPICompilerBadNNAPIOperationError(*msg)
