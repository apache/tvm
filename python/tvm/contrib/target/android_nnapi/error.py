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
"""Implements the errors and assertions function for Android NNAPI Compiler."""


class AndroidNNAPICompilerError(RuntimeError):
    """Android NNAPI compiler error base class.

    Parameters
    ----------
    msg: str
        The error message.
    """


class AndroidNNAPICompilerIncompatibleError(AndroidNNAPICompilerError):
    """Error caused by parsing unsupported Relay AST.

    Parameters
    ----------
    msg: str
        The error message.
    """


def assert_anc_compatibility(boolean, *msg):
    """Check for True or raise an AndroidNNAPICompilerIncompatibleError.

    Parameters
    ----------
    boolean: bool
        The checking condition.

    msg: str
        Optional string message to be raised.
    """
    if not boolean:
        raise AndroidNNAPICompilerIncompatibleError(*msg)
