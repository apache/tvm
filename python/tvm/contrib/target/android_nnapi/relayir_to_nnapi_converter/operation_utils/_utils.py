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
"""Utilities for converting tvm.relay.Call to Android NNAPI Operations."""


def name_args(args, arg_names):
    """Put arguments into dict for convenient lookup.

    Parameters
    ----------
    args: array of relay.Expr
        args of relay.Call.

    arg_names: array of string
        names of args.

    Returns
    -------
    args_map: dict of string to relay.Expr
        named args dict.
    """
    assert len(args) == len(arg_names)
    return dict(zip(arg_names, args))
