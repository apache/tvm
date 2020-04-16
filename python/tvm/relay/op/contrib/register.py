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
"""Register utilities for external codegen."""
_PATTERN_TABLES = {}


def register_pattern_table(compiler, table=None):
    """Register a pattern table for an external compiler.

    Pattern tables are used to create composite functions.
    See the MergeComposite pass.

    Parameters
    ----------
    compiler : str
        The name of compiler

    table : function, optional
        A function that returns the pattern table

    Returns
    -------
    fregister : function
        Register function if value is not specified.
    """
    def _register(t):
        """internal register function"""
        _PATTERN_TABLES[compiler] = t()
        return t
    return _register(table) if table is not None else _register


def get_pattern_table(compiler):
    """Get the pattern table associated with a compiler (if it's registered)."""
    return _PATTERN_TABLES[compiler] if compiler in _PATTERN_TABLES else None
