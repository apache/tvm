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
# pylint: disable=no-else-return, unidiomatic-typecheck, invalid-name
"""The type nodes of the Relay language."""
from enum import IntEnum


class Feature(IntEnum):
    """The features a program might contain."""

    fVar = 0
    fGlobalVar = 1
    fConstant = 2
    fTuple = 3
    fTupleGetItem = 4
    fFunction = 5
    fOp = 6
    fCall = 7
    fLet = 8
    fIf = 9
    fRefCreate = 10
    fRefRead = 11
    fRefWrite = 12
    fConstructor = 13
    fMatch = 14
    """ Whether any non-atom fragment of the program is shared, making the program a graph. """
    fGraph = 15
    """ Whether there is local fixpoint in the program. """
    fLetRec = 16
