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
"""Android NNAPI Operand-related helper methods on ExportObject."""


class Operand:
    """Android NNAPI Operand-related helper methods on ExportObject."""

    def __init__(self, export_obj):
        self._export_obj = export_obj

    def get_dtype(self, idx):
        """Get operand dtype.

        Parameters
        ----------
        idx: int
            operand to be queried.

        Returns
        -------
        dtype: str
            dtype of the queried operand.
        """
        return self._export_obj["types"][self._export_obj["operands"][idx]["type"]]["type"]

    def get_shape(self, idx):
        """Get operand shape.

        Parameters
        ----------
        idx: int
            operand to be queried.

        Returns
        -------
        shape: tuple of int or None
            shape of the queried operand. None if operand has no shape.
        """
        return self._export_obj["types"][self._export_obj["operands"][idx]["type"]].get(
            "shape", None
        )

    def get_rank(self, idx):
        """Get operand rank.

        Parameters
        ----------
        idx: int
            operand to be queried.

        Returns
        -------
        rank: int
            rank of the queried operand.
        """
        shape = self.get_shape(idx)
        if shape is None:
            return 0
        return len(shape)

    def get_value(self, idx):
        """Get operand value.

        Parameters
        ----------
        idx: int
            operand to be queried.

        Returns
        -------
        value:
            value of the queried operand. None if there's no value.
        """
        value_dict = self._export_obj["operands"][idx].get("value", None)
        if value_dict is None:
            return None

        if value_dict["type"] == "constant_idx":
            return self._export_obj["constants"][value_dict["value"]]["value"]
        assert value_dict["type"] == "memory_ptr"
        return value_dict["value"]

    def get_constant(self, idx):
        """Get operand constant.

        Parameters
        ----------
        idx: int
            operand to be queried.

        Returns
        -------
        obj: dict
            constant dict of the queried operand. None if there's no value.
        """
        value_dict = self._export_obj["operands"][idx].get("value", None)
        if value_dict is None or value_dict["type"] != "constant_idx":
            return None
        return self._export_obj["constants"][value_dict["value"]]

    def is_fuse_code(self, idx):
        """Check whether the operand pointed by idx is a FuseCode

        Parameters
        ----------
        idx: int
            the index of the queried operand.

        Returns
        -------
        b: bool
            the queried operand is a FuseCode or not.
        """
        dtype = self.get_dtype(idx)
        if dtype != "INT32":
            return False
        shape = self.get_shape(idx)
        if shape is not None:
            return False
        value = self.get_value(idx)
        return value in {
            "ANEURALNETWORKS_FUSED_NONE",
            "ANEURALNETWORKS_FUSED_RELU",
            "ANEURALNETWORKS_FUSED_RELU1",
            "ANEURALNETWORKS_FUSED_RELU6",
        }
