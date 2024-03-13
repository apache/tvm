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
# pylint: disable=redefined-builtin, invalid-name
"""Loss functions library for relax."""

from typing import Optional, Union

# isort: off
from typing_extensions import Literal

# isort: on

from ..block_builder import BlockBuilder
from ..expr import Expr, Var, Function, StructInfo

from ..op import abs, sum, mean, subtract, multiply, reshape, argmax
from ..op.nn import log_softmax, nll_loss


def _create_param_var(param: Union[Var, StructInfo], param_name: str) -> Var:
    """If param is a StructInfo, create a Var with the given StructInfo and name.

    If param is a Var, create a Var with the same StructInfo and name as the given param Var."""
    if isinstance(param, StructInfo):
        param = Var(param_name, param)
    if not isinstance(param, Var):
        raise TypeError("The type of param should be Var or StructInfo, but got " + type(param))
    return Var(param.name_hint, param.struct_info)


class Loss:
    r"""Base class of all loss.

    Generally, loss function will take one or more **input parameters** (that is outputs of
    the backbone of a model), one or more **target parameters**, and generate a scalar value
    denoting the loss.

    You can use `relax.transform.AppendLoss` to append the loss function to a one-dataflowblock
    backbone function in a IRModule. That will generate a one-dataflowblock function accepting
    instances and targets, and then returning the loss.

    Most loss functions involve a reduction of losses from all instances in a batch. We use
    `reduction` parameter to denote the reduction method. Possible reduction methods include
    `"mean"`, `"sum"` and `"none"`.

    Parameters
    ----------
    loss_name : str
        The name of the loss function. Should be provided when calling `super().__init__` in
        constructor functions of subclasses.

    num_backbone_outputs : int
        The number of `prediction_outputs` of the backbone function, alos the number of the
        backbone_prediction_outputs of the loss function. See `relax.transform.AppendLoss`.

        Should be provided when calling `super().__init__` in constructor functions of subclasses.

        For example, `CrossEntropyLoss` requires one backbone prediction output; `MarginRankingLoss`
        requires two backbone prediction outputs.

    reduction : Literal["mean", "sum", "none"]
        The reduction method to apply to output. Can be "mean", "sum" or "none".

        none : no reduction will be applied,
        mean : the sum of the output will be divided by the batch_size,
        sum : the output will be summed.
    """

    _valid_reductions = ["mean", "sum", "none"]

    def __init__(
        self,
        loss_name: str,
        num_backbone_outputs: int,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> None:
        self._loss_name = loss_name
        self._reduction = reduction
        self._num_backbone_outputs = num_backbone_outputs

        if self._reduction not in self._valid_reductions:
            raise ValueError("Reduction can only be one of these values: ", self._valid_reductions)

    @property
    def num_backbone_outputs(self) -> int:
        """Get the number of number of the outputs of the backbone function."""
        return self._num_backbone_outputs

    def _with_reduction(self, expr: Expr) -> Expr:
        """Add a reduction to the final loss.

        Parameters
        ----------
        expr : Expr
            The loss expr.

        Returns
        -------
        ret : Expr
            The reduced result.
        """
        if self._reduction == "sum":
            expr = sum(expr)
        elif self._reduction == "mean":
            expr = mean(expr)
        elif self._reduction != "none":
            raise ValueError("Reduction can only be one of these values: ", self._valid_reductions)
        return expr


class L1Loss(Loss):
    r"""Mean element-wise absolute value difference.

    Parameters
    ----------
    reduction : Literal["mean", "sum", "none"]
        The reduction method to apply to output. Can be "mean", "sum" or "none".

        none : no reduction will be applied,
        mean : the sum of the output will be divided by the batch_size,
        sum : the output will be summed.
    """

    def __init__(self, reduction: Literal["mean", "sum", "none"] = "mean") -> None:
        super().__init__("l1_loss", 1, reduction)

    def __call__(
        self,
        predictions: Union[Var, StructInfo],
        targets: Union[Var, StructInfo],
    ) -> Function:
        """Get the relax function of L1Loss. If the parameters are
        struct info, it will create corresponding variables.

        Parameters
        ----------
        predictions : Union[Var, StructInfo]
            The predictions of the model in the calculation of loss.
        targets : Union[Var, StructInfo]
            The ground truth in the calculation of loss.

        Returns
        -------
        The relax function of L1Loss with the loss name as its global symbol.
        """
        bb = BlockBuilder()

        predictions = _create_param_var(predictions, "predictions")
        targets = _create_param_var(targets, "targets")

        with bb.function(self._loss_name, [predictions, targets]):
            with bb.dataflow():
                lv = abs(subtract(predictions, targets))
                loss = bb.emit_output(self._with_reduction(lv))
            bb.emit_func_output(loss)

        return bb.get()[self._loss_name]


class MSELoss(Loss):
    r"""Measures the element-wise mean squared error.

    Parameters
    ----------
    reduction : Literal["mean", "sum", "none"]
        The reduction method to apply to output. Can be "mean", "sum" or "none".

        none : no reduction will be applied,
        mean : the sum of the output will be divided by the batch_size,
        sum : the output will be summed.
    """

    def __init__(self, reduction: Literal["mean", "sum", "none"] = "mean") -> None:
        super().__init__("mse_loss", 1, reduction)

    def __call__(
        self,
        predictions: Union[Var, StructInfo],
        targets: Union[Var, StructInfo],
    ) -> Function:
        """Get the relax function of MSELoss. If the parameters are
        struct info, it will create corresponding variables.

        Parameters
        ----------
        predictions : Union[Var, StructInfo]
            The predictions of the model in the calculation of loss.
        targets : Union[Var, StructInfo]
            The ground truth in the calculation of loss.

        Returns
        -------
        The relax function of MSELoss with the loss name as its global symbol.
        """
        bb = BlockBuilder()

        predictions = _create_param_var(predictions, "predictions")
        targets = _create_param_var(targets, "targets")

        with bb.function(self._loss_name, [predictions, targets]):
            with bb.dataflow():
                lv = subtract(predictions, targets)
                lv = multiply(lv, lv)
                loss = bb.emit_output(self._with_reduction(lv))
            bb.emit_func_output(loss)

        return bb.get()[self._loss_name]


class CrossEntropyLoss(Loss):
    r"""CrossEntropyLoss. It is a combination of a log_softmax computation and a nll_loss.

    Parameters
    ----------
    reduction : Literal["mean", "sum", "none"]
        The reduction method to apply to output. Can be "mean", "sum" or "none".

        none : no reduction will be applied,
        mean : the sum of the output will be divided by the batch_size,
        sum : the output will be summed.

    ignore_index : int
        Specifies a target value that is ignored and does not contribute to the input gradient.
    """

    ignore_index: int

    def __init__(
        self,
        reduction: Literal["mean", "sum", "none"] = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__("cross_entropy_loss", 1, reduction)
        self.ignore_index = ignore_index

    def __call__(
        self,
        predictions: Union[Var, StructInfo],
        targets: Union[Var, StructInfo],
        weights: Optional[Union[Var, StructInfo]] = None,
    ) -> Function:
        """Get the relax function of CrossEntropyLoss. If the parameters are
        struct info, it will create corresponding variables.

        Parameters
        ----------
        predictions : Union[Var, StructInfo]
            The predictions of the model in the calculation of loss.

        targets : Union[Var, StructInfo]
            The ground truth in the calculation of loss.

        weights : Optional[Union[Var, StructInfo]]
            a manual rescaling weight given to each class. It has to be a Tensor of size C.

        Returns
        -------
        The relax function of CrossEntropyLoss with the loss name as its global symbol.
        """
        bb = BlockBuilder()

        predictions = _create_param_var(predictions, "predictions")
        targets = _create_param_var(targets, "targets")

        arg_list = [predictions, targets]
        if weights:
            weights = _create_param_var(weights, "weights")
            arg_list.append(weights)

        with bb.function(self._loss_name, arg_list):
            with bb.dataflow():
                logits = bb.emit(log_softmax(predictions))
                loss = bb.emit_output(
                    nll_loss(logits, targets, weights, self._reduction, self.ignore_index)
                )
            bb.emit_func_output(loss)

        return bb.get()[self._loss_name]


class CategoricalCrossEntropyLoss(Loss):
    r"""CategoricalCrossEntropyLoss.
    It is a combination of a converting one-hot target vector to a label,
    a log_softmax computation and a nll_loss.

    Parameters
    ----------
    reduction : Literal["mean", "sum", "none"]
        The reduction method to apply to output. Can be "mean", "sum" or "none".

        none : no reduction will be applied,
        mean : the sum of the output will be divided by the batch_size,
        sum : the output will be summed.

    ignore_index : int
        Specifies a target value that is ignored and does not contribute to the input gradient.
    """

    ignore_index: int

    def __init__(
        self,
        reduction: Literal["mean", "sum", "none"] = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__("categorical_cross_entropy_loss", 1, reduction)
        self.ignore_index = ignore_index

    def __call__(
        self,
        predictions: Union[Var, StructInfo],
        targets: Union[Var, StructInfo],
        weights: Optional[Union[Var, StructInfo]] = None,
    ) -> Function:
        """Get the relax function of CategoricalCrossEntropyLoss. If the parameters are
        struct info, it will create corresponding variables.

        Parameters
        ----------
        predictions : Union[Var, StructInfo]
            The predictions of the model in the calculation of loss.

        targets : Union[Var, StructInfo]
            The ground truth in the calculation of loss.

        weights : Optional[Union[Var, StructInfo]]
            a manual rescaling weight given to each class. It has to be a Tensor of size C.

        Returns
        -------
        The relax function of CategoricalCrossEntropyLoss with the loss name as its global symbol.
        """

        if not "int" in targets.dtype:
            raise TypeError(
                f"Dtype of targets expected to be int/uint. \
                  However, the dtype of targets is {targets.dtype}"
            )

        bb = BlockBuilder()

        predictions = _create_param_var(predictions, "predictions")
        targets = _create_param_var(targets, "targets")

        arg_list = [predictions, targets]
        if weights:
            weights = _create_param_var(weights, "weights")
            arg_list.append(weights)

        # In the case of ignore_index >= 0,
        # the nll_loss function is used to handle the ignore index.
        # In other cases where ignore_index is not needed, just use the simpe product.
        with bb.function(self._loss_name, arg_list):
            with bb.dataflow():
                logits = bb.emit(log_softmax(predictions))
                if self.ignore_index >= 0:
                    targets = bb.emit(
                        reshape(argmax(targets, axis=1), shape=(targets.struct_info.shape[0],))
                    )
                    loss = bb.emit_output(
                        nll_loss(logits, targets, weights, self._reduction, self.ignore_index)
                    )
                else:
                    lv = bb.emit(-logits * targets.astype("float32"))
                    if weights:
                        lv = bb.emit(lv * weights)
                    loss = bb.emit_output(self._with_reduction(lv))
            bb.emit_func_output(loss)

        return bb.get()[self._loss_name]
