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
import pytest
import tvm.testing
import numpy as np

from tvm import relax, TVMError
from tvm.ir.base import assert_structural_equal
from tvm.script.parser import ir as I, relax as R, tir as T
from tvm.relax.training import SetupTrainer
from tvm.relax.training.optimizer import SGD
from tvm.relax.training.loss import MSELoss
from tvm.relax.training.trainer import Trainer


def _get_backbone():
    @I.ir_module
    class MLP:
        @R.function
        def predict(
            w0: R.Tensor((10, 5), "float32"),
            b0: R.Tensor((5,), "float32"),
            x: R.Tensor((1, 10), "float32"),
        ):
            with R.dataflow():
                lv0 = R.matmul(x, w0)
                lv1 = R.add(lv0, b0)
                out = R.nn.relu(lv1)
                R.output(out)
            return out

    return MLP, 2


def _make_dataset():
    N = 100
    return [
        [
            np.random.uniform(size=(1, 10)).astype(np.float32),
            np.array([[0, 0, 1, 0, 0]]).astype(np.float32),
        ]
        for _ in range(N)
    ]


@tvm.testing.parametrize_targets("llvm")
def test_execute(target, dev):
    backbone, params_num = _get_backbone()
    pred_sinfo = relax.TensorStructInfo((1, 5), "float32")

    setup_trainer = SetupTrainer(
        MSELoss(reduction="sum"),
        SGD(0.001),
        [pred_sinfo, pred_sinfo],
    )

    trainer = Trainer(backbone, params_num, setup_trainer)
    trainer.build(target, dev)
    trainer.xaiver_uniform_init_params()

    dataset = _make_dataset()
    last_loss = np.inf
    for epoch in range(5):
        for i, data in enumerate(dataset):
            loss = trainer.update_params(data[0], data[1])
    trainer.predict(dataset[0][0])


@tvm.testing.parametrize_targets("llvm")
def test_load_export_params(target, dev):
    backbone, params_num = _get_backbone()
    pred_sinfo = relax.TensorStructInfo((1, 5), "float32")

    setup_trainer = SetupTrainer(
        MSELoss(reduction="sum"),
        SGD(0.001),
        [pred_sinfo, pred_sinfo],
    )

    trainer = Trainer(backbone, params_num, setup_trainer)
    trainer.build(target, dev)
    trainer.xaiver_uniform_init_params()

    dataset = _make_dataset()
    for i, data in enumerate(dataset):
        trainer.update_params(data[0], data[1])

    param_dict = trainer.export_params()
    assert "w0" in param_dict
    assert "b0" in param_dict

    trainer1 = Trainer(backbone, params_num, setup_trainer)
    trainer1.build(target, dev)
    trainer1.load_params(param_dict)

    x_sample = dataset[np.random.randint(len(dataset))][0]
    tvm.testing.assert_allclose(
        trainer.predict(x_sample).numpy(), trainer1.predict(x_sample).numpy()
    )


def test_setting_error():
    backbone, params_num = _get_backbone()
    pred_sinfo = relax.TensorStructInfo((1, 5), "float32")

    setup_trainer = SetupTrainer(
        MSELoss(reduction="sum"),
        SGD(0.001),
        [pred_sinfo, pred_sinfo],
    )
    trainer = Trainer(backbone, params_num, setup_trainer)

    with pytest.raises(TVMError):
        trainer.vm
    dataset = _make_dataset()
    with pytest.raises(TVMError):
        trainer.predict(dataset[0][0])
    with pytest.raises(TVMError):
        trainer.update_params(dataset[0][0], dataset[0][1])


def test_invalid_mod():
    @I.ir_module
    class NotReturnSingleTensor:
        @R.function
        def predict(
            w0: R.Tensor((10, 5), "float32"),
            b0: R.Tensor((5,), "float32"),
            x: R.Tensor((1, 10), "float32"),
        ):
            R.func_attr({"params_num": 2})
            with R.dataflow():
                lv0 = R.matmul(x, w0)
                gv = R.add(lv0, b0)
                out = R.nn.relu(gv)
                R.output(gv, out)
            return gv, out

    pred_sinfo = relax.TensorStructInfo((1, 5), "float32")
    setup_trainer = SetupTrainer(
        MSELoss(reduction="sum"),
        SGD(0.001),
        [pred_sinfo, pred_sinfo],
    )

    with pytest.raises(ValueError):
        setup_trainer(NotReturnSingleTensor)

    @I.ir_module
    class NoParamsNumAttr:
        @R.function
        def predict(
            w0: R.Tensor((10, 5), "float32"),
            b0: R.Tensor((5,), "float32"),
            x: R.Tensor((1, 10), "float32"),
        ):
            with R.dataflow():
                lv0 = R.matmul(x, w0)
                lv1 = R.add(lv0, b0)
                out = R.nn.relu(lv1)
                R.output(out)
            return out

    with pytest.raises(TypeError):
        setup_trainer(NoParamsNumAttr)


if __name__ == "__main__":
    tvm.testing.main()
