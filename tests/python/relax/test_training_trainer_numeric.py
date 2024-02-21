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

import tvm
from tvm import relax, TVMError
from tvm.relax.training import SetupTrainer, Trainer
from tvm.relax.training.optimizer import SGD, Adam
from tvm.relax.training.loss import MSELoss
from tvm.script import ir as I, relax as R


def _get_backbone():
    @I.ir_module
    class MLP:
        I.module_attrs({"param_num": 2, "state_num": 0})

        @R.function
        def backbone(
            x: R.Tensor((1, 10), "float32"),
            w0: R.Tensor((10, 5), "float32"),
            b0: R.Tensor((5,), "float32"),
        ):
            with R.dataflow():
                lv0 = R.matmul(x, w0)
                lv1 = R.add(lv0, b0)
                out = R.nn.relu(lv1)
                R.output(out)
            return out

    return MLP


def _make_dataset():
    N = 100
    return [[np.ones((1, 10)).astype(np.float32), np.array([[0, 0, 1, 0, 0]], np.float32)]] * N


@tvm.testing.parametrize_targets("llvm")
def test_execute(target, dev):
    backbone = _get_backbone()
    pred_sinfo = relax.TensorStructInfo((1, 5), "float32")

    setup_trainer = SetupTrainer(
        MSELoss(reduction="sum"),
        Adam(0.01),
        [pred_sinfo, pred_sinfo],
    )

    train_mod = setup_trainer(backbone)
    ex = relax.build(train_mod, target)
    vm = relax.VirtualMachine(ex, dev, profile=True)

    trainer = Trainer(train_mod, vm, dev, False)
    trainer.zero_init_params()
    trainer.xaiver_uniform_init_params()

    dataset = _make_dataset()
    trainer.predict(dataset[0][0])
    trainer.update(dataset[0][0], dataset[0][1])
    trainer.profile_adjoint(dataset[0][0], dataset[0][1])


@tvm.testing.parametrize_targets("llvm")
def test_execute_numeric(target, dev):
    backbone = _get_backbone()
    pred_sinfo = relax.TensorStructInfo((1, 5), "float32")

    setup_trainer = SetupTrainer(
        MSELoss(reduction="sum"),
        SGD(0.01),
        [pred_sinfo, pred_sinfo],
    )

    train_mod = setup_trainer(backbone)
    ex = relax.build(train_mod, target)
    vm = relax.VirtualMachine(ex, dev)

    trainer = Trainer(train_mod, vm, dev, False)
    trainer.zero_init_params()

    dataset = _make_dataset()
    for _ in range(2):
        for input, label in dataset:
            loss = trainer.update(input, label)
    tvm.testing.assert_allclose(loss.numpy(), 3.1974423e-14)

    result = trainer.predict(dataset[0][0])
    result_expected = np.array([[0, 0, 0.9999998, 0, 0]], np.float32)
    tvm.testing.assert_allclose(result.numpy(), result_expected)


@tvm.testing.parametrize_targets("llvm")
def test_load_export_params(target, dev):
    backbone = _get_backbone()
    pred_sinfo = relax.TensorStructInfo((1, 5), "float32")

    setup_trainer = SetupTrainer(
        MSELoss(reduction="sum"),
        SGD(0.01),
        [pred_sinfo, pred_sinfo],
    )

    train_mod = setup_trainer(backbone)
    ex = relax.build(train_mod, target)
    vm = relax.VirtualMachine(ex, dev)

    trainer = Trainer(train_mod, vm, dev, False)
    trainer.xaiver_uniform_init_params()

    dataset = _make_dataset()
    for input, label in dataset:
        trainer.update(input, label)

    param_dict = trainer.export_params()
    assert "w0" in param_dict
    assert "b0" in param_dict

    trainer1 = Trainer(train_mod, vm, dev, False)
    trainer1.load_params(param_dict)

    x_sample = dataset[np.random.randint(len(dataset))][0]
    tvm.testing.assert_allclose(
        trainer.predict(x_sample).numpy(), trainer1.predict(x_sample).numpy()
    )


@tvm.testing.parametrize_targets("llvm")
def test_setting_error(target, dev):
    backbone = _get_backbone()
    pred_sinfo = relax.TensorStructInfo((1, 5), "float32")

    setup_trainer = SetupTrainer(
        MSELoss(reduction="sum"),
        SGD(0.01),
        [pred_sinfo, pred_sinfo],
    )

    train_mod = setup_trainer(backbone)
    ex = relax.build(train_mod, target)
    vm = relax.VirtualMachine(ex, dev)

    trainer = Trainer(train_mod, vm, dev, False)

    dataset = _make_dataset()
    # parameters are not inited
    with pytest.raises(TVMError):
        trainer.predict(dataset[0][0])
    with pytest.raises(TVMError):
        trainer.update(dataset[0][0], dataset[0][1])


if __name__ == "__main__":
    tvm.testing.main()
