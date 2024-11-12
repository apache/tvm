import tvm
import numpy as np
import torch
from tvm import relax, topi
from tvm.script import ir as I
from tvm.script import relax as R
from tvm.script import tir as T
from tvm import dlight as dl
from inceptron.runtime.datasets.mnist import MNISTDataset
from inceptron.runtime.datasets.dataloader import DataLoader
from tvm.relax.training.trainer import Trainer
from tvm.relax.training.setup_trainer import SetupTrainer
from tvm.relax.training.loss import CrossEntropyLoss, CategoricalCrossEntropyLoss
from tvm.relax.training.optimizer import SGD


bb = relax.BlockBuilder()
n = tvm.tir.SizeVar("batch_size", "int64")  # T.int64()
x = relax.Var("x", R.Tensor((n, 784), "float32"))
fc1_weight = relax.Var("fc1_weight", R.Tensor((128, 784), "float32"))
fc1_bias = relax.Var("fc1_bias", R.Tensor((128,), "float32"))
fc2_weight = relax.Var("fc2_weight", R.Tensor((10, 128), "float32"))
fc2_bias = relax.Var("fc2_bias", R.Tensor((10,), "float32"))
with bb.function("backbone", [x, fc1_weight, fc1_bias, fc2_weight, fc2_bias]):
    with bb.dataflow():
        lv0 = bb.emit(relax.op.matmul(x, relax.op.permute_dims(fc1_weight)) + fc1_bias)
        lv1 = bb.emit(relax.op.nn.relu(lv0))
        lv2 = bb.emit(
            relax.op.matmul(lv1, relax.op.permute_dims(fc2_weight)) + fc2_bias
        )
        gv = bb.emit_output(relax.op.expand_dims(lv2, -1))
    bb.emit_func_output(gv)  # (gv, ()))

mod = bb.get()
print(mod)
mod = mod.with_attrs(
    {
        "param_num": 4,  # Number of model parameters
        "state_num": 0,  # Number of states to maintain (none in this case)
    }
)

# Load MNIST Dataset, todo: one hot encoding for labels
dataset = MNISTDataset(
    input_name="input.1",
    root=".cache/datasets",
    subset=128,
)
batch_size = 64

# Train the model on MNIST
pred_sinfo = relax.TensorStructInfo((n, 10, 1), dtype="float32")
target_sinfo = relax.TensorStructInfo((n, 1), dtype="int64")

setup_trainer = SetupTrainer(
    CrossEntropyLoss(reduction="sum"), SGD(0.0001), [pred_sinfo, target_sinfo]
)

train_mod = setup_trainer.transform_module(mod, tvm.transform.PassContext())
train_mod.show()

dev = tvm.device("cuda")
target = tvm.target.Target.from_device(dev)
with target:
    train_mod = tvm.ir.transform.Sequential(
        [
            dl.ApplyDefaultSchedule(
                dl.gpu.Matmul(),
                dl.gpu.GEMV(),
                dl.gpu.GeneralReduction(),
                dl.gpu.Fallback(),
            ),
        ]
    )(train_mod)

with tvm.transform.PassContext(opt_level=3):
    ex = relax.build(train_mod, target)
vm = relax.VirtualMachine(ex, dev)

trainer = Trainer(train_mod, vm, dev, False)
trainer.xaiver_uniform_init_params()

epochs = 200
for epoch in range(epochs):
    epoch_correct = 0
    total_samples = 0
    with torch.no_grad():
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for data in dataloader:
            inputs = data[0]["input.1"] 
            labels = data[1]["label"]
            x_nd = tvm.nd.array(inputs.reshape((-1, 784)), device=dev)
            y_nd = tvm.nd.array(labels, device=dev)
            trainer.update([x_nd], [y_nd])
            preds = trainer.predict(x_nd)
            preds = preds.asnumpy().squeeze()
            predicted_digits = np.argmax(preds, axis=1)
            correct = (predicted_digits == labels).sum()
            epoch_correct += correct
            total_samples += len(labels)
    epoch_accuracy = epoch_correct / total_samples
    print(f"Epoch [{epoch+1}/{epochs}], Accuracy: {epoch_accuracy:.4f}")


