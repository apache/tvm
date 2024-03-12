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

"""
Wrap pytorch model with quantizer.
This example shows how to run PTQ, QAT, PTQ with distill...
Reference for MSC:
https://discuss.tvm.apache.org/t/rfc-unity-msc-introduction-to-multi-system-compiler/15251/5

This example use resnet50 from https://github.com/huyvnphan/PyTorch_CIFAR10/tree/master,
please download pt file and copy to args.checkpoint before run example
"""

import argparse
import torch
import torch.optim as optim

from tvm.contrib.msc.pipeline import TorchWrapper
from tvm.contrib.msc.core.tools import ToolType
from tvm.contrib.msc.core.utils.message import MSCStage
from _resnet import resnet50
from utils import *

parser = argparse.ArgumentParser(description="MSC train && eval example")
parser.add_argument(
    "--dataset",
    type=str,
    default="/tmp/msc_dataset",
    help="The folder saving training and testing datas",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="/tmp/msc_models",
    help="The folder saving training and testing datas",
)
parser.add_argument("--compile_type", type=str, default="tvm", help="The compile type of model")
parser.add_argument("--prune", action="store_true", help="Whether to use pruner")
parser.add_argument("--quantize", action="store_true", help="Whether to use quantizer")
parser.add_argument("--distill", action="store_true", help="Whether to use distiller for tool")
parser.add_argument("--gym", action="store_true", help="Whether to use gym for tool")
parser.add_argument("--test_batch", type=int, default=1, help="The batch size for test")
parser.add_argument("--test_iter", type=int, default=100, help="The iter for test")
parser.add_argument("--calibrate_iter", type=int, default=100, help="The iter for calibration")
parser.add_argument("--train_batch", type=int, default=32, help="The batch size for train")
parser.add_argument("--train_iter", type=int, default=200, help="The iter for train")
parser.add_argument("--train_epoch", type=int, default=5, help="The epoch for train")
args = parser.parse_args()


def get_config(calib_loader, train_loader):
    tools, dataset = [], {MSCStage.PREPARE: {"loader": calib_loader}}
    if args.prune:
        config = {"gym_configs": ["default"]} if args.gym else "default"
        tools.append((ToolType.PRUNER, config))
    if args.quantize:
        config = {"gym_configs": ["default"]} if args.gym else "default"
        tools.append((ToolType.QUANTIZER, config))
    if args.distill:
        config = {
            "options": {
                "optimizer": "adam",
                "opt_config": {"lr": 0.00000001, "weight_decay": 0.08},
            }
        }
        tools.append((ToolType.DISTILLER, config))
        dataset[MSCStage.DISTILL] = {"loader": train_loader}
    return TorchWrapper.create_config(
        inputs=[("input", [args.test_batch, 3, 32, 32], "float32")],
        outputs=["output"],
        compile_type=args.compile_type,
        dataset=dataset,
        tools=tools,
        skip_config={"all": "check"},
        verbose="info",
    )


if __name__ == "__main__":
    trainloader, testloader = get_dataloaders(args.dataset, args.train_batch, args.test_batch)

    def _get_calib_datas():
        for i, (inputs, _) in enumerate(testloader, 0):
            if i >= args.calibrate_iter > 0:
                break
            yield {"input": inputs}

    def _get_train_datas():
        for i, (inputs, _) in enumerate(trainloader, 0):
            if i >= args.train_iter > 0:
                break
            yield {"input": inputs}

    model = resnet50(pretrained=args.checkpoint)
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda:0"))

    acc = eval_model(model, testloader, max_iter=args.test_iter)
    print("Baseline acc: " + str(acc))

    model = TorchWrapper(model, get_config(_get_calib_datas, _get_train_datas))

    # optimize the model with tool
    model.optimize()
    acc = eval_model(model, testloader, max_iter=args.test_iter)
    print("Optimized acc: " + str(acc))

    # train the model with tool
    optimizer = optim.Adam(model.parameters(), lr=0.0000001, weight_decay=0.08)
    for ep in range(args.train_epoch):
        train_model(model, trainloader, optimizer, max_iter=args.train_iter)
        acc = eval_model(model, testloader, max_iter=args.test_iter)
        print("Train[{}] acc: {}".format(ep, acc))

    # compile the model
    model.compile()
    acc = eval_model(model, testloader, max_iter=args.test_iter)
    print("Compiled acc: " + str(acc))
