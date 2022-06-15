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
# pylint: disable=missing-docstring

import argparse
import os
from typing import List, Tuple

from tqdm import tqdm  # type: ignore
from tvm.meta_schedule.testing.relay_workload import get_network


# pylint: disable=too-many-branches
def _build_dataset() -> List[Tuple[str, List[int]]]:
    network_keys = []
    for name in [
        "resnet_18",
        "resnet_50",
        "mobilenet_v2",
        "mobilenet_v3",
        "wide_resnet_50",
        "resnext_50",
        "densenet_121",
        "vgg_16",
    ]:
        for batch_size in [1, 4, 8]:
            for image_size in [224, 240, 256]:
                network_keys.append((name, [batch_size, 3, image_size, image_size]))
    # inception-v3
    for name in ["inception_v3"]:
        for batch_size in [1, 2, 4]:
            for image_size in [299]:
                network_keys.append((name, [batch_size, 3, image_size, image_size]))
    # resnet3d
    for name in ["resnet3d_18"]:
        for batch_size in [1, 2, 4]:
            for image_size in [112, 128, 144]:
                network_keys.append((name, [batch_size, 3, image_size, image_size, 16]))
    # bert
    for name in ["bert_tiny", "bert_base", "bert_medium", "bert_large"]:
        for batch_size in [1, 2, 4]:
            for seq_length in [64, 128, 256]:
                network_keys.append((name, [batch_size, seq_length]))
    # dcgan
    for name in ["dcgan"]:
        for batch_size in [1, 4, 8]:
            for image_size in [64]:
                network_keys.append((name, [batch_size, 3, image_size, image_size]))
    return network_keys


def main():
    model_cache_dir = args.model_cache_dir
    try:
        os.makedirs(model_cache_dir, exist_ok=True)
    except OSError:
        print(f"Directory {model_cache_dir} cannot be created successfully.")
    keys = _build_dataset()
    for name, input_shape in tqdm(keys):
        get_network(name=name, input_shape=input_shape, cache_dir=model_cache_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # pylint: disable=invalid-name
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        help="Please provide the full path to the model cache dir.",
    )
    args = parser.parse_args()  # pylint: disable=invalid-name
    main()
