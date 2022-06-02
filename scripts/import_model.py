'''
Import common NN workloads and save as pickle files.
'''

import os
import pickle
from typing import List, Tuple
from tqdm import tqdm

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


if __name__ == '__main__':
    dataset_dir = os.path.join(os.path.dirname(os.getcwd()), 'dataset')
    relay_ir_dir = os.path.join(dataset_dir, 'relay_ir')
    params_dir = os.path.join(dataset_dir, 'params')
    os.makedirs(relay_ir_dir, exist_ok=True)
    os.makedirs(params_dir, exist_ok=True)

    keys = _build_dataset()
    for n, s in tqdm(keys):
        mod, params, _ = get_network(name=n, input_shape=s)
        mod_filename = n + '_' + '_'.join(str(i) for i in s) + '_relay_ir.pickle'
        with open(os.path.join(relay_ir_dir, mod_filename), 'wb') as f:
            pickle.dump(mod, f)
        params_filename = n + '_' + '_'.join(str(i) for i in s) + '_params.pickle'
        with open(os.path.join(params_dir, params_filename), 'wb') as f:
            pickle.dump(params, f)
