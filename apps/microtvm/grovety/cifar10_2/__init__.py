

def open_model(target):
    import pathlib, os
    import tvm.relay as relay
    import tvm
    import pickle
    import tvm

    current_dir = pathlib.Path(os.path.dirname(__file__)).resolve()

    tmp = pickle.load(open(current_dir / 'cifar10_cnn.params', 'rb'))
    params = dict((key, tvm.nd.array(value, tvm.context(str(target), 0))) for key, value in tmp.items())
    relay_mod = pickle.load(open(current_dir / "cifar10_cnn.mod", 'rb'))

    input = ("input_input", (1, 32, 32, 1), "uint8")
    output = ("Identity", (1, 3), "float32")

    return (relay_mod, params, input, output)


def get_data():
    import pickle
    import os, pathlib
    import numpy as np

    current_dir = pathlib.Path(os.path.dirname(__file__)).resolve()
    with open(current_dir / 'data/batches.meta', 'rb') as file:
        label_names = pickle.load(file, encoding='bytes')

    with open(current_dir / 'data/test_batch', 'rb') as file:
        test_batch = pickle.load(file, encoding='bytes')

    dataset = []
    i = 0
    while len(dataset) < 10:
        label = test_batch[b'labels'][i]
        label_str = label_names[b'label_names'][label].decode('UTF-8')
        data = test_batch[b'data'][i]

        i += 1
        if label_str not in ['cat', 'dog', 'frog']:
            continue

        data = data.reshape(3, 32, 32).transpose(1, 2, 0)
        # convert RGB to grayscale
        gray = 0.07 * data[:,:,2] + 0.72 * data[:,:,1] + 0.21 * data[:,:,0]
        data = gray.astype(np.uint8)

        dataset.append((label_str, data))

    return dataset