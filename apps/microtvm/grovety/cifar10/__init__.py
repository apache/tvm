

def open_model():
    import pathlib, os
    import tflite
    import tvm.relay as relay

    current_dir = pathlib.Path(os.path.dirname(__file__)).resolve()
    model_path = current_dir / "cifar_quant_8bit.tflite"

    with open(model_path, "rb") as file:
        tflite_model_buf = file.read()
        try:
            tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
        except AttributeError:
            import tflite.Model
            tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    input = ("input_input", (1, 32, 32, 1), "uint8")
    output = ("Identity", (1, 3), "float32")

    relay_mod, params = relay.frontend.from_tflite(tflite_model)
    return (relay_mod, params, input, output)


def get_data():
    import numpy as np
    import pickle
    import os, pathlib

    current_dir = pathlib.Path(os.path.dirname(__file__)).resolve()
    with open(current_dir / 'data/batches.meta', 'rb') as file:
        label_names = pickle.load(file, encoding='bytes')

    with open(current_dir / 'data/test_batch', 'rb') as file:
        test_batch = pickle.load(file, encoding='bytes')

    dataset = []
    i = 0
    while len(dataset) < 30:
        label = test_batch[b'labels'][i]
        label_str = label_names[b'label_names'][label].decode('UTF-8')
        data = test_batch[b'data'][i]
        i += 1
        if label_str not in ['cat', 'dog', 'frog']:
            continue
        dataset.append((label_str, data))

    return dataset