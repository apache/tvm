

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

    return [
        ("cat", np.random.randint(0, high=255, size=(1, 32, 32, 1), dtype="uint8")),
        ("dog", np.random.randint(0, high=255, size=(1, 32, 32, 1), dtype="uint8")),
        ("frog", np.random.randint(0, high=255, size=(1, 32, 32, 1), dtype="uint8")),
        ("cat", np.random.randint(0, high=255, size=(1, 32, 32, 1), dtype="uint8")),
        ("dog", np.random.randint(0, high=255, size=(1, 32, 32, 1), dtype="uint8")),
        ("frog", np.random.randint(0, high=255, size=(1, 32, 32, 1), dtype="uint8")),
    ]
