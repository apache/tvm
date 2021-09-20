def open_model():
    import pathlib, os
    import tflite
    import tvm.relay as relay

    current_dir = pathlib.Path(os.path.dirname(__file__)).resolve()
    model_path = current_dir / "sine_model.tflite"

    tflite_model_buf = open(model_path, "rb").read()
    try:
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    input = ("dense_4_input", (1, 1), "float32")
    output = ("Identity", (1, 1), "float32")

    relay_mod, params = relay.frontend.from_tflite(tflite_model)
    return (relay_mod, params, input, output)


def get_data():
    import numpy as np

    return [(str(e), e) for e in np.arange(0, 2, 0.1)]
