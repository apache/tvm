
def open_f32_model():
    import pathlib, os
    import onnx
    import tvm.relay as relay

    current_dir = pathlib.Path(os.path.dirname(__file__)).resolve()
    model_path = current_dir / "mnist-8.onnx"

    onnx_model = onnx.load(model_path)
    shape = {"Input3": (1, 1, 28, 28)}
    relay_mod, params = relay.frontend.from_onnx(onnx_model, shape=shape, freeze_params=True)
    relay_mod = relay.transform.DynamicToStatic()(relay_mod)

    input = ("Input3", (1, 1, 28, 28), 'float32')
    output = ("Plus214_Output_0", (1, 10), 'float32')
    return (relay_mod, params, input, output)


def open_model():
    import pathlib, os
    import tflite
    import tvm.relay as relay

    current_dir = pathlib.Path(os.path.dirname(__file__)).resolve()
    model_path = current_dir / "mnist_model_quant.tflite"

    tflite_model_buf = open(model_path, "rb").read()
    try:
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    input = ("input_1", (1, 28, 28), "uint8")
    output = ("Identity", (1, 10), "uint8")

    relay_mod, params = relay.frontend.from_tflite(tflite_model)
    return (relay_mod, params, input, output)


def get_data():
    import pathlib, os
    import numpy as np
    from PIL import Image

    image_files = ["digit-2.jpg", "digit-9.jpg"]
    current_dir = pathlib.Path(os.path.dirname(__file__)).resolve()

    dataset = []
    for file in image_files:
        img = Image.open(current_dir / file).resize((28, 28))
        img = np.asarray(img).astype("uint8")
        dataset.append((file, img))

    return dataset
