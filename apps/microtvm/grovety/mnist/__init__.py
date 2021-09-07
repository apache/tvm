
def open_model(model_path: str):
    import onnx
    import tvm.relay as relay
    onnx_model = onnx.load(model_path)
    shape = {"Input3": (1, 1, 28, 28)}
    relay_mod, params = relay.frontend.from_onnx(onnx_model, shape=shape, freeze_params=True)
    relay_mod = relay.transform.DynamicToStatic()(relay_mod)

    input = ("Input3", (1, 1, 28, 28), 'float32')
    output = ("Plus214_Output_0", (1, 10), 'float32')
    return (relay_mod, params, input, output)


def open_Q_model(model_path: str):
    import tflite
    import tvm.relay as relay

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
