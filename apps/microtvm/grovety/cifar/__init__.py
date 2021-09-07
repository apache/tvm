

def open_model(model_path: str):
    import tflite
    import tvm.relay as relay

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
