import os
# import onnx
import tensorflow as tf
from tensorflow.keras import models as models

import tvm
from tvm import relay


nets = [
    # "conv_transpose_2d.onnx",
    "DS-CNN-KWS_BN.h5",
    # "resnet.tflite",
    "lstm2.h5",
    ]
for n in nets:
    try:
        if n.endswith(".onnx"):
            onnx_model = onnx.load(n)
            input = onnx_model.graph.input[0]
            shape = tuple(d.dim_value if d.HasField("dim_value") else 1 for d in input.type.tensor_type.shape.dim)
            shape_dict = {input.name: shape}
            relay_mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        elif n.endswith(".tflite"):
            tflite_model_buf = open(n, "rb").read()
            try:
                import tflite

                tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
            except AttributeError:
                import tflite.Model

                tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

            # Load TFLite model and allocate tensors.
            interpreter = tf.lite.Interpreter(model_path=n)
            interpreter.allocate_tensors()

            # Get input and output tensors.
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            input = input_details[0]
            input_tensor = input["name"]
            input_shape = tuple(input["shape"])
            input_dtype = input["dtype"].__name__

            relay_mod, params = relay.frontend.from_tflite(
                tflite_model,
                shape_dict={input_tensor: input_shape},
                dtype_dict={input_tensor: input_dtype})
        elif n.endswith(".h5"):
            model = models.load_model(n, custom_objects={"mod": models})
            # print(model.summary())
            input_tensor = model.input.name.split(":")[0]
            input_shape = tuple(model.input.shape)
            print(model.input.shape)
            print(input_shape)
            shape_dict = {"inputs": (1, 1, 49, 10)}
            relay_mod, params = relay.frontend.from_keras(model, shape_dict)


        if relay_mod:
            with open(f'{n}.relay', "w") as fo:
                fo.write(relay_mod.astext(show_meta_data=True))
            relay_mod, _ = relay.optimize(relay_mod, "llvm", params)
            with open(f'{n}.relay_opt', "w") as fo:
                fo.write(relay_mod.astext(show_meta_data=True))

        print(f'\n{n} compiled')
    except Exception as e:
        print(f'\n{n} failed: {e}')