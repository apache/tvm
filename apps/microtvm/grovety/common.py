import pathlib
import os
import datetime
import subprocess
import tvm.relay as relay


PLATFORMS = {
    "qemu_x86": ("host", "qemu_x86"),
    "qemu_riscv32": ("host", "qemu_riscv32"),
    "qemu_riscv64": ("host", "qemu_riscv64"),
    "mps2_an521": ("mps2_an521", "mps2_an521"),
    "nrf5340dk": ("nrf5340dk", "nrf5340dk_nrf5340_cpuapp"),
    "stm32f746xx_disco": ("stm32f746xx", "stm32f746g_disco"),
    "stm32f746xx_nucleo": ("stm32f746xx", "nucleo_f746zg"),
    "stm32l4r5zi_nucleo": ("stm32l4r5zi", "nucleo_l4r5zi"),
    "zynq_mp_r5": ("zynq_mp_r5", "qemu_cortex_r5"),
    "LPCXpresso5569": ("nxp_lpc55S6x", "lpcxpresso55s69_ns")
}


def tvm_repo_root():
    return subprocess.check_output(["git", "rev-parse", "--show-toplevel"], encoding='utf-8').strip()


def create_workspace_dir(platform, project_name, mkdir=True):
    _, zephyr_board = PLATFORMS[platform]
    current_dir = pathlib.Path(os.path.dirname(__file__)).resolve()
    board_workspace = (
        current_dir
        / f"workspace/{project_name}_{zephyr_board}"
        / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    )
    board_workspace_base = str(board_workspace)
    number = 1
    while board_workspace.exists():
        board_workspace = pathlib.Path(board_workspace_base + f"-{number}")
        number += 1

    if mkdir:
        os.makedirs(board_workspace, exist_ok=True)
    else:
        os.makedirs(board_workspace.parent, exist_ok=True)

    return board_workspace


def download_sine_model():
    model_url = "https://people.linaro.org/~tom.gall/sine_model.tflite"
    model_file = "sine_model.tflite"
    return download_model(model_url, model_file)


def download_model(url: str, filename: str):
    from tvm.contrib.download import download_testdata
    return download_testdata(url, filename, module="data")


def open_tflite_model(model_path: str):
    import tflite
    import tensorflow as tf

    tflite_model_buf = open(model_path, "rb").read()
    try:
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in0 = input_details[0]
    out0 = output_details[0]

    input = (in0["name"], tuple(in0["shape"]), in0["dtype"].__name__)
    output = (out0["name"], tuple(out0["shape"]), out0["dtype"].__name__)

    relay_mod, params = relay.frontend.from_tflite(tflite_model)#, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype})
    return (relay_mod, params, input, output)


def print_relay(relay, params, show_metadata=True, optimize=False):
    if optimize:
        relay, _ = relay.optimize(relay, "llvm", params)

    text = relay.astext(show_meta_data=show_metadata)
    print(text)


def create_header_file(output_path, input, output):
    import numpy as np

    c_types = {
        'int8': 'int8_t',
        'int32': 'int32_t',
        'uint8': 'uint8_t',
        'float32': 'float'
    }

    input_tensor, input_shape, input_dtype = input
    output_tensor, output_shape, output_dtype = output

    file_path = pathlib.Path(f"{output_path}/model_data.h").resolve()

    with open(file_path, "w") as header_file:
        header_file.write(
            "#include <stddef.h>\n"\
            "#include <stdint.h>\n"\
            "#include <dlpack/dlpack.h>\n\n"\
            f"#define model_input_0 {input_tensor}\n"\
            f"#define INPUT_DATA_LEN {np.prod(input_shape)}\n\n"\
            f"{c_types[input_dtype]} input_data[INPUT_DATA_LEN] = {{0}};\n\n"\
            f"#define OUTPUT_DATA_LEN {np.prod(output_shape)}\n"\
            f"{c_types[output_dtype]} output_data[OUTPUT_DATA_LEN] = {{0}};\n\n"
        )


def read_line(fd, timeout_sec: int):
    data = ""
    new_line = False
    while True:
        if new_line:
            break
        new_data = fd.read(1, timeout_sec=timeout_sec)
        for item in new_data:
            new_c = chr(item)
            data = data + new_c
            if new_c == "\n":
                new_line = True
                break
    return data


def get_message(fd, expr: str, timeout_sec: int):
    while True:
        data = read_line(fd, timeout_sec)
        if expr in data:
            return data