import pathlib
import os
import datetime
import subprocess
from tvm.contrib.download import download_testdata
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
    parent_dir = pathlib.Path(os.path.dirname(__file__)).resolve()
    board_workspace = (
        parent_dir
        / f"workspace_{project_name}_{zephyr_board}"
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


def open_sine_model():
    model_url = "https://people.linaro.org/~tom.gall/sine_model.tflite"
    model_file = "sine_model.tflite"
    model_path = download_testdata(model_url, model_file, module="data")

    tflite_model_buf = open(model_path, "rb").read()

    try:
        import tflite

        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model

        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)


    input_tensor = "dense_4_input"
    input_shape = (1,)
    input_dtype = "float32"
    input = (input_tensor, input_shape, input_dtype)

    relay_mod, params = relay.frontend.from_tflite(tflite_model, shape_dict={input_tensor: input_shape}, dtype_dict={input_tensor: input_dtype})

    return (relay_mod, params, input)
