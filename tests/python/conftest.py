import sys
import tvm

collect_ignore = []
if sys.platform.startswith("win"):
    collect_ignore.append("frontend/caffe")
    collect_ignore.append("frontend/caffe2")
    collect_ignore.append("frontend/coreml")
    collect_ignore.append("frontend/darknet")
    collect_ignore.append("frontend/keras")
    collect_ignore.append("frontend/mxnet")
    collect_ignore.append("frontend/pytorch")
    collect_ignore.append("frontend/tensorflow")
    collect_ignore.append("frontend/tflite")
    collect_ignore.append("frontend/onnx")
    collect_ignore.append("driver/tvmc/test_autoscheduler.py")
    collect_ignore.append("unittest/test_auto_scheduler_cost_model.py")  # stack overflow
    # collect_ignore.append("unittest/test_auto_scheduler_measure.py") # exception ignored
    collect_ignore.append("unittest/test_auto_scheduler_search_policy.py")  # stack overflow
    # collect_ignore.append("unittest/test_auto_scheduler_measure.py") # exception ignored

    collect_ignore.append("unittest/test_tir_intrin.py")

if tvm.support.libinfo().get("USE_MICRO", "OFF") != "ON":
    collect_ignore.append("unittest/test_micro_transport.py")
