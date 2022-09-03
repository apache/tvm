from onnx.backend.base import Backend

from onnx.backend.base import BackendRep

from tvm import relay
import tvm

class TVMBackendRep(BackendRep):

    @staticmethod
    def get_input_data_shape_dict(model, input_data):
        if isinstance(input_data, list):
            input_names = {}
            shape_dict = {}
            for i, _ in enumerate(input_data):
                input_names[i] = model.graph.input[i].name
                shape_dict[input_names[i]] = input_data[i].shape
        else:
            input_names = model.graph.input[0].name
            shape_dict = {input_names: input_data.shape}

        return input_names, shape_dict

    def __init__(self, model, device, **kwargs):
        super(TVMBackendRep, self).__init__()

        # because models might be dynamic, there's not much we
        # can do to prepare a backend rep

        self._model = model
        if device == "CPU":
            self._device = tvm.cpu(0)
            self._target = "llvm"
        elif device == "CUDA":
            self._device = dev = tvm.cuda(0)
            self._target = "cuda"

    def run(self, inputs, **kwargs):
        if not isinstance(inputs, list):
            input_data = [inputs]

        _, shape_dict = self.get_input_data_shape_dict(self._model, inputs)

        model, params = relay.frontend.from_onnx(
                self._model,
                shape_dict)

        model = relay.transform.DynamicToStatic()(model)


        result = relay.create_executor("vm", mod=model, device=self._device, target=self._target).evaluate()(*inputs, *params)


        if isinstance(result, tvm.runtime.NDArray):
            return [result.numpy()]
        return [r.numpy() for r in result]

class TVMBackend(Backend):

    @classmethod
    def prepare(cls,
                model,
                device='CPU',
                **kwargs
                ):
        super(TVMBackend, cls).prepare(model, device, **kwargs)
        return TVMBackendRep(model, device, **kwargs)

    @classmethod
    def run_node(cls,
                 node,
                 inputs,
                 device='CPU',
                 outputs_info=None,
                 **kwargs
                 ):
        '''Simple run one operator and return the results.
        Args:
            outputs_info: a list of tuples, which contains the element type and
            shape of each output. First element of the tuple is the dtype, and
            the second element is the shape. More use case can be found in
            https://github.com/onnx/onnx/blob/main/onnx/backend/test/runner/__init__.py
        '''
        return super(TVMBackend, cls).run_node(node, inputs, device, outputs_info, **kwargs)

    @classmethod
    def supports_device(cls, device) -> bool:
        """
        Checks whether the backend is compiled with particular device support.
        In particular it's used in the testing suite.
        """
        return device == "CPU" or device == "CUDA"
