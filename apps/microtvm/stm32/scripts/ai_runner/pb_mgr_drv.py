# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
Driver for proto buff messages
"""

import time as t
import numpy as np

from google.protobuf.internal.encoder import _VarintBytes

# from .ai_runner import AiRunner, AiRunnerCallback, AiRunnerDriver
from .ai_runner import AiRunner, AiRunnerDriver
from .ai_runner import (
    HwIOError,
    InvalidMsgError,
    NotInitializedMsgError,
    AiRunnerError,
    InvalidParamError,
)
from .stm32_utility import stm32_id_to_str, stm32_config_to_str
from . import stm32msg_pb2 as stm32msg


class AiBufferFmt:
    """ Helper class to manipulate the AI buffer format"""

    AI_BUFFER_FORMAT_NONE = 0x00000040
    AI_BUFFER_FORMAT_FLOAT = 0x01821040

    AI_BUFFER_FORMAT_U8 = 0x00040440
    AI_BUFFER_FORMAT_U16 = 0x00040840
    AI_BUFFER_FORMAT_S8 = 0x00840440
    AI_BUFFER_FORMAT_S16 = 0x00840840

    AI_BUFFER_FORMAT_Q = 0x00840040
    AI_BUFFER_FORMAT_Q7 = 0x00840447
    AI_BUFFER_FORMAT_Q15 = 0x0084084F

    AI_BUFFER_FORMAT_UQ = 0x00040040
    AI_BUFFER_FORMAT_UQ7 = 0x00040447
    AI_BUFFER_FORMAT_UQ15 = 0x0004084F

    FMT_MASK = 0x01FFFFFF

    FLOAT_MASK = 0x1
    FLOAT_SHIFT = 24

    SIGN_MASK = 0x1
    SIGN_SHIFT = 23

    TYPE_ID_MASK = 0xF
    TYPE_ID_SHIFT = 17

    BITS_MASK = 0x7F
    BITS_SHIFT = 7

    FBITS_MASK = 0x7F
    FBITS_SHIFT = 0

    TYPE_NONE = 0x0
    TYPE_FLOAT = 0x1
    TYPE_Q = 0x2
    TYPE_BOOL = 0x3

    FLAG_CONST = 0x1 << 30
    FLAG_STATIC = 0x1 << 29
    FLAG_IS_IO = 0x1 << 27


def _fmt_to_dict(fmt):
    """Return a dict with fmt field values"""
    _dict = {
        "type": (fmt >> AiBufferFmt.TYPE_ID_SHIFT) & AiBufferFmt.TYPE_ID_MASK,
        "bits": (fmt >> AiBufferFmt.BITS_SHIFT) & AiBufferFmt.BITS_MASK,
        "fbits": (fmt >> AiBufferFmt.FBITS_SHIFT) & AiBufferFmt.FBITS_MASK,
        "sign": (fmt >> AiBufferFmt.SIGN_SHIFT) & AiBufferFmt.SIGN_MASK,
    }
    _dict["fbits"] = _dict["fbits"] - 64
    return _dict


def _to_version(ver):
    """Convert uint32 version to tuple (major, minor, sub)"""
    return (ver >> 24 & 0xFF, ver >> 16 & 0xFF, ver >> 8 & 0xFF)


def _fmt_to_np_type(fmt):
    """Return numpy type based on AI buffer format definition"""

    desc = _fmt_to_dict(fmt)

    _type = desc["type"]
    _bits = desc["bits"]
    _sign = desc["sign"]

    if _type == AiBufferFmt.TYPE_FLOAT:
        if _bits == 32:
            return np.float32
        if _bits == 16:
            return np.float16
        return np.float64

    if _type == AiBufferFmt.TYPE_NONE:
        return np.void

    if _type == AiBufferFmt.TYPE_BOOL:
        return np.bool

    if _type == AiBufferFmt.TYPE_Q:
        if _sign and _bits == 8:
            return np.int8
        if _sign and _bits == 16:
            return np.int16
        if _sign and _bits == 32:
            return np.int32
        if _bits == 8:
            return np.uint8
        if _bits == 16:
            return np.uint16
        if _bits == 32:
            return np.uint32

    raise NotImplementedError("AI type to NP type not supported")


class AiPbMsg(AiRunnerDriver):
    """Class to handle the messages (protobuf-based)"""

    def __init__(self, parent, io_drv):
        """Constructor"""
        if not hasattr(io_drv, "set_parent"):
            raise InvalidParamError("Invalid IO Hw Driver type (io_drv)")
        self._req_id = 0
        self._io_drv = io_drv
        self._models = dict()  # cache the description of the models
        self._sync = None  # cache for the sync message
        self._sys_info = None  # cache for sys info message
        self._io_drv.set_parent(self)
        super(AiPbMsg, self).__init__(parent)

    @property
    def is_connected(self):
        return self._io_drv.is_connected

    def connect(self, desc=None, **kwargs):
        """Connect to the stm.ai run-time"""
        if self._io_drv.is_connected:
            return False
        return self._io_drv.connect(desc, **kwargs)

    @property
    def capabilities(self):
        """Capabilities"""
        if self.is_connected:
            cap_ = [AiRunner.Caps.IO_ONLY]
            if self._sync.capability & stm32msg.CAP_INSPECTOR:
                cap_.extend([AiRunner.Caps.PER_LAYER, AiRunner.Caps.PER_LAYER_WITH_DATA])
            if self._sync.capability & stm32msg.CAP_SELF_TEST:
                cap_.append(AiRunner.Caps.SELF_TEST)
            return cap_
        return []

    def disconnect(self):
        self._models = dict()
        self._sys_info = None
        self._sync = None
        self._io_drv.disconnect()

    def short_desc(self):
        ver_ = "v{}.{}".format(stm32msg.P_VERSION_MAJOR, stm32msg.P_VERSION_MINOR)
        io_ = self._io_drv.short_desc()
        return "STM Proto-buffer protocol " + ver_ + " (" + io_ + ")"

    def _waiting_io_ack(self, timeout):
        """Wait a ack"""
        start_time = t.perf_counter()
        while True:
            if self._io_drv.read(1):
                break
            if t.perf_counter() - start_time > timeout / 1000.0:
                return False
        return True

    def _write_io_packet(self, payload, delay=0):
        iob = bytearray(stm32msg.IO_OUT_PACKET_SIZE + 1)
        iob[0] = len(payload)
        for i, val in enumerate(payload):
            iob[i + 1] = val
        if not delay:
            ww_ = self._io_drv.write(iob)
        else:
            ww_ = 0
            for elem in iob:
                ww_ += self._io_drv.write(elem.to_bytes(1, "big"))
                t.sleep(delay)
        return ww_

    def _write_delimited(self, mess, timeout=5000):
        """Helper function to write a message prefixed with its size"""

        if not mess.IsInitialized():
            raise NotInitializedMsgError

        buff = mess.SerializeToString()
        _head = _VarintBytes(mess.ByteSize())

        buff = _head + buff

        packs = [
            buff[i : i + stm32msg.IO_OUT_PACKET_SIZE]
            for i in range(0, len(buff), stm32msg.IO_OUT_PACKET_SIZE)
        ]

        n_w = self._write_io_packet(packs[0])
        for pack in packs[1:]:
            if not self._waiting_io_ack(timeout):
                break
            n_w += self._write_io_packet(pack)

        return n_w

    def _parse_and_check(self, data, msg_type=None):
        """Parse/convert and check the received buffer"""
        resp = stm32msg.respMsg()
        try:
            resp.ParseFromString(data)
        except BaseException as exc_:
            raise InvalidMsgError(str(exc_))
        if msg_type is None:
            return resp
        if resp.WhichOneof("payload") != msg_type:
            raise InvalidMsgError(
                "receive '{}' instead '{}'".format(resp.WhichOneof("payload"), msg_type)
            )
        return None

    def _waiting_msg(self, timeout, msg_type=None):
        """Helper function to receive a message"""
        buf = bytearray()

        packet_s = int(stm32msg.IO_IN_PACKET_SIZE + 1)
        if timeout == 0:
            t.sleep(0.2)

        start_time = t.monotonic()
        while True:
            p_buf = bytearray()
            while len(p_buf) < packet_s:
                io_buf = self._io_drv.read(packet_s - len(p_buf))
                if io_buf:
                    p_buf += io_buf
                else:
                    cum_time = t.monotonic() - start_time
                    if timeout and (cum_time > timeout / 1000):
                        raise TimeoutError(
                            "STM32 - read timeout {:.1f}ms/{}ms".format(cum_time * 1000, timeout)
                        )
                    if timeout == 0:
                        return self._parse_and_check(buf, msg_type)
            last = p_buf[0] & stm32msg.IO_HEADER_EOM_FLAG
            # cbuf[0] = cbuf[0] & 0x7F & ~stm32msg.IO_HEADER_SIZE_MSK
            p_buf[0] &= 0x7F  # & ~stm32msg.IO_HEADER_SIZE_MSK)
            if last:
                buf += p_buf[1 : 1 + p_buf[0]]
                break
            buf += p_buf[1:packet_s]
        resp = self._parse_and_check(buf, msg_type)
        return resp

    def _send_request(self, cmd, param=0, name=None, opt=0):
        """Build a request msg and send it"""
        self._req_id += 1
        req_msg = stm32msg.reqMsg()
        req_msg.reqid = self._req_id
        req_msg.cmd = cmd
        req_msg.param = param
        req_msg.opt = opt
        if name is not None and isinstance(name, str):
            req_msg.name = name
        else:
            req_msg.name = ""
        n_w = self._write_delimited(req_msg)
        return n_w, req_msg

    def _send_ack(self, param=0, err=0):
        """Build an acknowledge msg and send it"""
        ack_msg = stm32msg.ackMsg(param=param, error=err)
        return self._write_delimited(ack_msg)

    def _device_log(self, resp):
        """Process a log message from a device"""
        if resp.WhichOneof("payload") == "log":
            msg = "STM32:{}: {}".format(resp.log.level, resp.log.str)
            self._logger.info(msg)
            self._send_ack()
            return True
        return False

    def _waiting_answer(self, timeout=10000, msg_type=None, state=None):
        """Wait an answer/msg from the device and post-process it"""

        cont = True
        while cont:  # to manage the "log" msg
            resp = self._waiting_msg(timeout=timeout)
            if resp.reqid != self._req_id:
                raise InvalidMsgError(
                    "SeqID is not valid - {} instead {}".format(resp.reqid, self._req_id)
                )
            cont = self._device_log(resp)

        if msg_type and resp.WhichOneof("payload") != msg_type:
            raise InvalidMsgError(
                "receive '{}' instead '{}'".format(resp.WhichOneof("payload"), msg_type)
            )

        if state and state != resp.state:
            raise HwIOError("Invalid state: {} instead {}".format(resp.state, state))

        return resp

    def _cmd_sync(self, timeout):
        """SYNC command"""
        self._send_request(stm32msg.CMD_SYNC)
        resp = self._waiting_answer(timeout=timeout, msg_type="sync", state=stm32msg.S_IDLE)
        return resp.sync

    def _cmd_sys_info(self, timeout):
        """SYS_INFO command"""
        self._send_request(stm32msg.CMD_SYS_INFO)
        resp = self._waiting_answer(timeout=timeout, msg_type="sinfo", state=stm32msg.S_IDLE)
        return resp.sinfo

    def _cmd_network_info(self, timeout, param=0):
        """SYS_INFO command"""
        self._send_request(stm32msg.CMD_NETWORK_INFO, param=param)
        resp = self._waiting_answer(timeout=timeout, state=stm32msg.S_IDLE)
        if resp.WhichOneof("payload") == "ninfo":
            return resp.ninfo
        return None

    def _cmd_run(self, timeout, c_name, param):
        """NETWORK_RUN command"""
        self._send_request(stm32msg.CMD_NETWORK_RUN, param=param, name=c_name)
        resp = self._waiting_answer(timeout=timeout, msg_type="ack", state=stm32msg.S_WAITING)
        return resp

    def is_alive(self, timeout=500):
        try:
            self._sync = self._cmd_sync(timeout)
        except (AiRunnerError, TimeoutError) as exc_:
            self._logger.debug("is_alive() %s", str(exc_))
            return False
        return True

    def _to_device(self):
        """Return a dict with the device settings"""
        if self._sys_info is None:
            self._sys_info = self._cmd_sys_info(timeout=500)

        return {
            "dev_id": stm32_id_to_str(self._sys_info.devid),
            "sys_clock": self._sys_info.sclock,
            "bus_clock": self._sys_info.hclock,
            "config": stm32_config_to_str(self._sys_info.cache),
        }

    def _to_runtime(self, model_info):
        """Return a dict with the runtime attributes"""

        # Basic hack to detect the type of run-time
        name_rt = (
            "X-CUBE-AI"
            if model_info.tool_api_version != model_info.api_version
            else model_info.tool_revision
        )
        return {
            "name": name_rt,
            "version": _to_version(model_info.runtime_version),
            "capabilities": self.capabilities,
            "tools_version": _to_version(model_info.tool_version),
        }

    def _to_io_tensor(self, buffer, idx, name=None):
        """Return a dict with the tensor IO attributes"""

        item = {
            "name": "{}_{}".format(name if name else "tensor", idx + 1),
            "shape": (buffer.n_batches, buffer.height, buffer.width, buffer.channels),
            "type": _fmt_to_np_type(buffer.format),
            "scale": np.float32(buffer.scale) if buffer.scale != 0.0 else None,
            "zero_point": np.int32(buffer.zeropoint),
        }

        # adjust type of the zero_point
        if item["scale"] is not None:
            item["zero_point"].astype(item["type"])

        return item

    def _model_to_dict(self, model_info):
        """Return a dict with the network info"""
        return {
            "name": model_info.model_name,
            "model_datetime": model_info.model_datetime,  # date of creation
            "compile_datetime": model_info.compile_datetime,  # date of compilation
            "hash": model_info.model_signature,
            "n_nodes": model_info.n_nodes,
            "inputs": [
                self._to_io_tensor(d_, i, "input") for i, d_ in enumerate(model_info.inputs)
            ],
            "outputs": [
                self._to_io_tensor(d_, i, "output") for i, d_ in enumerate(model_info.outputs)
            ],
            "weights": model_info.weights.channels,
            "activations": model_info.activations.channels,
            "macc": model_info.n_macc,
            "runtime": self._to_runtime(model_info),
            "device": self._to_device(),
        }

    def get_info(self, name=None):
        """Return a dict with the network info of the given model"""
        if not self._models:
            return dict()
        if name is None or name not in self._models.keys():
            # first c-model is used
            name = self._models.keys()[0]
        model = self._models[name]
        return self._model_to_dict(model)

    def discover(self, flush=False):
        """Build the list of the available model"""
        if flush:
            self._models.clear()
        if self._models:
            return list(self._models.keys())
        param, cont = 0, True

        while cont:
            n_info = self._cmd_network_info(timeout=5000, param=param)
            if n_info is not None:
                self._models[n_info.model_name] = n_info
                msg = 'discover() found="{}"'.format(str(n_info.model_name))
                self._logger.debug(msg)
                param += 1
            else:
                cont = False
        return list(self._models.keys())

    def _to_buffer_msg(self, data, buffer_desc):
        """Convert ndarray to aiBufferByteMsg"""
        msg_ = stm32msg.aiBufferByteMsg()
        msg_.shape.n_batches = buffer_desc.n_batches
        msg_.shape.height = buffer_desc.height
        msg_.shape.width = buffer_desc.width
        msg_.shape.channels = buffer_desc.channels
        msg_.shape.format = buffer_desc.format
        msg_.shape.scale = 0.0
        msg_.shape.zeropoint = 0
        dt_ = np.dtype(data.dtype.type)
        dt_ = dt_.newbyteorder("<")
        msg_.datas = bytes(data.astype(dt_).flatten().tobytes())
        return msg_

    def _from_buffer_msg(self, msg, fill_with_zero=False):
        """Convert aiBufferByteMsg to ndarray"""
        if isinstance(msg, stm32msg.respMsg):
            buffer = msg.node.buffer
        elif isinstance(msg, stm32msg.nodeMsg):
            buffer = msg.node
        else:
            buffer = msg
        shape_ = (
            buffer.shape.n_batches,
            buffer.shape.height,
            buffer.shape.width,
            buffer.shape.channels,
        )
        dt_ = np.dtype(_fmt_to_np_type(buffer.shape.format))
        dt_ = dt_.newbyteorder("<")
        if fill_with_zero:  # or not buffer.datas:
            return np.zeros(shape_, dtype=dt_), shape_
        if not buffer.datas:
            return np.array([], dtype=dt_), shape_
        return np.reshape(np.frombuffer(buffer.datas, dtype=dt_), shape_), shape_

    def _send_buffer(self, data, buffer_desc, is_last=False):
        """Send a buffer to the device and wait an ack"""

        buffer_msg = self._to_buffer_msg(data, buffer_desc)
        self._write_delimited(buffer_msg)
        state = stm32msg.S_PROCESSING if is_last else stm32msg.S_WAITING
        self._waiting_answer(msg_type="ack", state=state)
        self._send_ack()

    def _receive_features(self, profiler, callback):
        """Collect the intermediate/hidden values"""

        # main loop to receive the datas
        idx_node = 0
        duration = 0.0
        while True:
            resp = self._waiting_answer(msg_type="node", timeout=50000)
            # state=stm32msg.S_PROCESSING)
            ilayer = resp.node

            is_internal = ilayer.type >> 16
            # is_internal = True if is_internal & stm32msg.LAYER_TYPE_INTERNAL_LAST or\
            #    is_internal & stm32msg.LAYER_TYPE_INTERNAL else False
            is_internal = (
                is_internal & stm32msg.LAYER_TYPE_INTERNAL_LAST
                or is_internal & stm32msg.LAYER_TYPE_INTERNAL
            )

            if not is_internal:
                return resp

            self._send_ack()

            if profiler:
                feature, shape = self._from_buffer_msg(resp.node.buffer)
                duration += ilayer.duration
                if idx_node >= len(profiler["c_nodes"]):
                    item = {
                        "c_durations": [ilayer.duration],
                        "m_id": ilayer.id,
                        "layer_type": ilayer.type & 0x7FFF,
                        "type": feature.dtype.type,
                        "shape": shape,  # feature.shape,
                        "scale": resp.node.buffer.shape.scale,
                        "zero_point": resp.node.buffer.shape.zeropoint,
                        "data": feature,
                    }
                    profiler["c_nodes"].append(item)
                else:
                    item = profiler["c_nodes"][idx_node]
                    item["c_durations"].append(ilayer.duration)
                    item["data"] = np.append(item["data"], feature, axis=0)

                if callback:
                    callback.on_node_end(
                        idx_node,
                        [feature],
                        logs={
                            "dur": ilayer.duration,
                            "shape": [shape],
                            "m_id": ilayer.id,
                            "layer-type": ilayer.type & 0x7FFF,
                        },
                    )

            # end main loop
            if ilayer.type >> 16 & stm32msg.LAYER_TYPE_INTERNAL_LAST:
                break

            idx_node += 1

        # retreive the report
        # legacy support (2.1 protocol) - not used here
        #  global execution time is reported in the output tensors
        self._waiting_answer(msg_type="report", timeout=20000, state=stm32msg.S_PROCESSING)
        self._send_ack()

        if profiler:
            profiler["c_durations"].append(duration)

        return None

    def invoke_sample(self, inputs, **kwargs):
        """Invoke the model (sample mode)"""

        if inputs[0].shape[0] != 1:
            raise HwIOError("Should be called with a batch size of 1")

        name = kwargs.pop("name", None)

        if name is None or name not in self._models.keys():
            raise InvalidParamError("Invalid requested model name: " + name)

        model = self._models[name]

        profiler = kwargs.pop("profiler", None)
        mode = kwargs.pop("mode", AiRunner.Mode.IO_ONLY)
        callback = kwargs.pop("callback", None)

        if mode == AiRunner.Mode.PER_LAYER:
            param = stm32msg.P_RUN_MODE_INSPECTOR_WITHOUT_DATA
        elif mode == AiRunner.Mode.PER_LAYER_WITH_DATA:
            param = stm32msg.P_RUN_MODE_INSPECTOR
        else:
            param = stm32msg.P_RUN_MODE_NORMAL

        s_outputs = []

        # start a RUN task
        self._cmd_run(timeout=1000, c_name=name, param=param)

        # send the inputs
        for idx, buffer_desc in enumerate(model.inputs):
            in_buff = inputs[idx]
            # is_last = True if (idx + 1) == model.n_inputs else False
            is_last = (idx + 1) == model.n_inputs
            self._send_buffer(in_buff, buffer_desc, is_last=is_last)

        # receive the features
        resp = self._receive_features(profiler, callback)

        # receive the outputs
        for idx, buffer_desc in enumerate(model.outputs):
            # is_last = True if (idx + 1) == model.n_outputs else False
            is_last = (idx + 1) == model.n_outputs
            state = stm32msg.S_DONE if is_last else stm32msg.S_PROCESSING
            if resp is None:
                resp = self._waiting_answer(msg_type="node", timeout=50000, state=state)
            output, _ = self._from_buffer_msg(resp.node.buffer)
            s_outputs.append(output)
            if not is_last:
                self._send_ack()
                resp = None

        if profiler:
            profiler["debug"]["exec_times"].append(resp.node.duration)
            if mode != AiRunner.Mode.IO_ONLY:
                dur = profiler["c_durations"][-1]
            else:
                dur = resp.node.duration
                profiler["c_durations"].append(dur)
        else:
            dur = resp.node.duration

        return s_outputs, dur
