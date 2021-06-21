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
# pylint: disable=invalid-name,missing-class-docstring,missing-function-docstring
"""The string templates for Android NNAPI codegen."""
import string

ANN_PREFIX = "ANEURALNETWORKS_"


class declare_type:
    @staticmethod
    def substitute(**kwargs):
        tipe = kwargs["tipe"]
        ret = ""
        ret += f"""ANeuralNetworksOperandType {tipe["name"]};
{tipe["name"]}.type = {tipe["type"]};
{tipe["name"]}.scale = 0.f;
{tipe["name"]}.zeroPoint = 0;
"""
        if "shape" in tipe:
            ret += f"""{tipe["name"]}.dimensionCount = {tipe["shape"]["rank"]};
static uint32_t {tipe["dim_name"]}[{tipe["shape"]["rank"]}] = {tipe["shape"]["str"]};
{tipe["name"]}.dimensions = {tipe["dim_name"]};
"""
        else:
            ret += f"""{tipe["name"]}.dimensionCount = 0;
{tipe["name"]}.dimensions = NULL;
"""

        return ret


declare_operand = string.Template(
    """JSON2NNAPI_CHECK_EQ(
  ANeuralNetworksModel_addOperand(
    ${model},
    &${type}
  ),
  ANEURALNETWORKS_NO_ERROR
); // Operand ${index}
"""
)

declare_constant = {
    "scalar": string.Template(
        """static ${dtype} ${name} = ${value};
"""
    ),
    "array": string.Template(
        """static ${dtype} ${name}[${length}] = ${value};
"""
    ),
}


class declare_memory:
    @staticmethod
    def substitute(**kwargs):
        file_path = kwargs["file_path"]
        mem_size = kwargs["mem_size"]
        ret = f"""{{
  ANeuralNetworksMemory* mem = nullptr;
  int fd = open("{file_path}", O_RDONLY);
  JSON2NNAPI_CHECK_NE(fd, -1);
  JSON2NNAPI_CHECK_EQ(
    ANeuralNetworksMemory_createFromFd(
      {mem_size},
      PROT_READ,
      fd,
      0,
      &mem
    ),
    ANEURALNETWORKS_NO_ERROR
  );
  this->memories_.push_back({{ fd, mem }});
}}
"""
        return ret


initialize_operand = {
    "memory_ptr": string.Template(
        """JSON2NNAPI_CHECK_EQ(
  ANeuralNetworksModel_setOperandValue(
    ${model},
    ${op_idx},
    ${memory_ptr},
    ${memory_size}
  ),
  ANEURALNETWORKS_NO_ERROR
);
"""
    ),
    "ann_memory": string.Template(
        """JSON2NNAPI_CHECK_EQ(
  ANeuralNetworksModel_setOperandValueFromMemory(
    ${model},
    ${op_idx},
    std::get< 1 >(this->memories_[${memory_idx}]),
    0,
    ${length}
  ),
  ANEURALNETWORKS_NO_ERROR
);
"""
    ),
}


class declare_operation:
    @staticmethod
    def substitute(**kwargs):
        inputs = kwargs["inputs"]
        outputs = kwargs["outputs"]
        model = kwargs["model"]
        op_code = kwargs["op_code"]
        ret = f"""{{
  static uint32_t inputIndexes[{inputs["length"]}] = {inputs["str"]};
  static uint32_t outputIndexes[{outputs["length"]}] = {outputs["str"]};
  JSON2NNAPI_CHECK_EQ(
    ANeuralNetworksModel_addOperation(
      {model},
      {op_code},
      {inputs["length"]},
      inputIndexes,
      {outputs["length"]},
      outputIndexes
    ),
    ANEURALNETWORKS_NO_ERROR
  );
}}
"""
        return ret


class declare_inputs_outputs:
    @staticmethod
    def substitute(**kwargs):
        model = kwargs["model"]
        inputs = kwargs["inputs"]
        outputs = kwargs["outputs"]
        ret = f"""static uint32_t {model}InputIndexes[{inputs["length"]}] = {inputs["str"]};
static uint32_t {model}OutputIndexes[{outputs["length"]}] = {outputs["str"]};
JSON2NNAPI_CHECK_EQ(
  ANeuralNetworksModel_identifyInputsAndOutputs(
    {model},
    {inputs["length"]},
    {model}InputIndexes,
    {outputs["length"]},
    {model}OutputIndexes
  ),
  ANEURALNETWORKS_NO_ERROR
);
"""
        return ret


class declare_wrapper_class:
    @staticmethod
    def substitute(**kwargs):
        clas = kwargs["class"]
        codes = kwargs["codes"]
        ret = f"""#define JSON2NNAPI_CHECK_EQ(a, b) {{ assert((a) == (b)); }}
#define JSON2NNAPI_CHECK_NE(a, b) {{ assert((a) != (b)); }}
class {clas["self"]["name"]}
{{
public:
  {clas["self"]["name"]}()
  {{
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksModel_create(&this->{clas["model"]["name"]}), ANEURALNETWORKS_NO_ERROR);
    this->createAnnModel();
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksModel_finish(this->{clas["model"]["name"]}), ANEURALNETWORKS_NO_ERROR);
#if __ANDROID_API__ >= 29 && defined(JSON2NNAPI_FORCE_CPU_FALLBACK)
    uint32_t num_nnapi_devices;
    JSON2NNAPI_CHECK_EQ(ANeuralNetworks_getDeviceCount(&num_nnapi_devices), ANEURALNETWORKS_NO_ERROR);
    ANeuralNetworksDevice * nnapi_fallback_dev;
    for (int i = 0; i < num_nnapi_devices; i++)
    {{
      JSON2NNAPI_CHECK_EQ(ANeuralNetworks_getDevice(i, &nnapi_fallback_dev), ANEURALNETWORKS_NO_ERROR);
      int32_t dev_type;
      JSON2NNAPI_CHECK_EQ(ANeuralNetworksDevice_getType(nnapi_fallback_dev, &dev_type), ANEURALNETWORKS_NO_ERROR);
      if (dev_type == ANEURALNETWORKS_DEVICE_CPU)
      {{
        break;
      }}
    }}
    {{
      const ANeuralNetworksDevice * const dev_list[] = {{ nnapi_fallback_dev }};
      JSON2NNAPI_CHECK_EQ(ANeuralNetworksCompilation_createForDevices(this->{clas["model"]["name"]}, dev_list, 1, &this->{clas["compilation"]["name"]}), ANEURALNETWORKS_NO_ERROR);
    }}
#else // #if __ANDROID_API__ >= 29 && defined(JSON2NNAPI_FORCE_CPU_FALLBACK)
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksCompilation_create(this->{clas["model"]["name"]}, &this->{clas["compilation"]["name"]}), ANEURALNETWORKS_NO_ERROR);
#endif // #if __ANDROID_API__ >= 29 && defined(JSON2NNAPI_FORCE_CPU_FALLBACK)
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksCompilation_finish(this->{clas["compilation"]["name"]}), ANEURALNETWORKS_NO_ERROR);
  }}

  ~{clas["self"]["name"]}()
  {{
    ANeuralNetworksCompilation_free(this->{clas["compilation"]["name"]});
    ANeuralNetworksModel_free(this->{clas["model"]["name"]});
    for (const auto &t: this->memories_)
    {{
      ANeuralNetworksMemory_free(std::get< 1 >(t));
      close(std::get< 0 >(t));
    }}
  }}

  void createAnnModel()
  {{
{codes["model_creation"]}
  }}

  void execute({clas["execution"]["func_params_decl_str"]})
  {{
    ANeuralNetworksExecution* {clas["execution"]["name"]} = nullptr;
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksExecution_create(this->{clas["compilation"]["name"]}, &{clas["execution"]["name"]}), ANEURALNETWORKS_NO_ERROR);

{codes["set_execution_io"]}

    ANeuralNetworksEvent* {clas["execution"]["end_event_name"]} = nullptr;
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksExecution_startCompute({clas["execution"]["name"]}, &{clas["execution"]["end_event_name"]}), ANEURALNETWORKS_NO_ERROR);
    JSON2NNAPI_CHECK_EQ(ANeuralNetworksEvent_wait({clas["execution"]["end_event_name"]}), ANEURALNETWORKS_NO_ERROR);
    ANeuralNetworksEvent_free({clas["execution"]["end_event_name"]});
    ANeuralNetworksExecution_free({clas["execution"]["name"]});
  }}

private:
  ANeuralNetworksModel* {clas["model"]["name"]} = nullptr;
  ANeuralNetworksCompilation* {clas["compilation"]["name"]} = nullptr;
  std::vector< std::tuple< int, ANeuralNetworksMemory* > > memories_;
}};
"""
        return ret


set_execution_input = string.Template(
    """JSON2NNAPI_CHECK_EQ(
  ANeuralNetworksExecution_setInput(
    ${execution},
    ${input_idx},
    nullptr,
    ${memory_ptr},
    ${memory_size}
  ),
  ANEURALNETWORKS_NO_ERROR
);
"""
)

set_execution_output = string.Template(
    """JSON2NNAPI_CHECK_EQ(
  ANeuralNetworksExecution_setOutput(
    ${execution},
    ${output_idx},
    nullptr,
    ${memory_ptr},
    ${memory_size}
  ),
  ANEURALNETWORKS_NO_ERROR
);
"""
)
