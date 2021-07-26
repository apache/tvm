#include "src/model.h"
#include "src/data/yes.c"
#include "src/data/no.c"
#include "src/data/unknown.c"
#include "src/data/silence.c"

static Model model;

void performInference(int8_t input_data[1960], char *data_name) {
  int8_t output_data[4];
  uint64_t start_time = micros();
  model.inference(input_data, output_data);
  uint64_t end_time = micros();

  Serial.print(data_name);
  Serial.print(",");
  Serial.print(end_time - start_time);
  Serial.print(",");
  for (int i = 0; i < 4; i++) {
    Serial.print(output_data[i]);
    Serial.print(",");
  }
  Serial.println();
}

void setup() {
  model = Model();
  Serial.begin(115200);
}

void loop() {
  Serial.println();
  Serial.println("category,runtime,yes,no,silence,unknown");
  performInference((int8_t*) input_yes, "yes");
  performInference((int8_t*) input_no, "no");
  performInference((int8_t*) input_silence, "silence");
  performInference((int8_t*) input_unknown, "unknown");
}
