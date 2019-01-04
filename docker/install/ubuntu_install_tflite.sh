#install the necessary dependancies for tflite
pip3 install flatbuffers
pip2 install flatbuffers

wget https://github.com/dmlc/web-data/raw/master/tensorflow/tflite/whl/tflite-0.0.1-py2-none-any.whl
wget https://github.com/dmlc/web-data/raw/master/tensorflow/tflite/whl/tflite-0.0.1-py3-none-any.whl
pip2 install tflite-0.0.1-py2-none-any.whl
pip3 install tflite-0.0.1-py3-none-any.whl
