# Download, build and install flatbuffers
git clone --recursive https://github.com/google/flatbuffers.git
cd flatbuffers
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release
make install -j8
cd ..
rm -rf flatbuffers

# Install flatbuffers python packages.
pip3 install flatbuffers
pip2 install flatbuffers

# Setup tflite from schema
mkdir tflite
cd tflite
wget -q https://raw.githubusercontent.com/tensorflow/tensorflow/r1.12/tensorflow/contrib/lite/schema/schema.fbs
flatc --python schema.fbs

cat <<EOM >setup.py
import setuptools

setuptools.setup(
    name="tflite",
    version="0.0.1",
    author="google",
    author_email="google@google.com",
    description="TFLite",
    long_description="TFLite",
    long_description_content_type="text/markdown",
    url="https://www.tensorflow.org/lite",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
EOM

cat <<EOM >__init__.py
name = "tflite"
EOM

# Install tflite over python2 and python3
python3 setup.py install
python2 setup.py install

cd ..
rm -rf tflite
